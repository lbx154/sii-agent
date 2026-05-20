"""Export benchmark_answered.csv rows as SII-Agent slime prompts."""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)

TEXT_TOOLS = [
    "web_search",
    "wiki_search",
    "wiki_page",
    "browsecomp_search",
    "browsecomp_open",
    "browser_open",
    "browser_open_many",
    "final_answer",
]
VISUAL_TOOLS = [
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "reverse_image_search",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="/root/sii-agent/data/benchmark_answered.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--n", type=int, default=0, help="Number of rows; 0 means all selected rows.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--indices", default="", help="Comma/space-separated original CSV row indices; overrides n/offset.")
    parser.add_argument(
        "--allowed-tools",
        default=None,
        help="Comma-separated tool override for all rows. final_answer is appended if missing.",
    )
    parser.add_argument(
        "--chat-prompt",
        action="store_true",
        help="Write question as a chat message list for slime processor-backed models.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Exclude browser_open/browser_open_many from default tools.")
    return parser.parse_args()


def _split_tools(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tools = [item.strip() for item in raw.split(",") if item.strip()]
    if "final_answer" not in tools:
        tools.append("final_answer")
    return list(dict.fromkeys(tools))


def _parse_indices(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    indices: list[int] = []
    seen: set[int] = set()
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part:
            continue
        index = int(part)
        if index < 0:
            raise ValueError("--indices values must be non-negative")
        if index not in seen:
            indices.append(index)
            seen.add(index)
    return indices


def _select_rows(rows: list[dict[str, str]], n: int, offset: int, indices: list[int] | None) -> list[tuple[int, dict[str, str]]]:
    if indices is not None:
        selected = []
        for index in indices:
            if index >= len(rows):
                raise IndexError(f"--indices value {index} is outside CSV row range 0..{len(rows) - 1}")
            selected.append((index, rows[index]))
        return selected
    limit = len(rows) if n <= 0 else n
    return [(idx, row) for idx, row in enumerate(rows[offset:], start=offset)][:limit]


def _guess_image_ext(mime: str | None, data: bytes) -> str:
    if mime:
        guessed = mimetypes.guess_extension(mime.split(";", 1)[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _decode_image_payload(image: str) -> tuple[bytes, str | None, str]:
    value = "".join(image.strip().split())
    if value.startswith("data:image/"):
        header, sep, encoded = value.partition(",")
        if not sep or not encoded:
            raise ValueError("data image URL is missing a base64 payload")
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/png"
        return base64.b64decode(encoded, validate=True), mime, "data_url"
    return base64.b64decode(value, validate=True), None, "raw_base64"


def _materialize_image(image: str, idx: int, image_dir: Path) -> tuple[str, dict[str, Any]]:
    image = image.strip()
    if not image:
        return "", {"kind": "none", "original_chars": 0}
    if image.startswith(("http://", "https://")):
        return image, {"kind": "url", "original_chars": len(image), "source": image}
    if len(image) < 4096:
        try:
            path = Path(image).expanduser()
            if path.exists() and path.is_file():
                return str(path.resolve()), {"kind": "path", "original_chars": len(image), "source": str(path.resolve())}
        except OSError:
            pass
    try:
        data, mime, kind = _decode_image_payload(image)
    except (binascii.Error, ValueError) as exc:
        return "", {"kind": "invalid_image_payload", "original_chars": len(image), "error": f"{type(exc).__name__}: {exc}"}
    digest = hashlib.sha1(data).hexdigest()[:12]
    ext = _guess_image_ext(mime, data)
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"benchmark-csv-{idx}-{digest}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return str(path.resolve()), {"kind": kind, "original_chars": len(image), "bytes": len(data), "mime": mime, "path": str(path)}


def _default_tools(has_image: bool, no_browser: bool) -> list[str]:
    text_tools = [tool for tool in TEXT_TOOLS if not no_browser or not tool.startswith("browser_")]
    if has_image:
        return VISUAL_TOOLS + text_tools
    return text_tools


def _build_question(problem: str, image_ref: str) -> str:
    problem = " ".join(str(problem or "").split())
    parts = [
        "You are answering a benchmark factual research question.",
        f"Question: {problem}",
    ]
    if image_ref:
        parts.extend(
            [
                f"Image path: {image_ref}",
                "Use visual/OCR search tools first when the image is needed, then verify names, dates, places, counts, and other external facts with search or browser evidence.",
            ]
        )
    parts.append(
        "Use search/retrieval and browser tools as needed. Do not use memory, shell, or non-listed tools. "
        "When done, call `final_answer` with only the concise answer span."
    )
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out = Path(args.out)
    image_dir = Path(args.image_dir) if args.image_dir else out.parent / "benchmark_images"
    out.parent.mkdir(parents=True, exist_ok=True)
    override_tools = _split_tools(args.allowed_tools)

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    selected = _select_rows(rows, args.n, args.offset, _parse_indices(args.indices))
    counts = {"rows": 0, "image_rows": 0, "invalid_images": 0}
    with out.open("w", encoding="utf-8") as f:
        for idx, row in selected:
            image_ref, image_meta = _materialize_image(str(row.get("image") or ""), idx, image_dir)
            if image_ref:
                counts["image_rows"] += 1
            elif image_meta.get("kind") == "invalid_image_payload":
                counts["invalid_images"] += 1
            allowed_tools = override_tools or _default_tools(bool(image_ref), args.no_browser)
            metadata: dict[str, Any] = {
                "id": f"benchmark-csv-{idx}",
                "task": "benchmark_csv",
                "split": "test",
                "row_index": idx,
                "allowed_tools": allowed_tools,
                "image_meta": image_meta,
            }
            if image_ref:
                metadata["image"] = image_ref
            record = {
                "question": (
                    [{"role": "user", "content": _build_question(str(row.get("problem") or ""), image_ref)}]
                    if args.chat_prompt
                    else _build_question(str(row.get("problem") or ""), image_ref)
                ),
                "answer": str(row.get("answer") or "").strip(),
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["rows"] += 1
    print(json.dumps({"out": str(out), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
