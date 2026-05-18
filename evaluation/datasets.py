"""Dataset loaders → uniform list of {id, question, answer}."""
from __future__ import annotations
import base64
import hashlib
import os
import random
import re
from pathlib import Path
from typing import Iterator

DEFAULT_SPLITS = {
    "simpleqa": "test",
    "simplevqa": "test",
    "2wiki": "validation",
    "browsecomp-plus": "test",
}
AVAILABLE_SPLITS = {
    "simpleqa": ("test", "few_shot"),
    "simplevqa": ("test",),
    "2wiki": ("train", "validation", "test"),
    "browsecomp-plus": ("test",),
}


def load_simpleqa(n: int | None = None, split: str = "test") -> Iterator[dict]:
    from datasets import load_dataset
    if split == "train":
        raise ValueError("SimpleQA has no public train split in the configured loaders; use 2Wiki train or SimpleQA few_shot/test with an explicit held-out carve-out.")
    if split not in AVAILABLE_SPLITS["simpleqa"]:
        raise ValueError(f"Unsupported SimpleQA split '{split}'. Available: {AVAILABLE_SPLITS['simpleqa']}")

    if split == "test":
        # Tiny / canonical SimpleQA (text). Fallback to OpenAI's release if available.
        try:
            ds = load_dataset("basicv8vc/SimpleQA", split=split)
            key_q, key_a = "problem", "answer"
        except Exception:
            ds = load_dataset("lighteval/SimpleQA", split=split)
            key_q, key_a = "question", "answer"
    else:
        ds = load_dataset("lighteval/SimpleQA", split=split)
        key_q = "question" if "question" in ds.column_names else "problem"
        key_a = "answer"
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        yield {"id": f"simpleqa-{split}-{i}", "task": "simpleqa", "split": split, "question": ex[key_q], "answer": ex[key_a]}


def _simplevqa_cache_dir() -> Path:
    return Path(os.getenv("SIMPLEVQA_IMAGE_CACHE", "logs/simplevqa_images"))


def _image_extension(data: bytes) -> str:
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return "img"


def _materialize_simplevqa_image(image: object, data_id: object) -> str:
    cache = _simplevqa_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    if isinstance(image, str):
        data = base64.b64decode("".join(image.split()), validate=False)
        ext = _image_extension(data)
        digest = hashlib.sha1(data).hexdigest()[:12]
        path = cache / f"{data_id}_{digest}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        return str(path.resolve())

    if isinstance(image, bytes):
        data = image
        ext = _image_extension(data)
        digest = hashlib.sha1(data).hexdigest()[:12]
        path = cache / f"{data_id}_{digest}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        return str(path.resolve())

    if hasattr(image, "save"):
        path = cache / f"{data_id}.png"
        if not path.exists():
            image.save(path)
        return str(path.resolve())

    raise TypeError(f"Unsupported SimpleVQA image type: {type(image).__name__}")


def load_simplevqa(n: int | None = None, split: str = "test") -> Iterator[dict]:
    from datasets import load_dataset
    if split not in AVAILABLE_SPLITS["simplevqa"]:
        raise ValueError(f"Unsupported SimpleVQA split '{split}'. Available: {AVAILABLE_SPLITS['simplevqa']}")

    ds = load_dataset("m-a-p/SimpleVQA", split=split)
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        data_id = ex.get("data_id", i)
        image_path = _materialize_simplevqa_image(ex["image"], data_id)
        question = (
            "You are answering a SimpleVQA visual factuality question.\n"
            f"Image path: {image_path}\n"
            f"Question: {ex['question']}\n\n"
            "If `visual_web_search` is available, call it first with the image path and exact question. "
            "Otherwise use `image_to_text` on the image path first. Do not treat the first visual guess "
            "as proven: compare multiple candidates, OCR/visible clues, and search evidence before answering. "
            "If the question is Chinese, use the common Chinese answer form rather than an English alias when possible. "
            "When done, call `final_answer` with only the concise answer phrase."
        )
        yield {
            "id": f"simplevqa-{split}-{data_id}",
            "task": "simplevqa",
            "split": split,
            "question": question,
            "answer": ex["answer"],
            "language": ex.get("language"),
            "original_category": ex.get("original_category"),
            "vqa_category": ex.get("vqa_category"),
        }


_CONTEXT_STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "film", "song", "did",
    "does", "was", "were", "who", "what", "when", "where", "which", "whose",
    "both", "have", "has", "his", "her", "its", "they", "their", "came", "out",
}


def _context_tokens(text: str) -> set[str]:
    return {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9\u00C0-\u024F]+", text or "")
        if len(tok) > 2 and tok.lower() not in _CONTEXT_STOPWORDS
    }


def _normalize_title(text: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", text or "")
    value = re.sub(r"[^A-Za-z0-9\u00C0-\u024F]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def _format_2wiki_context(context: object, question: str = "", max_chars: int = 12000) -> str:
    if not isinstance(context, dict):
        return ""
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    q_norm = _normalize_title(question)
    q_tokens = _context_tokens(question)
    q_lower = question.lower()
    blocks: list[dict] = []
    for title, sents in zip(titles, sentences):
        if isinstance(sents, list):
            text = " ".join(str(sent).strip() for sent in sents if str(sent).strip())
        else:
            text = str(sents).strip()
        if not text:
            continue
        title_s = str(title)
        title_norm = _normalize_title(title_s)
        title_tokens = _context_tokens(title_s)
        text_tokens = _context_tokens(text[:900])
        score = 0
        if title_norm and title_norm in q_norm:
            score += 90
        if title_tokens:
            score += 10 * len(q_tokens & title_tokens)
        score += len(q_tokens & text_tokens)
        blocks.append(
            {
                "title": title_s,
                "title_norm": title_norm,
                "text": text,
                "score": score,
                "index": len(blocks),
            }
        )

    # If a question-matching paragraph names another context title, boost that
    # second-hop title. This keeps 2Wiki bridge evidence near the top without
    # using supporting_facts or gold answers.
    strong_blocks = [b for b in blocks if b["score"] >= 8]
    for block in blocks:
        title_norm = block["title_norm"]
        if not title_norm:
            continue
        title_tokens = _context_tokens(str(block["title"]))
        for source in strong_blocks:
            if source is block:
                continue
            source_norm = _normalize_title(str(source["text"]))
            source_tokens = _context_tokens(str(source["text"])[:1400])
            if title_norm in source_norm or len(title_tokens & source_tokens) >= min(2, len(title_tokens)):
                block["score"] += 45
                break

    blocks.sort(key=lambda b: (-int(b["score"]), int(b["index"])))
    rendered = "\n\n".join(f"Title: {b['title']}\n{b['text']}" for b in blocks)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars].rstrip() + " ..."


def load_2wiki(
    n: int | None = None,
    split: str = "validation",
    offset: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[dict]:
    from datasets import load_dataset
    if split not in AVAILABLE_SPLITS["2wiki"]:
        raise ValueError(f"Unsupported 2Wiki split '{split}'. Available: {AVAILABLE_SPLITS['2wiki']}")
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if offset or n is not None:
        end = len(ds) if n is None else min(len(ds), offset + n)
        ds = ds.select(range(min(offset, len(ds)), end))
    id_prefix = f"2wiki-{split}-seed{seed}" if shuffle else f"2wiki-{split}"
    for i, ex in enumerate(ds):
        ans = ex.get("answer") or (ex.get("answers", {}) or {}).get("text", [None])[0]
        context = _format_2wiki_context(ex.get("context"), question=str(ex["question"]))
        question = (
            "You are answering a factual question using provided context and tools.\n"
            f"Question: {ex['question']}\n\n"
            "Use the provided context as primary evidence when it is sufficient. "
            "If the context is insufficient or ambiguous, use available retrieval tools to fill the missing fact. "
            "Do not over-search: once the answer is supported, call `final_answer` with a concise answer phrase.\n"
            "General answer-format rules:\n"
            "- Return only the final answer span, not a sentence or explanation.\n"
            "- For yes/no questions, answer exactly `yes` or `no` in lowercase.\n"
            "- Preserve the granularity supported by evidence for names, places, dates, causes, countries, and nationalities."
            f"\n\nProvided context:\n{context}"
        )
        yield {"id": f"{id_prefix}-{offset + i}", "task": "2wiki", "split": split, "question": question, "answer": ans}


def load_browsecomp_plus(n: int | None = None, split: str = "test") -> Iterator[dict]:
    if split not in AVAILABLE_SPLITS["browsecomp-plus"]:
        raise ValueError(f"Unsupported BrowseComp-Plus split '{split}'. Available: {AVAILABLE_SPLITS['browsecomp-plus']}")

    from evaluation.run_browsecomp import DEFAULT_CANARY, load_examples as load_browsecomp_examples

    rows = load_browsecomp_examples(source=None, n=0 if n is None else n, offset=0, canary=DEFAULT_CANARY)
    for i, ex in enumerate(rows):
        evidence_docids = [
            str(doc.get("docid"))
            for doc in ex.get("evidence_docs", [])
            if isinstance(doc, dict) and doc.get("docid") is not None
        ]
        yield {
            "id": f"browsecomp-{ex['query_id']}",
            "task": "browsecomp-plus",
            "split": split,
            "query_id": str(ex["query_id"]),
            "question": ex["query"],
            "answer": ex.get("answer"),
            "evidence_docids": evidence_docids,
        }


LOADERS = {
    "simpleqa": load_simpleqa,
    "simplevqa": load_simplevqa,
    "2wiki": load_2wiki,
    "browsecomp-plus": load_browsecomp_plus,
}


def load_examples(
    task: str,
    n: int | None = None,
    offset: int = 0,
    split: str | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict]:
    split = split or DEFAULT_SPLITS[task]
    if task == "2wiki":
        return list(load_2wiki(n, split=split, offset=offset, shuffle=shuffle, seed=seed))
    if shuffle:
        rows = list(LOADERS[task](None, split=split))
        random.Random(seed).shuffle(rows)
        return rows[offset:] if n is None else rows[offset:offset + n]
    limit = n + offset if n is not None else None
    return list(LOADERS[task](limit, split=split))[offset:]
