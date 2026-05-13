"""Offline Preference Distillation for agent trajectory selection and DPO data.

The deployable ranker is a cheap best-of-N policy. The LlamaFactory export is the
main OPD path: GPT-5.4 supplies offline chosen/rejected preferences and Qwen is
trained with DPO/sigmoid, i.e. KL-regularized policy distillation against a
reference model:

  -log sigmoid(beta * [(log pi_theta(y+) - log pi_ref(y+))
                      - (log pi_theta(y-) - log pi_ref(y-))])

LlamaFactory's DPO trainer uses a reverse-KL f-divergence internally for this
reference-model path; ORPO/SimPO exports are only ablations.

For tool-using agents, the LlamaFactory export can also train preference pairs
over executable Qwen tool-call turns. The default stays on final-answer text
because it is the least disruptive option for the existing ReAct tool policy;
use `--lf-export-mode final_tool|action` for tool-call ablations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shlex
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv


REFUSAL_PHRASES = (
    "cannot find",
    "could not identify",
    "not enough information",
    "insufficient",
    "unable to",
    "i do not have",
    "i don't have",
    "no definitive",
    "cannot determine",
    "not found",
    "do not know",
)

FEATURE_NAMES = [
    "question_chars",
    "question_tokens",
    "answer_chars",
    "answer_tokens",
    "answer_question_token_ratio",
    "is_empty_answer",
    "is_refusal",
    "has_caveat",
    "steps",
    "tool_calls",
    "search_calls",
    "browse_calls",
    "wiki_calls",
    "final_answer_calls",
    "unique_tool_count",
    "stopped_final",
    "forced_finish_count",
    "has_reflection",
    "trajectory_events",
]

DISALLOWED_FEATURE_FIELDS = {"expected", "correct", "exact", "f1", "gold", "reference"}
_JSON_RE = re.compile(r"\{.*\}", re.S)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_run_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "runs.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(f"run JSONL not found: {candidate}")
    return candidate


def load_run_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = resolve_run_path(raw_path)
        source = str(path.parent if path.name == "runs.jsonl" else path)
        for i, record in enumerate(_read_jsonl(path)):
            item = dict(record)
            item["_source"] = source
            item["_source_index"] = i
            records.append(item)
    return records


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("id") is None:
            continue
        grouped[str(record["id"])].append(record)
    return dict(grouped)


def task_split(record: dict[str, Any]) -> str:
    task = str(record.get("task") or "unknown")
    split = str(record.get("split") or "unknown")
    return f"{task}:{split}"


def train_task_splits(records: list[dict[str, Any]]) -> list[str]:
    return sorted({task_split(record) for record in records})


def _tokens(text: str) -> list[str]:
    return text.split()


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def _tool_count(record: dict[str, Any], *names: str) -> int:
    counts = record.get("tool_call_counts") or {}
    total = 0
    for name, count in counts.items():
        if any(key in str(name) for key in names):
            try:
                total += int(count)
            except (TypeError, ValueError):
                pass
    return total


def extract_features(record: dict[str, Any]) -> np.ndarray:
    """Return inference-safe features; never use expected/correct/exact/f1."""
    if any(field in FEATURE_NAMES for field in DISALLOWED_FEATURE_FIELDS):
        raise AssertionError("gold-dependent feature name is not allowed")
    question = str(record.get("question") or "")
    answer = str(record.get("predicted") or record.get("answer") or "")
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    counts = record.get("tool_call_counts") or {}
    finish_reasons = record.get("finish_reasons") or []
    reflection = record.get("reflection")
    trajectory = record.get("trajectory") or []
    caveat = any(word in answer.lower() for word in ("maybe", "possibly", "likely", "uncertain", "not sure"))
    values = [
        len(question),
        len(q_tokens),
        len(answer),
        len(a_tokens),
        len(a_tokens) / max(1, len(q_tokens)),
        1.0 if not answer.strip() else 0.0,
        1.0 if _is_refusal(answer) else 0.0,
        1.0 if caveat else 0.0,
        float(record.get("steps") or 0),
        float(record.get("tool_calls") or 0),
        float(_tool_count(record, "search")),
        float(_tool_count(record, "browse")),
        float(_tool_count(record, "wiki")),
        float(_tool_count(record, "final_answer")),
        float(len(counts)),
        1.0 if record.get("stop_reason") == "final" else 0.0,
        float(sum(1 for reason in finish_reasons if str(reason).startswith("forced"))),
        1.0 if reflection else 0.0,
        float(len(trajectory)),
    ]
    return np.array(values, dtype=np.float64)


def score_record(record: dict[str, Any], model: dict[str, Any]) -> float:
    features = extract_features(record)
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    weights = np.array(model["weights"], dtype=np.float64)
    return float(np.dot(weights, (features - mean) / std) + float(model.get("bias", 0.0)))


def _record_cost(record: dict[str, Any]) -> float:
    return float(record.get("tool_calls") or 0) + 0.25 * float(record.get("steps") or 0)


def _correct(record: dict[str, Any]) -> bool:
    return bool(record.get("correct") or record.get("exact"))


def _f1(record: dict[str, Any]) -> float:
    try:
        return float(record.get("f1") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def gold_preferences(
    grouped: dict[str, list[dict[str, Any]]],
    min_f1_margin: float,
    min_cost_margin: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for example_id, candidates in grouped.items():
        if len(candidates) < 2:
            continue
        for i, left in enumerate(candidates):
            for right in candidates[i + 1:]:
                winner = loser = None
                if _correct(left) != _correct(right):
                    winner, loser = (left, right) if _correct(left) else (right, left)
                    reason = "correctness"
                elif _correct(left) and _correct(right):
                    cost_diff = _record_cost(left) - _record_cost(right)
                    if abs(cost_diff) >= min_cost_margin:
                        winner, loser = (left, right) if cost_diff < 0 else (right, left)
                        reason = "correct_cost"
                    else:
                        continue
                else:
                    f1_diff = _f1(left) - _f1(right)
                    if abs(f1_diff) >= min_f1_margin:
                        winner, loser = (left, right) if f1_diff > 0 else (right, left)
                        reason = "f1_margin"
                    else:
                        continue
                pairs.append(_pair_record(example_id, winner, loser, "gold", reason))
    return pairs


def _pair_record(example_id: str, winner: dict[str, Any], loser: dict[str, Any], expert: str, reason: str) -> dict[str, Any]:
    return {
        "id": example_id,
        "task": winner.get("task") or loser.get("task"),
        "split": winner.get("split") or loser.get("split"),
        "expert": expert,
        "reason": reason,
        "winner_source": winner.get("_source"),
        "loser_source": loser.get("_source"),
        "winner": winner,
        "loser": loser,
    }


def _candidate_digest(question: str, expected: str | None, candidates: list[dict[str, Any]], model: str, reference_mode: str) -> str:
    payload = {
        "question": question,
        "expected": expected if reference_mode == "include" else None,
        "model": model,
        "reference_mode": reference_mode,
        "candidates": [
            {
                "source": candidate.get("_source"),
                "predicted": candidate.get("predicted"),
                "steps": candidate.get("steps"),
                "tool_calls": candidate.get("tool_calls"),
                "stop_reason": candidate.get("stop_reason"),
            }
            for candidate in candidates
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("key"):
            cache[str(record["key"])] = record
    return cache


def _append_cache(path: Path | None, record: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _env_model() -> str:
    return (
        os.getenv("OPD_EXPERT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-5.4"
    )


def _env_base_url() -> str:
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_BASE_URL")
    if base_url:
        return base_url
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/") + "/openai/v1/"
    raise ValueError("Set OPENAI_BASE_URL or AZURE_OPENAI_BASE_URL in .env for GPT-5.4 OPD expert calls.")


def _env_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env for GPT-5.4 OPD expert calls.")
    return api_key


def _selector_prompt(
    question: str,
    expected: str | None,
    candidates: list[tuple[str, dict[str, Any]]],
    reference_mode: str,
    include_trajectory: bool,
) -> str:
    blocks = []
    for label, candidate in candidates:
        answer = str(candidate.get("predicted") or "").strip()
        snippet = ""
        if include_trajectory and candidate.get("trajectory"):
            snippet = "\nTrajectory excerpt:\n" + json.dumps(candidate["trajectory"][:8], ensure_ascii=False)[:3000]
        blocks.append(
            f"Candidate {label}\n"
            f"Answer: {answer}\n"
            f"Steps: {candidate.get('steps')} | Tool calls: {candidate.get('tool_calls')} | "
            f"Stop: {candidate.get('stop_reason')}\n"
            f"Tool counts: {json.dumps(candidate.get('tool_call_counts') or {}, ensure_ascii=False)}"
            f"{snippet}"
        )
    reference = f"\nReference answer for training only:\n{expected}\n" if reference_mode == "include" and expected else ""
    labels = "|".join(label for label, _ in candidates)
    return (
        "You are GPT-5.4 acting as an expert preference labeler for agent trajectory distillation.\n"
        "Choose the candidate whose final answer is most useful and correct for the question. "
        "Prefer exact, concise answers supported by the trajectory metadata. Penalize refusals, "
        "empty answers, irrelevant long explanations, and answers that do not address all constraints.\n\n"
        f"Question:\n{question}\n"
        f"{reference}\n"
        + "\n\n".join(blocks)
        + f"\n\nReturn only JSON: {{\"choice\":\"{labels}\", \"reason\":\"short reason\"}}."
    )


def _parse_choice(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    return json.loads(match.group(0) if match else text)


def _chat_create(client: Any, model: str, prompt: str, timeout_retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(timeout_retries + 1):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=256,
                )
            except TypeError:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"GPT-5.4 expert call failed: {type(last_error).__name__}: {last_error}")


def llm_preferences(
    grouped: dict[str, list[dict[str, Any]]],
    model: str,
    base_url: str,
    api_key: str,
    cache_path: Path | None,
    reference_mode: str,
    include_trajectory: bool,
    max_candidates: int,
    retries: int,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90)
    cache = _load_cache(cache_path)
    pairs: list[dict[str, Any]] = []
    for example_id in sorted(grouped):
        candidates = [candidate for candidate in grouped[example_id] if str(candidate.get("predicted") or "").strip()]
        if len(candidates) < 2:
            continue
        candidates = candidates[:max_candidates]
        question = str(candidates[0].get("question") or "")
        expected = str(candidates[0].get("expected") or "") or None
        key = _candidate_digest(question, expected, candidates, model, reference_mode)
        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        if key in cache:
            choice = str(cache[key].get("choice") or "").strip().upper()
            reason = str(cache[key].get("reason") or "cache")
        else:
            prompt = _selector_prompt(
                question,
                expected,
                list(zip(labels, candidates, strict=True)),
                reference_mode,
                include_trajectory,
            )
            obj = _parse_choice(_chat_create(client, model, prompt, retries))
            choice = str(obj.get("choice") or "").strip().upper()
            reason = str(obj.get("reason") or "gpt5.4_preference")
            cache_record = {"key": key, "id": example_id, "choice": choice, "reason": reason, "model": model}
            cache[key] = cache_record
            _append_cache(cache_path, cache_record)
        if choice not in labels:
            raise RuntimeError(f"GPT-5.4 returned invalid choice for {example_id}: {choice}")
        winner = candidates[labels.index(choice)]
        for loser in candidates:
            if loser is not winner:
                pairs.append(_pair_record(example_id, winner, loser, f"llm:{model}", reason))
    return pairs


def _stable_validation_id(example_id: str, val_fraction: float) -> bool:
    bucket = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction


def _standardize(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vstack([extract_features(record) for record in records])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _pair_matrix(pairs: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    deltas = []
    for pair in pairs:
        winner = (extract_features(pair["winner"]) - mean) / std
        loser = (extract_features(pair["loser"]) - mean) / std
        deltas.append(winner - loser)
    return np.vstack(deltas) if deltas else np.zeros((0, len(FEATURE_NAMES)))


def _pair_accuracy(deltas: np.ndarray, weights: np.ndarray) -> float:
    if len(deltas) == 0:
        return 0.0
    return float((deltas @ weights > 0).mean())


def train_ranker(
    pairs: list[dict[str, Any]],
    epochs: int,
    lr: float,
    l2: float,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not pairs:
        raise ValueError("No preference pairs available for OPD training.")
    all_records = [pair["winner"] for pair in pairs] + [pair["loser"] for pair in pairs]
    mean, std = _standardize(all_records)
    train_pairs = [pair for pair in pairs if not _stable_validation_id(str(pair["id"]), val_fraction)]
    val_pairs = [pair for pair in pairs if _stable_validation_id(str(pair["id"]), val_fraction)]
    if not train_pairs:
        train_pairs, val_pairs = pairs, []
    train_x = _pair_matrix(train_pairs, mean, std)
    val_x = _pair_matrix(val_pairs, mean, std)
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for index in order:
            delta = train_x[index]
            logit = float(np.dot(weights, delta))
            prob = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
            grad = -(1.0 - prob) * delta + l2 * weights
            weights -= lr * grad
    metrics = {
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "train_pair_accuracy": _pair_accuracy(train_x, weights),
        "validation_pair_accuracy": _pair_accuracy(val_x, weights),
    }
    model = {
        "kind": "sii-agent-opd-ranker",
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "bias": 0.0,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return model, metrics


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sft_records(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        winners.setdefault(str(pair["id"]), pair["winner"])
    records = []
    for example_id, record in sorted(winners.items()):
        records.append(
            {
                "id": example_id,
                "task": record.get("task"),
                "split": record.get("split"),
                "source": record.get("_source"),
                "messages": [
                    {"role": "user", "content": str(record.get("question") or "")},
                    {"role": "assistant", "content": str(record.get("predicted") or "")},
                ],
                "trajectory": record.get("trajectory"),
            }
        )
    return records


def _llamafactory_prompt(record: dict[str, Any]) -> str:
    return (
        "Answer the question using the SII-Agent tool-based QA policy. "
        "Return a concise final answer.\n\n"
        f"Question: {str(record.get('question') or '').strip()}"
    )


def _tool_specs_for_record(record: dict[str, Any]) -> str:
    from tools import tool_specs

    task = str(record.get("task") or "").lower()
    if task == "browsecomp-plus":
        names = ("browsecomp_search", "browsecomp_get_document", "final_answer")
    else:
        names = ("web_search", "wiki_search", "browse", "final_answer")
    return json.dumps(tool_specs(names), ensure_ascii=False)


def _parse_tool_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _function_call_value(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps([{"name": name, "arguments": arguments}], ensure_ascii=False)


def _assistant_response_from_event(event: dict[str, Any]) -> dict[str, str] | None:
    tool_calls = event.get("tool_calls") or []
    calls = []
    for tool_call in tool_calls:
        fn = (tool_call or {}).get("function") or {}
        name = str(fn.get("name") or "").strip()
        arguments = _parse_tool_arguments(fn.get("arguments"))
        if not name or arguments is None:
            return None
        calls.append({"name": name, "arguments": arguments})
    if calls:
        return {"from": "function_call", "value": json.dumps(calls, ensure_ascii=False)}

    content = str(event.get("content") or "").strip()
    if content:
        return {"from": "gpt", "value": content}
    return None


def _trajectory_decisions(record: dict[str, Any]) -> list[tuple[list[dict[str, str]], dict[str, str]]]:
    decisions: list[tuple[list[dict[str, str]], dict[str, str]]] = []
    history: list[dict[str, str]] = [{"from": "human", "value": _llamafactory_prompt(record)}]
    for event in record.get("trajectory") or []:
        role = event.get("role")
        if role == "assistant":
            response = _assistant_response_from_event(event)
            if response is None:
                continue
            if len(history) % 2 == 1:
                decisions.append(([dict(message) for message in history], response))
            history.append(response)
        elif role == "tool":
            content = str(event.get("content") or "").strip()
            if content and len(history) % 2 == 0:
                history.append({"from": "observation", "value": content[:4000]})
    return decisions


def _final_answer_response(record: dict[str, Any]) -> dict[str, str]:
    return {
        "from": "function_call",
        "value": _function_call_value(
            "final_answer",
            {
                "answer": str(record.get("predicted") or record.get("answer") or "").strip(),
                "rationale": "Selected by GPT-5.4 OPD preference labeling.",
            },
        ),
    }


def _action_pair_messages(pair: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]] | None:
    winner_decisions = _trajectory_decisions(pair["winner"])
    loser_decisions = _trajectory_decisions(pair["loser"])
    for (winner_prompt, winner_response), (loser_prompt, loser_response) in zip(
        winner_decisions, loser_decisions, strict=False
    ):
        if winner_prompt != loser_prompt:
            break
        if winner_response != loser_response:
            return winner_prompt, winner_response, loser_response
    return None


def _llamafactory_pair(pair: dict[str, Any], export_mode: str) -> dict[str, Any]:
    winner = pair["winner"]
    loser = pair["loser"]
    tools = _tool_specs_for_record(winner)
    metadata = {
        "id": pair.get("id"),
        "task": pair.get("task"),
        "split": pair.get("split"),
        "expert": pair.get("expert"),
        "reason": pair.get("reason"),
        "winner_source": pair.get("winner_source"),
        "loser_source": pair.get("loser_source"),
        "export_mode": export_mode,
    }

    if export_mode == "answer":
        return {
            "conversations": [
                {
                    "from": "human",
                    "value": _llamafactory_prompt(winner),
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": str(winner.get("predicted") or "").strip(),
            },
            "rejected": {
                "from": "gpt",
                "value": str(loser.get("predicted") or "").strip(),
            },
            "tools": "[]",
            "metadata": metadata,
        }

    action_pair = _action_pair_messages(pair) if export_mode == "action" else None
    if action_pair is None:
        conversations = [{"from": "human", "value": _llamafactory_prompt(winner)}]
        chosen = _final_answer_response(winner)
        rejected = _final_answer_response(loser)
        metadata["export_mode"] = "final_tool" if export_mode == "final_tool" else "action_fallback_final_tool"
    else:
        conversations, chosen, rejected = action_pair

    return {
        "conversations": conversations,
        "chosen": chosen,
        "rejected": rejected,
        "tools": tools,
        "metadata": metadata,
    }


def _write_llamafactory_artifacts(
    pairs: list[dict[str, Any]],
    out_dir: Path,
    dataset_name: str,
    model_name_or_path: str,
    pref_loss: str,
    run_name: str,
    pref_beta: float,
    lora_rank: int,
    learning_rate: str,
    num_train_epochs: float,
    gradient_accumulation_steps: int,
    cutoff_len: int,
    export_mode: str,
    template: str,
) -> dict[str, str]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{dataset_name}.json"
    data = [_llamafactory_pair(pair, export_mode) for pair in pairs]
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        dataset_name: {
            "file_name": data_path.name,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "tools": "tools",
            },
        }
    }
    dataset_info_path = out_dir / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    use_ref_model = pref_loss not in {"orpo", "simpo"}
    repo_root = Path(__file__).resolve().parents[1]
    safe_run_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name)
    output_dir = (repo_root / "saves" / "qwen35-9b" / "lora" / f"{safe_run_name}_{pref_loss}").resolve()
    config = {
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_alpha": 2 * lora_rank,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "pref_beta": pref_beta,
        "pref_loss": pref_loss,
        "dataset_dir": str(out_dir),
        "dataset": dataset_name,
        "template": template,
        "cutoff_len": cutoff_len,
        "max_samples": 100000,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        "output_dir": str(output_dir),
        "logging_steps": 5,
        "save_steps": 200,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": float(learning_rate),
        "num_train_epochs": num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
    }
    config_path = out_dir / f"qwen35_9b_lora_opd_{pref_loss}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    quoted_repo_root = shlex.quote(str(repo_root))
    quoted_config_path = shlex.quote(str(config_path))
    script = f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -n "${{SII_AGENT_ROOT:-}}" ]; then
  REPO_ROOT="$SII_AGENT_ROOT"
else
  REPO_ROOT={quoted_repo_root}
fi
if [ ! -d "$REPO_ROOT/.git" ]; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)"
fi
LLAMAFACTORY_DIR="${{LLAMAFACTORY_DIR:-$REPO_ROOT/third_party/LLaMA-Factory}}"
LF_PYTHON="${{LF_PYTHON:-/root/.venv/bin/python}}"
if [ ! -x "$LF_PYTHON" ]; then
  LF_PYTHON="$(command -v python)"
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  if [ ! -d "$LLAMAFACTORY_DIR/.git" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory "$LLAMAFACTORY_DIR"
  fi
  "$LF_PYTHON" -m pip install -e "$LLAMAFACTORY_DIR"
  export PATH="$(dirname "$LF_PYTHON"):$PATH"
fi

cd "$LLAMAFACTORY_DIR"
CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-4,5}}" FORCE_TORCHRUN=1 \\
  llamafactory-cli train {quoted_config_path}
"""
    script_path = out_dir / "train_llamafactory_opd.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)

    return {
        "llamafactory_dataset": str(data_path),
        "llamafactory_dataset_info": str(dataset_info_path),
        "llamafactory_config": str(config_path),
        "llamafactory_train_script": str(script_path),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_run_records(args.runs)
    grouped = {key: value for key, value in group_records(records).items() if len(value) >= 2}
    if args.expert == "llm":
        model_name = args.model or _env_model()
        pairs = llm_preferences(
            grouped,
            model_name,
            args.base_url or _env_base_url(),
            args.api_key or _env_api_key(),
            Path(args.cache) if args.cache else out / "gpt54_preference_cache.jsonl",
            args.reference_mode,
            args.include_trajectory_for_expert,
            args.max_candidates,
            args.retries,
        )
        expert_name = f"llm:{model_name}"
    else:
        pairs = gold_preferences(grouped, args.min_f1_margin, args.min_cost_margin)
        expert_name = "gold"

    preferences_path = Path(args.preferences_out) if args.preferences_out else out / "preferences.jsonl"
    _write_jsonl(preferences_path, pairs)
    model, metrics = train_ranker(pairs, args.epochs, args.lr, args.l2, args.val_fraction, args.seed)
    model["metadata"] = {
        "expert": expert_name,
        "reference_mode": args.reference_mode if args.expert == "llm" else None,
        "train_sources": [str(resolve_run_path(path)) for path in args.runs],
        "train_task_splits": train_task_splits(records),
        "num_records": len(records),
        "num_candidate_groups": len(grouped),
        "num_preference_pairs": len(pairs),
        "metrics": metrics,
    }
    model_path = out / "opd_policy.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    sft_path = Path(args.sft_out) if args.sft_out else out / "winning_sft.jsonl"
    _write_jsonl(sft_path, _sft_records(pairs))
    llamafactory_artifacts = _write_llamafactory_artifacts(
        pairs,
        Path(args.llamafactory_out) if args.llamafactory_out else out / "llamafactory",
        args.llamafactory_dataset_name,
        args.model_name_or_path,
        args.pref_loss,
        args.llamafactory_run_name or Path(args.out).name,
        args.lf_pref_beta,
        args.lf_lora_rank,
        args.lf_learning_rate,
        args.lf_num_train_epochs,
        args.lf_gradient_accumulation_steps,
        args.lf_cutoff_len,
        args.lf_export_mode,
        args.lf_template,
    )
    summary = {
        "model_path": str(model_path),
        "preferences_path": str(preferences_path),
        "sft_path": str(sft_path),
        **llamafactory_artifacts,
        **model["metadata"],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OPD trajectory ranker from agent run JSONL files.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or runs.jsonl files. Need at least two variants with shared ids.")
    parser.add_argument("--out", required=True, help="Output directory for policy, preferences, and SFT data.")
    parser.add_argument("--expert", choices=["llm", "gold"], default="llm", help="Preference label source. Default uses GPT-5.4 from .env.")
    parser.add_argument("--model", default=None, help="Expert model. Defaults to OPD_EXPERT_MODEL/OPENAI_MODEL/AZURE_OPENAI_DEPLOYMENT/gpt-5.4.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL/AZURE_OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key. Prefer .env; this is only for controlled local runs.")
    parser.add_argument("--reference-mode", choices=["include", "hide"], default="include", help="Whether GPT-5.4 sees the train reference answer while labeling preferences.")
    parser.add_argument("--include-trajectory-for-expert", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--preferences-out", default=None)
    parser.add_argument("--sft-out", default=None)
    parser.add_argument("--llamafactory-out", default=None, help="Directory for LlamaFactory ranking dataset/config/script.")
    parser.add_argument("--llamafactory-dataset-name", default="sii_agent_opd_dpo")
    parser.add_argument("--llamafactory-run-name", default=None, help="Suffix for LlamaFactory output_dir under saves/qwen35-9b/lora.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pref-loss", choices=["sigmoid", "orpo", "simpo"], default="sigmoid", help="LlamaFactory preference loss. sigmoid is DPO; orpo/simpo are reference-free.")
    parser.add_argument("--lf-pref-beta", type=float, default=0.1)
    parser.add_argument("--lf-lora-rank", type=int, default=16)
    parser.add_argument("--lf-learning-rate", default="5.0e-6")
    parser.add_argument("--lf-num-train-epochs", type=float, default=1.0)
    parser.add_argument("--lf-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lf-cutoff-len", type=int, default=4096)
    parser.add_argument(
        "--lf-export-mode",
        choices=["answer", "final_tool", "action"],
        default="answer",
        help="LlamaFactory DPO target: legacy plain answers, final_answer tool calls, or first divergent trajectory action.",
    )
    parser.add_argument(
        "--lf-template",
        default="qwen3_5_nothink",
        help="LlamaFactory chat template. qwen3_5_nothink matches Qwen3.5 tool-call formatting.",
    )
    parser.add_argument("--min-f1-margin", type=float, default=0.1)
    parser.add_argument("--min-cost-margin", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
