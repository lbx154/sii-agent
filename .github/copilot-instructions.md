# Copilot instructions for SII-Agent

## Commands

Install/runtime setup:

```bash
pip install -r requirements.txt
cp configs/.env.example .env
python -m scripts.build_wiki_fts --source data/wiki25/wiki25_sample.jsonl --out data/wiki25/wiki25_fts.sqlite
python -m scripts.download_browsecomp_index --out indexes
```

Fast validation:

```bash
python -m scripts.smoke
```

Run a single benchmark sample by using `--n 1` plus `--offset`:

```bash
LLM_BACKEND=vllm VLLM_BASE_URL=http://127.0.0.1:8004/v1 VLLM_MODEL=sii-opd-v13-merged-sglang VLLM_ENABLE_THINKING=0 \
python -m evaluation.run_eval --task 2wiki --split validation --mode baseline --n 1 --offset 0 --max-llm-tokens 12000 --out logs/eval_smoke
```

Common 2Wiki baseline/evolved comparison:

```bash
python -m evaluation.run_eval --task 2wiki --split validation --mode baseline --n 500 --concurrency 128 --max-llm-tokens 12000 --out logs/opd_eval
python -m evaluation.run_eval --task 2wiki --split validation --mode evolved --n 500 --concurrency 128 --evolve-batch-size 128 --max-llm-tokens 12000 --out logs/opd_eval
```

Qwen3-32B semantic judging for completed 2Wiki runs:

```bash
python -m evaluation.judge_semantic \
  --run-dirs logs/opd_eval/<run-root>/2wiki_baseline_<ts> logs/opd_eval/<run-root>/2wiki_evolved_<ts> \
  --base-url http://127.0.0.1:8005/v1 \
  --model Qwen3-32B \
  --concurrency 64 \
  --out-prefix semantic_judge_qwen32
```

BrowseComp-Plus:

```bash
python -m evaluation.run_browsecomp --mode evolved --n 0 --concurrency 32
python -m evaluation.merge_browsecomp_runs --run-dirs <run-a> <run-b> --out <merged>
python -m evaluation.route_browsecomp_runs --primary-run <cheap-run> --fallback-run <selector-run> --out <routed>
```

SGLang serving for the full merged OPD model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/sglang-venv/bin/python -m sglang.launch_server \
  --model-path /root/sii-agent/saves/qwen35-9b/merged/v13_step_final_opd_32k_full_lora_manual_merged \
  --port 8004 --tp-size 4 --mem-fraction-static 0.8 --context-length 262144 \
  --served-model-name sii-opd-v13-merged-sglang \
  --trust-remote-code --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton --mm-attention-backend sdpa
```

There is no project-level pytest/ruff/mypy configuration in the root repository. Use smoke runs and targeted `evaluation.run_eval --n 1` or small `--n` slices for validation.

## Architecture

The main flow is:

```text
evaluation/run_eval.py
  -> evaluation/datasets.py
  -> agent/runner.py
  -> agent/react.py
  -> agent/llm.py + tools/registry.py dispatch
  -> agent/scoring.py + memory/store.py
```

- `agent/react.py` is the ReAct loop. It builds the system prompt, chooses the active tool profile, calls the LLM with OpenAI-style tool specs, parses both native OpenAI `tool_calls` and Qwen textual `<tool_call><function=...>` output, dispatches tools, and forces `final_answer` near the step limit.
- `agent/runner.py` wraps ReAct into `run_baseline` and `run_evolved`. It owns 2Wiki answer postprocessing, reflection gating, retry selection, and `RunOutcome` attribution fields (`first_result`, `retry_result`, `selected_attempt`, etc.).
- `agent/llm.py` is the backend switch. `LLM_BACKEND=azure` uses Azure OpenAI/AAD; `LLM_BACKEND=vllm` is also used for SGLang because both expose OpenAI-compatible `/v1`.
- `tools/registry.py` registers tools by import side effect. The default benchmark profile is `web_search,wiki_search,wiki_page,browser_open,browser_open_many,final_answer`; visual/rich/all profiles expose additional image/browser tools.
- `tools/wiki.py` is offline wiki25 search/page lookup. Prefer the SQLite FTS index at `data/wiki25/wiki25_fts.sqlite`; the JSONL BM25 path is fallback.
- `tools/search.py` sends `web_search` and `reverse_image_search` through the harness search-proxy; offline Wiki remains in `tools/wiki.py`.
- `evaluation/datasets.py` normalizes task loaders. 2Wiki examples embed a relevance-ranked `Provided context` and strict answer-format rules; SimpleVQA materializes images under `logs/simplevqa_images`.
- `memory/store.py` persists evolved runs as JSONL under the run memory root. For 2Wiki, seeded lessons/skills/policies exist but lesson injection, typed policies, and skills are opt-in.
- `training/opd.py` exports offline preference data and LlamaFactory DPO configs; `training/slime_sii_rollout.py` integrates on-policy tool-use rollouts with slime.

## Repository-specific conventions

- For SGLang/Qwen3.5, keep `VLLM_ENABLE_THINKING=0` for agent/eval runs. The client sends `chat_template_kwargs.enable_thinking=false`; otherwise responses can spend tokens in `reasoning_content` and break tool/final-answer parsing.
- Use `--tool-call-parser qwen3_coder` for Qwen3.5 SGLang serving. `hermes` can intermittently reject Qwen textual tool-call output with server-side 400s.
- Full OPD v13 LoRA is not loaded through SGLang runtime LoRA. The complete adapter targets Qwen3.5 linear-attention modules (`in_proj_a/b/qkv/z`, `out_proj`); serve the manually merged HF checkpoint instead of the pruned runtime adapter when comparing performance.
- 2Wiki safe defaults keep reflection/lessons/skills/typed policies disabled:

```bash
unset SII_2WIKI_ENABLE_REFLECTION
unset SII_2WIKI_ENABLE_LESSONS
unset SII_2WIKI_ENABLE_SKILLS
unset SII_2WIKI_ENABLE_TYPED_POLICIES
```

- Enable legacy 2Wiki lesson ablations explicitly with `SII_2WIKI_ENABLE_REFLECTION=1` and `SII_2WIKI_ENABLE_LESSONS=1`. Keep `SII_2WIKI_ENABLE_SKILLS` unset unless testing the experimental skill path.
- Do not use gold answers in test-time memory. `--gold-reflection` is for controlled experiments only; normal evolved runs should rely on self-reflection without expected-answer leakage.
- For 2Wiki, local scoring is exact/F1 based (`agent/scoring.py`, correct if exact or F1 >= 0.9). Qwen32 semantic judging, when available, is separate and should not be confused with local `summary.json` accuracy.
- `--save-traces` writes full trajectories to `runs.jsonl`; use it for parser/tool/debug investigations but avoid it for routine large sweeps unless traces are needed.
- Avoid treating generated artifacts as source: `logs/`, `saves/`, model directories (`Qwen3-32B/`, `Qwen3.5-9B/`), indexes, and downloaded third-party trees are large runtime assets.
- Third-party repos under `third_party/` have their own conventions; do not apply their CLAUDE/agent guidance to this root project unless editing inside those vendored directories.

## Current project context

- The main optimization target is 2WikiMultihopQA with Qwen3.5-9B + OPD v13 behavior, comparing `baseline` vs `evolved`.
- The most important recent serving path is the full manual LoRA merge at `saves/qwen35-9b/merged/v13_step_final_opd_32k_full_lora_manual_merged`, also backed up under `/data/v0-boxiuli/sii-agent/saves/qwen35-9b/merged/v13_step_final_opd_32k_full_lora_manual_merged`.
- SGLang is the contest-compatible serving backend. The root code still calls it through `LLM_BACKEND=vllm` because `agent/llm.py` only distinguishes Azure vs OpenAI-compatible local `/v1`.
- Full runtime LoRA through SGLang is not equivalent to the OPD v13 adapter unless SGLang supports all Qwen3.5 linear-attention targets. Do not compare pruned runtime-LoRA numbers against full-LoRA/vLLM numbers as if they were the same model.
- Local 2Wiki `summary.json` metrics are useful for fast iteration, but many differences are answer-granularity issues. If Qwen3-32B judge service is available, label semantic-judge results separately from local exact/F1.
- Recent trace/eval diagnosis found that tool parsing matters a lot. Native OpenAI `tool_calls`, Qwen textual `<tool_call><function=...>`, and JSON-list final answers are all valid model outputs that `agent/react.py` should handle.

## Preferred workflow for this repository

- Communicate findings in Chinese when working with the primary user.
- Prefer doing the experiment/change, then reporting measured results. Avoid stopping at a plan when commands can be run safely.
- Before every experiment run, print the exact configuration for review: model path/name, serving port/base URL, backend env vars, task/split/n/offset, baseline/evolved mode, concurrency, token and timeout limits, tool profile, memory/reflection/lesson/skill env vars, and output directory.
- For expensive evaluations, first run a smoke slice (`--n 10` or `--n 20`), then scale to 500 examples.
- Use high concurrency for large local runs when the server can handle it; `--concurrency 128` is the common 2Wiki 500-run setting.
- When comparing baseline and evolved, keep `task`, `split`, `n`, `offset`, `concurrency`, token limits, tool profile, model endpoint, and memory/reflection env vars identical except for `--mode`/`--evolve-batch-size`.
- When reporting whether evolved helps, include both absolute scores and the net delta. Also inspect whether evolved mechanisms actually fired: `lesson_context`, `memory_context_refs`, `first_attempt`, `retry_attempt`, `selected_attempt`, `retry_selected`, and `reflection_useful` when traces are saved.
- After local 2Wiki evals, use Qwen3-32B as the semantic judge when available, and report judge accuracy separately from local exact/F1.
- For trace debugging, preserve enough fields to see tool calls, model content/reasoning behavior, final answer, retry/reflection behavior, and finish reasons. Distinguish parser failures from reasoning failures.
- If the machine may be reclaimed, persist important code snapshots, merged checkpoints, summaries, and notes under `/data/v0-boxiuli/sii-agent` in addition to `/root/sii-agent`.
- Keep large eval/model artifacts out of source changes unless explicitly asked. Prefer documenting paths to artifacts rather than copying them into tracked files.
- For 2Wiki, be cautious with new prompt policies, skills, or postprocessors. Ablate them independently and compare against a safe default because reflection/lesson/skill changes have been noisy.
