# Current 2Wiki best-performance snapshot

This repository state is the current best-performance/safest 2Wiki version to keep as the main backup.

- Commit: `158bcfe` (`Restore 2Wiki lesson evolution defaults`)
- Primary task: `2wiki`
- Agent model: Qwen3.5-9B + OPD v13 LoRA (`sii-opd-v13`)
- Judge model: Qwen3-32B only when doing LLM judging; do not use it as the 2Wiki agent.

## Best observed local results

Best observed local 500-example run before the skill experiment:

- Run: `logs/opd_eval/2wiki_memory_ranked_eval500_1778915745/`
- Baseline: `382/500 = 76.4%`
- Evolved: `389/500 = 77.8%`
- Net gain: `+7/500 = +1.4%`

Latest restored-mechanism reproduction after the skill experiment:

- Run: `logs/opd_eval/2wiki_restored_lessons_oldcfg_eval500_1778924793/`
- Baseline: `372/500 = 74.4%`
- Evolved: `375/500 = 75.0%`
- Net gain: `+3/500 = +0.6%`

Conclusion: evolved can beat baseline, but the gain is still small and unstable. The useful part is the legacy reflection-to-lesson loop, not the current skill prompt injection.

## Current default after trace/evolution diagnosis

The latest diagnosis found one implementation issue and one mechanism issue:

- Qwen-style textual tool calls (`<function=...><parameter=...>`) were not parsed, so some apparent searches in traces were not executed. This is now fixed in `agent/react.py`.
- 2Wiki reflection/lesson injection and the experimental typed policy cards are not consistently positive under Qwen3-32B semantic judging. They are now opt-in for ablation rather than default behavior.

Safe default:

```bash
unset SII_2WIKI_ENABLE_REFLECTION
unset SII_2WIKI_ENABLE_LESSONS
unset SII_2WIKI_ENABLE_SKILLS
unset SII_2WIKI_ENABLE_TYPED_POLICIES
```

Opt-in ablations:

```bash
# Legacy lesson/reflection path.
export SII_2WIKI_ENABLE_REFLECTION=1
export SII_2WIKI_ENABLE_LESSONS=1

# Experimental typed policies.
export SII_2WIKI_ENABLE_TYPED_POLICIES=1

# Experimental skill path.
export SII_2WIKI_ENABLE_SKILLS=1
```

## Required configuration for legacy reflection experience ablation

Use evolved mode with the legacy lesson mechanism enabled:

```bash
cd /root/sii-agent

export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://127.0.0.1:8002/v1
export VLLM_MODEL=sii-opd-v13
export VLLM_API_KEY=EMPTY
export VLLM_ENABLE_THINKING=1
export SEARCH_PROXY_URL=http://127.0.0.1:1227
export SEARCH_PROXY_FETCH=0
export SEARCH_PROXY_MAX_CHARS=0
export WIKI25_INDEX_PATH=data/wiki25/wiki25_fts.sqlite

export SII_2WIKI_ENABLE_REFLECTION=1
export SII_2WIKI_ENABLE_LESSONS=1
unset SII_2WIKI_ENABLE_SKILLS
unset SII_2WIKI_ENABLE_TYPED_POLICIES
```

Important behavior:

- `SII_2WIKI_ENABLE_REFLECTION=1` enables 2Wiki reflection; it is disabled by default after the latest diagnosis.
- `SII_2WIKI_ENABLE_LESSONS=1` enables seeded and dynamic lesson injection for 2Wiki; it is disabled by default after the latest diagnosis.
- `SII_2WIKI_ENABLE_SKILLS` must stay unset for best current performance. If set to `1`, reflection writes dynamic skills instead of legacy lessons, and the tested skill prompt path currently underperforms.
- `SII_2WIKI_ENABLE_TYPED_POLICIES=1` enables experimental typed policy cards; latest 500-run was judge-neutral but not positive.
- In evolved mode, reflection writes reusable items to `memory/lessons.jsonl`; later batches retrieve those lessons through `memory.render_for_prompt(..., task="2wiki")`.
- For 2Wiki, success memories are disabled by default; only seeded lessons plus useful failure reflections are injected.

Recommended evaluation command:

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode evolved \
  --n 500 \
  --offset 0 \
  --max-steps 8 \
  --concurrency 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --min-llm-call-seconds 5 \
  --tool-profile benchmark \
  --out logs/opd_eval
```

## Notes

- The skill mechanism is implemented but should be treated as experimental/opt-in for now.
- Reflection itself is noisy; it helps only when its filtered reusable lessons are passed to later batches.
- High token limits reduce truncation noise. The best historical run used `--max-llm-tokens 12000` and `--max-llm-call-seconds 600`.
- Trace attribution now records first/retry attempts and retry selection fields, so future runs can measure whether reflection/retry caused each win or loss.
