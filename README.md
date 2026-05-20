# SII-Agent

SII-Agent 是一个面向问答评测和工具调用研究的 Agent 框架。它把 **ReAct 多步推理**、**搜索/浏览/Wiki/视觉工具**、**长期记忆与反思进化**、**批量评测** 和 **OPD 偏好蒸馏** 放在同一个最小可运行工程里。

当前项目主线是：用 Qwen3.5-9B / OPD v13 在 2WikiMultihopQA 上比较 `baseline` 和 `evolved`，同时保留 SimpleQA、SimpleVQA、BrowseComp-Plus 和训练数据导出的实验入口。

## 先理解这个项目在做什么

一次普通运行大致是：

```text
问题
  -> evaluation/run_eval.py 读取数据集样本
  -> agent/runner.py 选择 baseline 或 evolved
  -> agent/react.py 让模型循环调用工具
  -> tools/registry.py 分发 web_search / wiki_search / browser_open / final_answer 等工具
  -> agent/scoring.py 评分
  -> memory/store.py 在 evolved 模式下记录经验或 lessons
```

两种核心模式：

| 模式 | 含义 | 适合做什么 |
|---|---|---|
| `baseline` | 只跑 ReAct，不读写长期记忆，不做反思 | 作为对照组、排查模型/工具是否正常 |
| `evolved` | 可注入历史 lessons，失败后可反思、重试并写入记忆 | 研究“经验积累是否提升后续样本” |

## 目录速览

| 路径 | 作用 |
|---|---|
| `agent/` | ReAct 循环、LLM 客户端、baseline/evolved runner、评分与反思逻辑 |
| `tools/` | 工具注册与实现：搜索、Wiki、网页浏览、视觉工具、BrowseComp 检索、`final_answer` |
| `evaluation/` | SimpleQA / SimpleVQA / 2Wiki / BrowseComp-Plus 的评测脚本和结果合并脚本 |
| `memory/` | 文件型长期记忆：`episodes.jsonl`、`lessons.jsonl`、`skills.jsonl`；短期工作记忆 |
| `training/` | OPD / DPO 数据导出、LlamaFactory 配置生成、slime rollout 接入 |
| `harness/` | 评测 harness 控制器（并发调度、结果收集） |
| `scripts/` | 索引构建、模型服务启动、smoke test、辅助数据转换 |
| `configs/` | `.env` 示例配置 |
| `docs/` | 比赛提交说明、实验报告、研究笔记 |
| `logs/`, `data/`, `indexes/`, `saves/` | 运行产物、数据、索引和模型权重；默认不应提交到 Git |

## 环境准备

建议使用 Python 3.10+。先安装依赖并复制环境变量模板：

```bash
pip install -r requirements.txt
cp configs/.env.example .env
```

`.env` 里最重要的是 LLM 后端：

```bash
# 本地 vLLM / SGLang，二者都走 OpenAI-compatible /v1 接口
LLM_BACKEND=vllm
VLLM_BASE_URL=http://127.0.0.1:8004/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen3.5-9B
VLLM_ENABLE_THINKING=0

# 或者 Azure OpenAI
# LLM_BACKEND=azure
# AZURE_OPENAI_ENDPOINT=...
# AZURE_OPENAI_DEPLOYMENT=gpt-5.4
```

> `python -m scripts.smoke` 会真实调用模型服务；如果本地没有启动 vLLM/SGLang 或 Azure 凭据不可用，它会连接失败，这是环境问题而不是代码问题。

## 启动本地模型服务

### SGLang（当前推荐的比赛兼容路径）

如果机器上已经有 Qwen3.5-9B 或手动合并后的 OPD 权重，可以用仓库脚本启动 SGLang：

```bash
python -m venv --system-site-packages /root/sglang-venv
/root/sglang-venv/bin/python -m pip install -U pip setuptools wheel
/root/sglang-venv/bin/python -m pip install 'sglang[all]==0.5.11'

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_PYTHON=/root/sglang-venv/bin/python \
SGLANG_MODEL=/root/sii-agent/Qwen3.5-9B \
SGLANG_PORT=8004 \
SGLANG_TP=4 \
SGLANG_CONTEXT_LENGTH=32768 \
SGLANG_SERVED_MODEL_NAME=Qwen3.5-9B \
SGLANG_TOOL_CALL_PARSER=qwen3_coder \
bash scripts/start_qwen_sglang.sh
```

另开一个 shell 设置客户端环境：

```bash
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://127.0.0.1:8004/v1
export VLLM_MODEL=Qwen3.5-9B
export VLLM_API_KEY=EMPTY
export VLLM_ENABLE_THINKING=0
```

如果要跑当前 OPD v13 完整效果，优先服务手动 merge 后的 HF checkpoint，而不是 SGLang runtime LoRA：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_PYTHON=/root/sglang-venv/bin/python \
SGLANG_MODEL=/root/sii-agent/saves/qwen35-9b/merged/v13_step_final_opd_32k_full_lora_manual_merged \
SGLANG_PORT=8004 \
SGLANG_TP=4 \
SGLANG_CONTEXT_LENGTH=262144 \
SGLANG_SERVED_MODEL_NAME=sii-opd-v13-merged-sglang \
SGLANG_TOOL_CALL_PARSER=qwen3_coder \
bash scripts/start_qwen_sglang.sh

export VLLM_MODEL=sii-opd-v13-merged-sglang
export VLLM_ENABLE_THINKING=0
```

注意：

- Qwen3.5/SGLang 跑评测时保持 `VLLM_ENABLE_THINKING=0`，否则模型可能把 token 花在 `reasoning_content`，影响工具调用和最终答案解析。
- `qwen3_coder` tool-call parser 比 `hermes` 更适合当前 Qwen3.5 textual tool-call 输出。
- `scripts/create_sglang_supported_lora.py` 可以生成只保留 SGLang 支持 target 的 pruned LoRA，但它不等价于完整 OPD v13 LoRA，不能直接与完整 merge 权重的结果比较。

### vLLM（开发备用）

```bash
bash scripts/start_qwen_vllm.sh

export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://127.0.0.1:18877/v1
export VLLM_MODEL=Qwen/Qwen3.5-9B
export VLLM_ENABLE_THINKING=0
```

### Azure OpenAI（开发备用）

```bash
az login
export LLM_BACKEND=azure
export AZURE_OPENAI_ENDPOINT=<your-endpoint>
export AZURE_OPENAI_DEPLOYMENT=gpt-5.4
```

Azure 路径使用 AAD 鉴权，见 `agent/llm.py` 和 `configs/.env.example`。

## 准备检索索引

不是所有任务都必须有全部索引。按需要准备：

### 2Wiki / Wiki 搜索

离线 Wiki 工具优先读取 SQLite FTS 索引：

```bash
python -m scripts.build_wiki_fts \
  --source data/wiki25/wiki25_sample.jsonl \
  --out data/wiki25/wiki25_fts.sqlite

export WIKI25_INDEX_PATH=data/wiki25/wiki25_fts.sqlite
```

`web_search` 优先调用课题提供的 search-proxy；离线 Wiki 仍通过单独的 `wiki_search` / `wiki_page` 工具使用。代理端负责 Serper + Jina，API key 留在代理端：

```bash
export SEARCH_PROXY_URL=http://127.0.0.1:1227
export SEARCH_PROXY_FETCH=0
export SEARCH_PROXY_MAX_CHARS=0
export SEARCH_PROXY_IMAGE_UPLOAD_BACKENDS=tmpfiles,catbox,proxy
```

如果没有 `SEARCH_PROXY_URL`，`web_search` 会直接调用 Serper Search，`reverse_image_search` 会直接调用 Serper Lens；此时需要在本机配置 key：

```bash
export SERPER_API_KEY=...
export SERPER_FETCH=0
export SERPER_MAX_CHARS=0
# 可选：fetch 正文时用于 Jina Reader
export JINA_API_KEY=...
```

`reverse_image_search` 会优先调用 search-proxy 的 image/lens 搜索；无 proxy 时使用 Serper Lens，并保留本地图片上传与文本 fallback。

### BrowseComp-Plus

```bash
python -m scripts.download_browsecomp_index --out indexes
export BROWSECOMP_INDEX_PATH=indexes/bm25
```

BrowseComp-Plus 使用固定语料检索，主工具名是 `search`，默认返回 top-5 文档片段。

## 第一次运行

先确认模型和工具链能通：

```bash
python -m scripts.smoke
```

再跑一个最小 2Wiki 样本：

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode baseline \
  --n 1 \
  --offset 0 \
  --max-llm-tokens 12000 \
  --out logs/eval_smoke
```

输出会写到 `logs/...`，其中 `summary.json` 是最常看的汇总文件；加 `--save-traces` 会写完整轨迹 `runs.jsonl`，适合调试工具调用和解析问题。

## 常用评测命令

### 2Wiki baseline / evolved 小样本对比

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode baseline \
  --n 20 \
  --offset 0 \
  --concurrency 32 \
  --max-llm-tokens 12000 \
  --out logs/opd_eval

python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode evolved \
  --n 20 \
  --offset 0 \
  --concurrency 32 \
  --evolve-batch-size 32 \
  --max-llm-tokens 12000 \
  --out logs/opd_eval
```

### 2Wiki 500 样本主实验

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode baseline \
  --n 500 \
  --offset 0 \
  --concurrency 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval

python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode evolved \
  --n 500 \
  --offset 0 \
  --concurrency 128 \
  --evolve-batch-size 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval
```

如果要显式测试 2Wiki 的长期记忆注入，可以开启 lessons/reflection，并让 evolved 从前序 batch 的成功 episode 中检索策略记忆：

```bash
export SII_2WIKI_ENABLE_REFLECTION=1
export SII_2WIKI_ENABLE_LESSONS=1
unset SII_2WIKI_ENABLE_SKILLS
unset SII_2WIKI_ENABLE_TYPED_POLICIES

python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode evolved \
  --n 500 \
  --offset 0 \
  --concurrency 128 \
  --evolve-batch-size 128 \
  --include-success-memory \
  --memory-k 3 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval
```

更公平的 memory 评测流程不要在 validation/test 上用 gold 生成记忆。可以先从 2Wiki train 蒸馏 lessons，再把同一个 memory root 以 read-only 注入 validation/test：

```bash
python -m evaluation.run_2wiki_memory_eval \
  --memory-n 500 \
  --memory-offset 0 \
  --eval-split validation \
  --eval-n 500 \
  --eval-offset 0 \
  --concurrency 128 \
  --evolve-batch-size 128 \
  --memory-k 3 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --tool-profile benchmark \
  --run-baseline \
  --out logs/opd_eval
```

这个封装会在 distill 阶段设置 `SII_2WIKI_ENABLE_REFLECTION=1`、关闭 lesson 检索并用 train gold reflection 写入 `lessons.jsonl`；在 scored eval 阶段设置 `SII_2WIKI_ENABLE_LESSONS=1`、`memory_mode=read_only`、`use_gold_for_reflection=false`。如果手动传入已有 `--memory-root` 且使用 fresh 模式，必须加 `--force-fresh-memory` 才会删除旧记忆。

对比 baseline 和 evolved 时，除 `--mode` / `--evolve-batch-size` 外，保持模型端点、split、`n`、`offset`、concurrency、token 限制、工具 profile 和环境变量一致。

### SimpleQA / SimpleVQA

```bash
python -m evaluation.run_eval --task simpleqa --mode baseline --n 50 --concurrency 32

python -m evaluation.run_eval \
  --task simplevqa \
  --mode baseline \
  --n 20 \
  --tool-profile visual \
  --concurrency 16
```

SimpleQA 当前 loader 没有 public train split；不要把 SimpleQA test 同时用于训练和评测。

### BrowseComp-Plus

```bash
python -m evaluation.run_browsecomp \
  --mode evolved \
  --n 20 \
  --concurrency 32 \
  --out logs/browsecomp
```

`--n 0` 表示跑完整集合。输出目录会包含官方兼容的 `runs/` JSON、解密后的 ground truth 和 qrel evidence，可再用官方评测脚本打分。

多个 BrowseComp run 可以无 gold 合并或路由：

```bash
python -m evaluation.merge_browsecomp_runs \
  --run-dirs <run-a> <run-b> \
  --out <merged>

python -m evaluation.route_browsecomp_runs \
  --primary-run <cheap-run> \
  --fallback-run <selector-run> \
  --out <routed>
```

## 2Wiki 当前安全默认值

最近的实验结论是：2Wiki 的 evolved 机制能提升，但收益小且不稳定；反思、lesson、skill 和 typed policy 都应该按 ablation 显式开启，不要默认混在主实验里。

安全默认：

```bash
unset SII_2WIKI_ENABLE_REFLECTION
unset SII_2WIKI_ENABLE_LESSONS
unset SII_2WIKI_ENABLE_SKILLS
unset SII_2WIKI_ENABLE_TYPED_POLICIES
```

如果要复现实验中的 legacy lesson/reflection 路径：

```bash
export SII_2WIKI_ENABLE_REFLECTION=1
export SII_2WIKI_ENABLE_LESSONS=1
unset SII_2WIKI_ENABLE_SKILLS
unset SII_2WIKI_ENABLE_TYPED_POLICIES
```

重要边界：

- 普通 evolved 运行不要用 `--gold-reflection`；它只适合受控实验，否则会把 gold answer 泄漏进记忆。
- `--save-traces` 能帮助确认 `lesson_context`、`memory_context_refs`、`first_attempt`、`retry_attempt`、`selected_attempt`、`retry_selected` 和 `reflection_useful` 是否真的触发。
- 本地 2Wiki `summary.json` 是 exact/F1 评分；如果另跑 Qwen3-32B 语义 judge，结果要单独标注，不能和本地 accuracy 混为一谈。

## 工具 profile

ReAct 默认不会暴露所有工具，而是按 profile 控制工具集合：

| Profile | 工具集合 | 用途 |
|---|---|---|
| `benchmark` / `default` | `web_search`, `wiki_search`, `wiki_page`, `browser_open`, `browser_open_many`, `final_answer` | 2Wiki、SimpleQA 等文本任务默认路径 |
| `visual` | 文本工具 + `visual_web_search`, `image_to_text`, `image_to_search_queries`, `reverse_image_search` | SimpleVQA / 图片问答 |
| `rich` / `full` | 文本、视觉、浏览器交互工具 | 需要 JS 页面或点击输入时 |
| `memory` / `self_retrieval` | benchmark 工具 + `memory_search`, `memory_stats`, `memory_list`, `memory_get`，训练态还会暴露 `memory_create/update/delete` | 让 agent 自己按需检索和维护 lessons/episodes/skills |
| `all` | 注册表里所有工具 | 调试或展示能力，不建议大规模评测默认使用 |

命令行用 `--tool-profile visual`，或通过环境变量设置：

```bash
export SII_AGENT_TOOL_PROFILE=visual
```

记忆默认是全局的：evolved run 会复用并追加到 `SII_AGENT_MEMORY_ROOT`（默认 `logs/memory`），不再默认写到每个 run 目录下。该目录统一保存 `episodes.jsonl`、`lessons.jsonl`、`skills.jsonl`。自检索记忆模式会把 `memory_search` 和 `memory_list` / `memory_get` / `memory_create` / `memory_update` / `memory_delete` 暴露给 agent，让它自己按需检索和维护记忆：

```bash
python -m evaluation.run_eval \
  --task 2wiki --split validation --mode evolved \
  --tool-profile self_retrieval \
  --self-retrieval-memory
```

如需临时隔离实验，可显式传 `--memory-root <dir>`；如需清空全局记忆再跑，才使用 `--memory-mode fresh`。

### 运行时态：train / test

Agent 现在有两个长期记忆运行时态：

| 运行态 | 行为 |
|---|---|
| `train` | 默认态。evolved run 复用并写入全局 memory；反思产生的有效 lesson/skill 会持久化；自检索 profile 可使用 memory 增删改查。 |
| `test` | 测试态。强制 memory read-only，不清空、不追加、不写反思结果；`memory_create/update/delete` 不会暴露给 ReAct，直接调用也会返回只读错误；`--gold-reflection` 会被忽略。 |

```bash
# 积累经验
python -m evaluation.run_eval --task 2wiki --split train --mode evolved --runtime-mode train

# 使用冻结的全局记忆做评测
python -m evaluation.run_eval \
  --task 2wiki --split validation --mode evolved \
  --runtime-mode test \
  --tool-profile self_retrieval \
  --self-retrieval-memory
```

`bash_exec` 是高风险实验工具，默认不暴露；只有设置 `SII_AGENT_ENABLE_SHELL_TOOL=1` 且使用 `memory`/`self_retrieval`/`rich`/`full`/`all` profile 时才会出现在工具集中。

如果需要使用 `/root/harness-sii-browser-service` 里的 browser-service，先启动服务，再设置：

```bash
export SANDBOX_BASE_URL=http://127.0.0.1:8080
```

此时 `browser_open`、`browser_open_many`、`browser_text`、`browser_click`、`browser_type` 会通过 browser-service 暴露的 CDP 浏览器执行；未设置时仍使用本地 Playwright 或原 All-in-One Sandbox 路径。

也可以把工具以 HTTP 服务暴露：

```bash
uvicorn tools.server:app --host 0.0.0.0 --port 8080
```

## 语义 Judge

2Wiki 本地评分偏 exact/F1，容易受答案粒度影响。若有 Qwen3-32B judge 服务，可以对已完成 run 做语义判断：

```bash
python -m evaluation.judge_semantic \
  --run-dirs logs/opd_eval/<run-root>/2wiki_baseline_<ts> logs/opd_eval/<run-root>/2wiki_evolved_<ts> \
  --base-url http://127.0.0.1:8005/v1 \
  --model Qwen3-32B \
  --concurrency 64 \
  --out-prefix semantic_judge_qwen32
```

报告结果时同时写清楚：

- local exact/F1 accuracy
- semantic judge accuracy
- baseline 与 evolved 的绝对分数和净差值

## OPD / 偏好蒸馏

高级入口在 `training/opd.py`。典型流程是先保存 baseline/evolved 轨迹，再用专家模型构造偏好数据，导出 LlamaFactory DPO 配置：

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split train \
  --mode baseline \
  --n 200 \
  --concurrency 32 \
  --save-traces \
  --out logs/opd_traces

python -m evaluation.run_eval \
  --task 2wiki \
  --split train \
  --mode evolved \
  --n 200 \
  --concurrency 32 \
  --evolve-batch-size 32 \
  --save-traces \
  --out logs/opd_traces

python -m training.opd \
  --runs logs/opd_traces/<baseline-run> logs/opd_traces/<evolved-run> \
  --out logs/opd/gpt54_2wiki \
  --expert llm \
  --pref-loss sigmoid \
  --model-name-or-path Qwen/Qwen3.5-9B \
  --lf-export-mode answer \
  --lf-template qwen3_5_nothink
```

默认导出模式 `--lf-export-mode answer` 只优化最终答案文本，通常最稳；`final_tool` / `action` 会更强地改变工具策略，适合单独 ablation。

slime 在线 Agent OPD 的接入点是 `training/slime_sii_rollout.py`，脚本在 `scripts/slime/` 下。

## 常见坑

| 现象 | 常见原因 / 处理 |
|---|---|
| `scripts.smoke` 报 `APIConnectionError` | 模型服务没启动、端口不对，或 `.env` 的 `VLLM_BASE_URL` / Azure 配置不可用 |
| 模型一直输出 reasoning，没有工具调用 | Qwen3.5/SGLang 没关 thinking；设置 `VLLM_ENABLE_THINKING=0` |
| SGLang tool-call 解析偶发 400 | 启动时设置 `SGLANG_TOOL_CALL_PARSER=qwen3_coder` |
| 2Wiki evolved 比 baseline 差 | 这是可能的；先用安全默认关闭 reflection/lesson/skill，再逐项 ablation |
| 大量运行很慢 | 先跑 `--n 10` 或 `--n 20` smoke slice，再扩大到 500；确认 concurrency 与服务吞吐匹配 |
| 想提交代码但有大文件 | `logs/`, `data/`, `indexes/`, `saves/`, `Qwen*`, `third_party/` 默认在 `.gitignore` 中，不要手动 add |

更多 2Wiki 当前最佳配置和历史结果见 `docs/2wiki_best_results.md`。
