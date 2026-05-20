# SII-Agent

这是一个面向问答评测的 Agent 框架，核心研究目标是验证「长期记忆 + 反思」能否让模型在多跳问答任务上随经验积累而持续进步。

项目当前主线：用 Qwen3.5-9B（经 OPD 偏好蒸馏）在 2WikiMultihopQA 上对比 baseline 和 evolved 两种模式的表现差异。

---

## 目录结构

```
sii-agent/
├── agent/           # Agent 核心：ReAct 循环、LLM 调用、反思逻辑
├── tools/           # 工具实现：搜索、Wiki、浏览器、视觉、记忆读写
├── evaluation/      # 各任务评测入口和结果处理
├── memory/          # 长期记忆：lessons/episodes 文件存储与检索
├── training/        # OPD 偏好蒸馏数据导出
├── harness/         # 步数/超时/重复检测控制器
├── services/        # 外部依赖服务（浏览器、搜索代理、任务调度）
│   ├── browser-service/   # Headless Chromium HTTP 服务
│   ├── search-proxy/      # Serper + Jina 搜索代理
│   └── task-runner/       # BrowseComp 评测任务调度器
├── scripts/         # 启动脚本、索引构建、数据转换
├── configs/         # 环境变量模板
├── docs/            # 实验报告和比赛提交说明
├── data/            # 数据集（不入 Git）
├── logs/            # 运行日志和产物（不入 Git）
├── indexes/         # 检索索引（不入 Git）
└── saves/           # 模型权重（不入 Git）
```

---

## 从零开始跑通整个实验

下面是完整的操作步骤。假设你有一台至少 4×A100 (40GB) 或等效 GPU 的机器，系统是 Ubuntu/CentOS，Python 3.10+。

### 第一步：克隆仓库 & 装依赖

```bash
git clone https://github.com/lbx154/sii-agent.git
cd sii-agent
pip install -r requirements.txt
```

如果你只有一个全局 Python 环境不想污染，用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 第二步：准备模型权重

你需要 Qwen3.5-9B 的 HuggingFace 格式权重。两种方式：

**方式一：用原始 Qwen3.5-9B**（不带 OPD 微调，跑 baseline 够用）

```bash
# 从 HuggingFace 下载
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./Qwen3.5-9B
```

**方式二：下载已训练好的权重**

我们提供了两个公开的微调版本：

- OPD 偏好蒸馏版本：[Takagiv/qwen35-9b-sii-opd-best](https://huggingface.co/Takagiv/qwen35-9b-sii-opd-best)
- SFT 监督微调版本：[Takagiv/sft217-aligned-lr1e5-b8-e3-20260519-1330](https://huggingface.co/Takagiv/sft217-aligned-lr1e5-b8-e3-20260519-1330)

```bash
# 下载 OPD 版本
huggingface-cli download Takagiv/qwen35-9b-sii-opd-best --local-dir ./saves/qwen35-9b-sii-opd-best

# 或下载 SFT 版本
huggingface-cli download Takagiv/sft217-aligned-lr1e5-b8-e3-20260519-1330 --local-dir ./saves/sft217-aligned
```

下载完后在启动 SGLang 时把 `SGLANG_MODEL` 指向对应路径即可。

### 第三步：启动模型推理服务

我们用 SGLang 把模型跑成一个 OpenAI 兼容的 HTTP 服务。

先装 SGLang（建议放独立 venv，避免和主项目依赖冲突）：

```bash
python -m venv /root/sglang-venv
/root/sglang-venv/bin/python -m pip install -U pip setuptools wheel
/root/sglang-venv/bin/python -m pip install 'sglang[all]==0.5.11'
```

然后启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_PYTHON=/root/sglang-venv/bin/python \
SGLANG_MODEL=./Qwen3.5-9B \
SGLANG_PORT=8004 \
SGLANG_TP=4 \
SGLANG_CONTEXT_LENGTH=32768 \
SGLANG_SERVED_MODEL_NAME=Qwen3.5-9B \
SGLANG_TOOL_CALL_PARSER=qwen3_coder \
bash scripts/start_qwen_sglang.sh
```

等看到类似 `The server is fired up and ready to roll!` 的日志就行了。如果你的 GPU 数量不是 4 张，把 `CUDA_VISIBLE_DEVICES` 和 `SGLANG_TP` 改成你实际的卡数。

启动后，另开一个终端，验证服务是否正常：

```bash
curl http://127.0.0.1:8004/v1/models
```

应该能看到返回模型列表。

### 第四步：配置环境变量

```bash
cp configs/.env.example .env
```

编辑 `.env`，最关键的几项：

```bash
LLM_BACKEND=vllm
VLLM_BASE_URL=http://127.0.0.1:8004/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen3.5-9B
VLLM_ENABLE_THINKING=0
```

或者直接 export 到当前 shell（不用 .env 文件也行）：

```bash
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://127.0.0.1:8004/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=Qwen3.5-9B
export VLLM_ENABLE_THINKING=0
```

> 注意：`VLLM_ENABLE_THINKING=0` 是必须的。Qwen3.5 默认会把 token 花在内部思考上，不关掉的话工具调用解析会出问题。

### 第五步：准备 2Wiki 数据

2WikiMultihopQA 数据集需要提前下好。评测脚本通过 `evaluation/datasets.py` 加载，默认路径是 HuggingFace datasets cache。第一次跑会自动从网上拉，如果机器没外网，需要手动下载并指定 `HF_DATASETS_CACHE`。

如果需要离线 Wiki 工具（用于 `wiki_search` / `wiki_page`），还要构建 FTS 索引：

```bash
python -m scripts.build_wiki_fts \
  --source data/wiki25/wiki25_sample.jsonl \
  --out data/wiki25/wiki25_fts.sqlite

export WIKI25_INDEX_PATH=data/wiki25/wiki25_fts.sqlite
```

对于基础的 2Wiki 实验，这一步可以跳过——默认用 `web_search` 就能跑。

### 第六步：冒烟测试

```bash
python -m scripts.smoke
```

这会实际调用模型，确认端到端链路通畅。如果报 `APIConnectionError`，说明模型服务没启动或端口不对。

然后跑一个最小样本：

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode baseline \
  --n 1 \
  --offset 0 \
  --max-llm-tokens 12000 \
  --out logs/smoke_test
```

成功的话会在 `logs/smoke_test/` 下生成结果文件。

---

## 两种核心模式

| 模式 | 含义 |
|------|------|
| `baseline` | 纯 ReAct，不读写记忆，不做反思。作为对照组。 |
| `evolved` | 可注入历史 lessons，失败后做反思 + 重试，成功经验写入记忆。 |

---

## 跑实验

### Baseline（对照组）

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode baseline \
  --n 500 \
  --concurrency 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval
```

### Evolved（实验组，带记忆）

需要先开启反思和 lesson 注入：

```bash
export SII_2WIKI_ENABLE_REFLECTION=1
export SII_2WIKI_ENABLE_LESSONS=1
```

然后跑：

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode evolved \
  --n 500 \
  --concurrency 128 \
  --evolve-batch-size 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval
```

`--evolve-batch-size 128` 表示每 128 个样本为一个 micro-batch，一个 batch 内共享同一份冻结的 memory snapshot；batch 结束后新产生的 lessons 才对下一个 batch 可见。

### Train 模式（积累记忆）

先用 train split 让 Agent 积累经验：

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split train \
  --mode evolved \
  --runtime-mode train \
  --n 500 \
  --concurrency 128 \
  --evolve-batch-size 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval
```

运行结束后，记忆默认存在 `logs/memory/`（由 `SII_AGENT_MEMORY_ROOT` 控制），里面有 `lessons.jsonl` 和 `episodes.jsonl`。

### Test 模式（冻结记忆做评测）

在 train 之后，用冻结的记忆在 validation/test 上评测：

```bash
python -m evaluation.run_eval \
  --task 2wiki \
  --split validation \
  --mode evolved \
  --runtime-mode test \
  --memory-root logs/memory \
  --n 500 \
  --concurrency 128 \
  --evolve-batch-size 128 \
  --max-wall-seconds 600 \
  --max-llm-tokens 12000 \
  --max-llm-call-seconds 600 \
  --out logs/opd_eval
```

`--runtime-mode test` 会强制 memory 只读：不写新 lesson，不追加 episode，不做 gold reflection。

### 一体化 Memory 评测

如果你想一键完成「train 积累 → test 评测 → baseline 对照」全流程：

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

它会自动走完：在 train split 做记忆蒸馏 → 冻结记忆到 validation 做 evolved 评测 → 跑一遍 baseline 做对照。

---

## 环境变量速查

| 变量 | 作用 | 必需？ |
|------|------|--------|
| `LLM_BACKEND` | `vllm` 或 `azure` | 是 |
| `VLLM_BASE_URL` | 模型服务地址（如 `http://127.0.0.1:8004/v1`）| 是（vllm 模式） |
| `VLLM_MODEL` | 模型名，需和 SGLang 的 `--served-model-name` 一致 | 是 |
| `VLLM_API_KEY` | 模型服务的 API key，本地服务填 `EMPTY` | 是 |
| `VLLM_ENABLE_THINKING` | 必须设为 `0`，否则 Qwen3.5 会内部思考导致工具调用失败 | 是 |
| `SII_2WIKI_ENABLE_REFLECTION` | 设为 `1` 开启反思 | evolved 模式建议开 |
| `SII_2WIKI_ENABLE_LESSONS` | 设为 `1` 开启 lesson 注入 | evolved 模式建议开 |
| `SII_AGENT_MEMORY_ROOT` | 全局记忆存储路径，默认 `logs/memory` | 否 |
| `SII_AGENT_RUNTIME_MODE` | `train` 或 `test`，也可以命令行 `--runtime-mode` 覆盖 | 否 |
| `WIKI25_INDEX_PATH` | Wiki FTS 索引路径 | 用 wiki_search 时需要 |
| `SEARCH_PROXY_URL` | 搜索代理地址 | 没外网时需要 |
| `SANDBOX_BASE_URL` | Browser-service 地址 | 用浏览器交互时需要 |

---

## 外部服务（services/ 目录）

GPU 服务器通常没有外网，也没有图形界面。以下三个服务解决这两个问题。

### Browser-Service（Headless Chromium HTTP 服务）

位于 `services/browser-service/`。在一台有图形环境或至少能跑 Chromium 的机器上启动（比如你的 Mac 或一台 CPU 服务器）：

```bash
cd services/browser-service
chmod +x run.sh
./run.sh
```

首次启动会自动：创建 venv → 装依赖 → 下载 Playwright Chromium → 监听 `0.0.0.0:8080`。

启动后在 GPU 服务器上设置：

```bash
export SANDBOX_BASE_URL=http://<browser-machine-ip>:8080
```

然后 Agent 的 `browser_open` / `browser_click` / `browser_type` 等工具就会通过这个服务操作浏览器。

API 文档：`http://<browser-machine-ip>:8080/docs`（Swagger 自动生成）。

更多配置见 `services/browser-service/.env.example` 和 `services/browser-service/README.md`。

### Search-Proxy（搜索代理）

位于 `services/search-proxy/`。部署在有外网的机器上，替 GPU 服务器做 Google 搜索和网页抓取：

```bash
cd services/search-proxy
export SERPER_API_KEY=你的serper密钥
export JINA_API_KEY=你的jina密钥    # 可选
export PROXY_API_TOKEN=$(openssl rand -hex 16)  # 鉴权用
./run.sh
```

默认监听 `127.0.0.1:8090`。通过 SSH 端口转发把 GPU 服务器的 8090 映射过来：

```bash
# 在 GPU 服务器上
ssh -L 8090:127.0.0.1:8090 user@cpu-server -N &

export SEARCH_PROXY_URL=http://127.0.0.1:8090
export SEARCH_PROXY_TOKEN=<你上面生成的token>
```

或者如果两台机器在同一内网，直接把 search-proxy 改为监听 `0.0.0.0`，GPU 服务器直连也行。

### Task-Runner（BrowseComp 评测调度器）

位于 `services/task-runner/`。这是 BrowseComp-Plus 任务的评测调度脚本，负责把问题分发给 Agent 并收集轨迹。一般实验不需要动它，只有跑 BrowseComp-Plus 比赛时才用到。

用法见 `services/task-runner/README_browser.md` 和 `services/task-runner/README_search.md`。

---

## 工具 Profile

Agent 不会一次性暴露所有工具，而是按 profile 选择性开放：

| Profile | 包含的工具 | 场景 |
|---------|-----------|------|
| `benchmark` | web_search, wiki_search, wiki_page, browser_open, browser_open_many, final_answer | 2Wiki / SimpleQA 默认 |
| `visual` | 上面 + 图像搜索、图转文 | SimpleVQA |
| `rich` / `full` | 全套浏览器交互 + 视觉 | 需要点击输入的页面 |
| `memory` / `self_retrieval` | benchmark + memory_search/list/get/create/update/delete | 让 Agent 自主检索和维护记忆 |
| `all` | 所有注册工具 | 调试用，不建议正式实验 |

命令行指定：`--tool-profile visual`

---

## 输出文件说明

每次运行会在 `--out` 指定的目录下生成：

```
logs/opd_eval/
├── 2wiki_baseline_<timestamp>/
│   ├── summary.json          # 命中率、F1、耗时等汇总
│   ├── results.jsonl         # 每道题的详细结果
│   └── runs.jsonl            # 完整轨迹（仅 --save-traces 时生成）
└── 2wiki_evolved_<timestamp>/
    ├── summary.json
    ├── results.jsonl
    └── runs.jsonl
```

`summary.json` 里最常看的字段：
- `exact_accuracy`：精确匹配准确率
- `f1_accuracy`：F1 准确率
- `n`：实际跑了多少题
- `avg_steps`：平均工具调用步数

---

## 记忆系统

evolved 模式的核心是长期记忆。记忆存储在 `SII_AGENT_MEMORY_ROOT`（默认 `logs/memory`）下：

- `lessons.jsonl`：从错误中提炼的经验教训，每条包含触发条件和解决策略
- `episodes.jsonl`：成功/失败的完整轨迹摘要

**记忆生成规则**：只有当模型出错、且根据反思总结的 lesson 能在重试中纠正错误时，这条 lesson 才会被保留。避免产生无效记忆。

**记忆检索**：evolved 模式下，每道新题会根据问题关键词从 lessons 中检索最相关的 k 条（`--memory-k` 控制数量）注入 prompt。

**记忆维护**：可通过 `--memory-maintenance-interval N` 开启定期清理，每 N 题做一次全量扫描，合并重复、删除过时记录。

---

## 常见问题

| 现象 | 排查方向 |
|------|----------|
| `APIConnectionError` | 模型服务没启动，或 `VLLM_BASE_URL` 端口对不上 |
| 模型一直输出文字不调工具 | `VLLM_ENABLE_THINKING` 没设为 0 |
| SGLang 启动报 OOM | 降低 `SGLANG_CONTEXT_LENGTH` 或减少 TP 数（意味着用更少卡） |
| evolved 效果比 baseline 差 | 这是可能的。先关掉 reflection/lessons（`unset SII_2WIKI_ENABLE_REFLECTION` 等），做 ablation |
| 搜索工具报超时 | 检查 search-proxy 是否在跑，SSH 隧道是否连通 |
| 浏览器工具报连接失败 | 检查 browser-service 是否在跑，SANDBOX_BASE_URL 是否正确 |
| 想提交代码但有大文件 | `logs/`、`data/`、`indexes/`、`saves/`、`Qwen*` 等已在 .gitignore，别手动 add |

---

## 其他任务

### SimpleQA

```bash
python -m evaluation.run_eval --task simpleqa --mode baseline --n 50 --concurrency 32
```

### SimpleVQA（图片问答）

```bash
python -m evaluation.run_eval \
  --task simplevqa \
  --mode baseline \
  --n 20 \
  --tool-profile visual \
  --concurrency 16
```

### BrowseComp-Plus

需要先下载 BM25 索引：

```bash
python -m scripts.download_browsecomp_index --out indexes
export BROWSECOMP_INDEX_PATH=indexes/bm25
```

然后：

```bash
python -m evaluation.run_browsecomp \
  --mode evolved \
  --n 20 \
  --concurrency 32 \
  --out logs/browsecomp
```

---

## OPD 偏好蒸馏（高级）

如果你要做模型微调而不只是评测，流程是：

1. 保存 baseline + evolved 的完整轨迹（`--save-traces`）
2. 用 `training/opd.py` 构造偏好数据对
3. 导出 LlamaFactory DPO 配置
4. 用 LlamaFactory 训练

```bash
# 1. 保存轨迹
python -m evaluation.run_eval --task 2wiki --split train --mode baseline --n 200 --save-traces --out logs/opd_traces
python -m evaluation.run_eval --task 2wiki --split train --mode evolved --n 200 --save-traces --out logs/opd_traces

# 2-3. 构造偏好数据 + 导出 LlamaFactory 配置
python -m training.opd \
  --runs logs/opd_traces/<baseline-run> logs/opd_traces/<evolved-run> \
  --out logs/opd/export \
  --expert llm \
  --pref-loss sigmoid \
  --model-name-or-path Qwen/Qwen3.5-9B \
  --lf-export-mode answer \
  --lf-template qwen3_5_nothink
```

---

## 语义 Judge（可选）

本地评分是 exact match / F1，粒度粗。如果有更大的模型（比如 Qwen3-32B）做 judge：

```bash
python -m evaluation.judge_semantic \
  --run-dirs logs/opd_eval/<baseline-dir> logs/opd_eval/<evolved-dir> \
  --base-url http://127.0.0.1:8005/v1 \
  --model Qwen3-32B \
  --concurrency 64
```

---

## 完整复现清单

简单列一下从头到尾要做的事：

1. 克隆代码、装依赖
2. 下载 Qwen3.5-9B 权重
3. 装 SGLang 并启动模型服务
4. 设置环境变量（LLM_BACKEND, VLLM_BASE_URL, VLLM_MODEL, VLLM_ENABLE_THINKING=0）
5. 跑 `python -m scripts.smoke` 确认链路
6. 跑 baseline 500 样本
7. 开 reflection + lessons，跑 evolved 500 样本
8. 比对 `summary.json` 里的 exact_accuracy 和 f1_accuracy

如果需要 train/test 分离评测：先在 train split 用 `--runtime-mode train` 积累记忆，再在 validation split 用 `--runtime-mode test` 冻结评测。

更多 2Wiki 历史结果和最佳配置见 `docs/2wiki_best_results.md`。
