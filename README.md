# SII-Agent · 自进化任务求解智能体

最小但完整的 "尝试 → 反思 → 进化" 闭环 Agent，用于 SimpleQA / 2Wiki / BrowseComp-Plus。

## 模块
```
agent/        ReAct 主循环 + Reflection
tools/        search / browser / wiki  (FastAPI 沙盒可独立部署)
harness/      运行控制：超时 / 步数 / 死循环检测 / 并发
memory/       短期 (trajectory) + 长期 (episodic + lessons) 记忆
evaluation/   SimpleQA / 2Wiki / BrowseComp-Plus 评测脚本
configs/      模型 / 工具 / 评测配置 (yaml)
scripts/      一键跑基线 / 进化版 / 打榜
```

## 快速开始
```bash
pip install -r requirements.txt
cp configs/.env.example .env
python -m scripts.build_wiki_index --max-docs 1000000
python -m scripts.build_wiki_fts --source data/wiki25/wiki25_sample.jsonl --out data/wiki25/wiki25_fts.sqlite
python -m scripts.build_browsecomp_fts --out data/browsecomp-plus/browsecomp_fts.sqlite
python -m scripts.smoke
python -m evaluation.run_eval --task simpleqa --mode baseline --n 50 --concurrency 32 --max-llm-tokens 1536
python -m evaluation.run_eval --task simpleqa --mode evolved --n 50 --concurrency 32 --evolve-batch-size 8 --max-llm-tokens 1536
python -m evaluation.run_browsecomp --mode evolved --n 20 --concurrency 32
```

## 模型 Backend
- `vllm`   :  本地 vLLM/SGLang 部署的 Qwen3.5-9B（默认/最终打榜）
- `azure`  :  Azure OpenAI (gpt-5.2 / gpt-5.4)，AAD 鉴权（开发期备用）

切换只需改 `.env` 的 `LLM_BACKEND`。

## 本地 Qwen3.5-9B vLLM
```bash
bash scripts/start_qwen_vllm.sh
```
默认用 `--enforce-eager` 和 8k context，避免 Qwen3.5 的 compile/cudagraph 启动卡住；正式长上下文实验可设置 `VLLM_MAX_MODEL_LEN=32000` 后重启。

## 搜索工具
- `web_search`: 默认 `SEARCH_BACKENDS=ddg,wiki`，先在线 DuckDuckGo，再追加/兜底离线 Wikipedia。
- `image_search`: DuckDuckGo Images 图像检索，返回图片 URL、来源页和标题，补齐“图搜文/图像证据定位”的工具入口。
- `wiki_search`: 离线 BM25/FTS，索引来自 `XLDDD/wiki25`。JSONL 数据保存在 `data/wiki25/wiki25_sample.jsonl`，推荐再构建 `data/wiki25/wiki25_fts.sqlite`，查询时走 SQLite FTS5 倒排索引。
- `browsecomp_search` / `browsecomp_get_document`: 面向 BrowseComp-Plus 固定语料的 SQLite FTS5 检索工具，输出官方 docid，便于计算 retrieval recall 和提交打榜结果。
- `browse`: 静态网页正文抽取；`browse_many`: 并发读取多 URL，适合一次性比对多个搜索结果页面。
- Playwright 沙盒浏览器：`browser_open` / `browser_text` / `browser_click` / `browser_type` / `browser_close` 支持真实页面访问、点击、输入和会话状态；如机器未安装 Chromium，先运行 `python -m playwright install chromium`。
- 工具沙盒：`uvicorn tools.server:app --host 0.0.0.0 --port 8080` 可通过 HTTP 暴露 `/tools`、`/call` 和有界并发的 `/call_many`。

## 并发评测 / 批量进化
- baseline 评测支持 `--concurrency N`，可直接并发调用本地 vLLM。
- evolved 评测支持 `--evolve-batch-size N`：每个 batch 先固定上一轮 memory 快照，再并发执行解题、反思和重试，batch 完成后写入的新 lessons 会进入下一批检索。
- `--max-llm-tokens` / `--max-llm-call-seconds` 给每步 LLM 调用硬上限，避免单个样本无限生成拖住整批。
- `--offset` 可切分 dev/eval；`--memory-root` + `--memory-mode fresh|read_write|read_only` 支持先在 dev 写长期记忆，再在 held-out eval 只读复用；`--no-reflection` 可隔离“只读记忆”效果；`--gold-reflection` 默认关闭，避免把 gold answer 泄漏进长期记忆；`--save-traces` 会把完整 trajectory 写入 `runs.jsonl`。

## BrowseComp-Plus 打榜
- `python -m scripts.build_browsecomp_fts` 会从 `Tevatron/browsecomp-plus-corpus` 构建固定语料 FTS 索引。
- `python -m evaluation.run_browsecomp --mode evolved --n 0 --concurrency 32` 会解密官方 query 集、本地运行 agent，并在 `logs/browsecomp/.../runs/` 生成官方兼容 JSON 文件。
- 输出目录同时包含 `browsecomp_plus_decrypted.jsonl` 和 `qrel_evidence.txt`，可配合 BrowseComp-Plus 官方 `scripts_evaluation/evaluate_run.py --input_dir <runs> --ground_truth <...> --qrel_evidence <...>` 做 judge 评分。
- `python -m evaluation.merge_browsecomp_runs --run-dirs <run-a> <run-b> ... --out <merged>` 可用无 gold 的 LLM selector 在多条 agent 轨迹答案中选择最终答案；`python -m evaluation.route_browsecomp_runs --primary-run <cheap-run> --fallback-run <selector-run> --out <routed>` 可只在廉价轨迹出现拒答、超长答案或搜索预算打满时切到更贵的 selector run。

## 评分项映射
| 评分维度 | 代码位置 |
|---|---|
| 工具：search / browser | `tools/` |
| ReAct + Harness | `agent/react.py`, `harness/` |
| Reflection | `agent/reflection.py` |
| Memory | `memory/` |
| 评测 / 进化对比 | `evaluation/` |
