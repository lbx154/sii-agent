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
python -m scripts.build_wiki_index --max-docs 50000
python -m scripts.smoke
python -m evaluation.run_eval --task simpleqa --mode baseline --n 50
python -m evaluation.run_eval --task simpleqa --mode evolved --n 50
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
- `wiki_search`: 离线 BM25，索引来自 `XLDDD/wiki25`。默认先构建 5 万条轻量索引用于调试，正式实验可扩大或全量构建。

## 评分项映射
| 评分维度 | 代码位置 |
|---|---|
| 工具：search / browser | `tools/` |
| ReAct + Harness | `agent/react.py`, `harness/` |
| Reflection | `agent/reflection.py` |
| Memory | `memory/` |
| 评测 / 进化对比 | `evaluation/` |
