# SII-Agent · 自进化任务求解智能体

最小但完整的 "尝试 → 反思 → 进化" 闭环 Agent，用于 SimpleQA / SimpleVQA / 2Wiki / BrowseComp-Plus。

## 模块
```
agent/        ReAct 主循环 + Reflection
tools/        search / browser / wiki  (FastAPI 沙盒可独立部署)
harness/      运行控制：超时 / 步数 / 死循环检测 / 并发
memory/       短期 (trajectory) + 长期 (episodic + lessons) 记忆
evaluation/   SimpleQA / SimpleVQA / 2Wiki / BrowseComp-Plus 评测脚本
configs/      模型 / 工具 / 评测配置 (yaml)
scripts/      一键跑基线 / 进化版 / 打榜
```

## 快速开始
```bash
pip install -r requirements.txt
cp configs/.env.example .env
python -m scripts.build_wiki_index --max-docs 1000000
python -m scripts.build_wiki_fts --source data/wiki25/wiki25_sample.jsonl --out data/wiki25/wiki25_fts.sqlite
python -m scripts.download_browsecomp_index --out indexes
python -m scripts.smoke
python -m evaluation.run_eval --task simpleqa --mode baseline --n 50 --concurrency 32 --max-llm-tokens 1536
python -m evaluation.run_eval --task simpleqa --mode evolved --n 50 --concurrency 32 --evolve-batch-size 8 --max-llm-tokens 1536
python -m evaluation.run_browsecomp --mode evolved --n 20 --concurrency 32
```

## 模型 Backend
- `vllm`   :  本地 vLLM/SGLang 部署的 Qwen3.5-9B（默认/最终打榜）
- `azure`  :  Azure OpenAI (gpt-5.2 / gpt-5.4)，AAD 鉴权（开发期备用）

切换只需改 `.env` 的 `LLM_BACKEND`。

## 本地 Qwen3.5-9B SGLang
赛题环境只允许 SGLang 时，使用 OpenAI-compatible `/v1` 接口：

```bash
python -m venv --system-site-packages /root/sglang-venv
/root/sglang-venv/bin/python -m pip install -U pip setuptools wheel
/root/sglang-venv/bin/python -m pip install 'sglang[all]==0.5.11'

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_PYTHON=/root/sglang-venv/bin/python \
SGLANG_MODEL=/root/sii-agent/Qwen3.5-9B \
SGLANG_PORT=8004 \
SGLANG_SERVED_MODEL_NAME=Qwen3.5-9B \
bash scripts/start_qwen_sglang.sh

export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://127.0.0.1:8004/v1
export VLLM_MODEL=Qwen3.5-9B
export VLLM_ENABLE_THINKING=0
```

注意：SGLang 的 Qwen3.5 thinking 输出需要显式关闭，否则短 `max_tokens` 下可能只返回 `reasoning_content`。`start_qwen_sglang.sh` 默认用 `--mm-attention-backend sdpa` 避开本机 `flash_attn` ABI 不兼容问题。

### SGLang + OPD LoRA
完整 v13 LoRA 包含 SGLang 0.5.11 当前不支持的 Qwen3.5 linear-attention target（`in_proj_a/b/qkv/z`、`out_proj`），直接加载会失败。可先生成只保留 SGLang 支持模块的 adapter 做运行时实验：

```bash
/root/sglang-venv/bin/python scripts/create_sglang_supported_lora.py \
  --src saves/qwen35-9b/lora/v13_step_final_opd_32k_rank8_beta03_lr2e6_epoch2_sigmoid \
  --dst saves/qwen35-9b/lora/v13_step_final_opd_32k_sglang_supported \
  --base-model /root/sii-agent/Qwen3.5-9B

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_PYTHON=/root/sglang-venv/bin/python \
SGLANG_MODEL=/root/sii-agent/Qwen3.5-9B \
SGLANG_PORT=8004 \
SGLANG_SERVED_MODEL_NAME=sii-opd-v13-sglang \
SGLANG_LORA_NAME=sii-opd-v13-sglang \
SGLANG_LORA_PATH=/root/sii-agent/saves/qwen35-9b/lora/v13_step_final_opd_32k_sglang_supported \
bash scripts/start_qwen_sglang.sh

export VLLM_BASE_URL=http://127.0.0.1:8004/v1
export VLLM_MODEL=sii-opd-v13-sglang
export VLLM_ENABLE_THINKING=0
```

这个 pruned adapter 不是完整 v13 LoRA 等价物；若赛题要求完整 LoRA 效果，需要后续尝试离线 merge 成完整 HF 权重或等待/修改 SGLang 对 Qwen3.5 linear-attention LoRA target 的支持。

## 本地 Qwen3.5-9B vLLM
```bash
bash scripts/start_qwen_vllm.sh
```
默认用 `--enforce-eager` 和 8k context，避免 Qwen3.5 的 compile/cudagraph 启动卡住；正式长上下文实验可设置 `VLLM_MAX_MODEL_LEN=32000` 后重启。

## 搜索工具
- `web_search`: 默认 `SEARCH_BACKENDS=ddg,wiki`，在线 DuckDuckGo 与离线 Wikipedia 等多个后端会并发执行并按配置顺序合并结果，降低搜索耗时。
- `image_search`: DuckDuckGo Images 文本搜图，返回图片 URL、来源页和标题；它不是反向图搜图。
- `wiki_search`: 离线 BM25/FTS，索引来自 `XLDDD/wiki25`。JSONL 数据保存在 `data/wiki25/wiki25_sample.jsonl`，推荐再构建 `data/wiki25/wiki25_fts.sqlite`，查询时走 SQLite FTS5 倒排索引。
- `search`: 面向 BrowseComp-Plus 固定语料的官方兼容本地检索工具，默认使用官方 Pyserini/Lucene BM25 索引 `indexes/bm25`，固定返回 top-5 JSON 结果（docid、score、512-token snippet），便于计算 retrieval recall 和提交打榜结果；`get_document` 和旧 `browsecomp_*` 名称仅作为兼容入口保留。
- `browse`: 静态网页正文抽取；`browse_many`: 并发读取多 URL，适合一次性比对多个搜索结果页面。
- Playwright 沙盒浏览器：`browser_open` / `browser_open_many` / `browser_text` / `browser_click` / `browser_type` / `browser_close` 支持真实页面访问、并发打开、点击、输入和会话状态；如机器未安装 Chromium，先运行 `python -m playwright install chromium`。
- All-in-One 在线搜索沙盒可作为可选浏览器后端：先运行 `docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest`，再设置 `AIO_SANDBOX_BASE_URL=http://127.0.0.1:8080`。配置后 `browser_*` 工具会通过沙盒的 CDP 浏览器执行；未配置时自动使用本地 Playwright。
- `image_to_text`: 使用 `.env` 中 OpenAI-compatible VLM endpoint 对图片 URL/本地图片做 OCR、视觉线索和候选实体描述。
- `visual_web_search`: 面向 SimpleVQA 等视觉事实题的封装工具；先用 VLM 产出 OCR/视觉线索和多个候选，再对候选及非实体 OCR 线索做 web/wiki 检索，最后返回紧凑的答案建议、置信度和反证说明，减少“第一眼看错后搜索强化错误”的问题。
- 默认 ReAct 工具集保持轻量：`web_search,wiki_search,browse,browse_many,final_answer`；视觉问答建议设置 `SII_AGENT_TOOL_PROFILE=visual` 或评测传 `--tool-profile visual`；需要展示完整图搜/浏览器能力时设置 `SII_AGENT_TOOL_PROFILE=rich`；需要暴露所有已注册工具时设置 `SII_AGENT_TOOL_PROFILE=all` 或评测时传 `--tool-profile all`。
- 工具沙盒：`uvicorn tools.server:app --host 0.0.0.0 --port 8080` 可通过 HTTP 暴露 `/tools`、`/call` 和有界并发的 `/call_many`。

## 并发评测 / 批量进化
- baseline 评测支持 `--concurrency N`，可直接并发调用本地 vLLM。
- evolved 评测支持 `--evolve-batch-size N`：每个 batch 先固定上一轮 memory 快照，再并发执行解题、反思和重试，batch 完成后写入的新 lessons 会进入下一批检索。
- 长短记忆：长期记忆由 evolved 模式写入 `episodes.jsonl` / `lessons.jsonl` 并在后续样本检索注入；短期工作记忆可用 `--short-memory` 开启，只在单次 ReAct 尝试内压缩已尝试查询、docid/证据片段和死路，不持久化、不读取 gold，默认关闭以保持基线可复现。
- `--max-llm-tokens` / `--max-llm-call-seconds` 给每步 LLM 调用硬上限，避免单个样本无限生成拖住整批。
- `--offset` 可切分 dev/eval；`--memory-root` + `--memory-mode fresh|read_write|read_only` 支持先在 dev 写长期记忆，再在 held-out eval 只读复用；`--no-reflection` 可隔离“只读记忆”效果；`--gold-reflection` 默认关闭，避免把 gold answer 泄漏进长期记忆；`--save-traces` 会把完整 trajectory 写入 `runs.jsonl`。

## BrowseComp-Plus 打榜
- `python -m scripts.download_browsecomp_index --out indexes` 会下载官方 `Tevatron/browsecomp-plus-indexes` BM25/Lucene 索引；旧的 `scripts.build_browsecomp_fts` 仅作为 SQLite FTS5 兼容路径保留。
- `python -m evaluation.run_browsecomp --mode evolved --n 0 --concurrency 32` 会解密官方 query 集、本地运行 agent，并在 `logs/browsecomp/.../runs/` 生成官方兼容 JSON 文件。
- 输出目录同时包含 `browsecomp_plus_decrypted.jsonl` 和 `qrel_evidence.txt`，可配合 BrowseComp-Plus 官方 `scripts_evaluation/evaluate_run.py --input_dir <runs> --ground_truth <...> --qrel_evidence <...>` 做 judge 评分。
- `python -m evaluation.merge_browsecomp_runs --run-dirs <run-a> <run-b> ... --out <merged>` 可用无 gold 的 LLM selector 在多条 agent 轨迹答案中选择最终答案；`python -m evaluation.route_browsecomp_runs --primary-run <cheap-run> --fallback-run <selector-run> --out <routed>` 可只在廉价轨迹出现拒答、超长答案或搜索预算打满时切到更贵的 selector run。

## GPT-5.4 OPD / 偏好蒸馏
- 数据：2Wiki 有 `train/validation/test`，可用 `--split train` 做 OPD 训练；当前 SimpleQA loader 没有 public train split，只有 `test` 和很小的 `few_shot`，不要把 SimpleQA test 同时用于训练和评测。
- 专家：`python -m training.opd` 默认读取 `.env` 中的 `OPD_EXPERT_MODEL` / `AZURE_OPENAI_DEPLOYMENT`，使用 GPT-5.4 给多条 agent 轨迹打 chosen/rejected 偏好。
- 数学口径：这里的 OPD 不是自定义 loss，而是“离线偏好数据上的 KL-regularized policy distillation”。默认用 DPO/sigmoid：
  `-log σ(β[(logπθ(y+)−logπref(y+)) − (logπθ(y−)−logπref(y−))])`，即用 reference model 的 reverse-KL 约束把 GPT-5.4 偏好蒸馏进 Qwen student。LlamaFactory 的 DPO trainer 内部也设置 `f_divergence_type="reverse_kl"`。
- 训练：脚本会导出 LlamaFactory ranking 数据、`dataset_info.json`、DPO 配置和 `train_llamafactory_opd.sh`；默认 `pref_loss: sigmoid`。默认 `--lf-export-mode answer` 只优化最终答案文本，最稳；`final_tool` / `action` 可导出 Qwen3.5 tool-call 格式用于实验，但会更强地改变 ReAct 工具策略。ORPO/SimPO 是 reference-free ablation，不作为 OPD 默认。
- 两卡训练：生成的 `train_llamafactory_opd.sh` 默认 `CUDA_VISIBLE_DEVICES=4,5 FORCE_TORCHRUN=1`，会在缺少 `llamafactory-cli` 时 clone/install `hiyouga/LLaMA-Factory` 到 `third_party/`。

示例：
```bash
python -m evaluation.run_eval --task 2wiki --split train --mode baseline --n 200 --concurrency 32 --save-traces --out logs/opd_traces
python -m evaluation.run_eval --task 2wiki --split train --mode evolved --n 200 --concurrency 32 --evolve-batch-size 8 --save-traces --out logs/opd_traces
python -m training.opd --runs logs/opd_traces/<baseline-run> logs/opd_traces/<evolved-run> --out logs/opd/gpt54_2wiki --expert llm --pref-loss sigmoid --model-name-or-path Qwen/Qwen3.5-9B --lf-export-mode answer --lf-template qwen3_5_nothink
bash logs/opd/gpt54_2wiki/llamafactory/train_llamafactory_opd.sh
python -m evaluation.apply_opd_policy --policy logs/opd/gpt54_2wiki/opd_policy.json --runs <heldout-run-a> <heldout-run-b> --out logs/opd_eval/heldout
```

## slime 在线 Agent OPD
- 目标：用 `Qwen3.5-9B` 作为 student，在 SII-Agent harness 里做 on-policy tool-use rollout；`Qwen3.5-27B` 作为 SGLang teacher，提供轨迹 token-level logprob，slime 用 `--use-opd --opd-type sglang` 蒸馏 student 的 agent 行为。
- 训练集：默认从 2Wiki train 切片导出 prompt；评测集：BrowseComp-Plus test，工具白名单切换为 `search,final_answer`，避免把开放搜索带进固定语料评测。
- 关键接入点：`training.slime_sii_rollout.generate` 实现多步 ReAct/tool rollout 并返回 slime `Sample`；`teacher_logprob_rm` / `post_process_rewards` 复用 slime OPD teacher logprob 路径。

示例：
```bash
bash scripts/slime/prepare-sii-opd-data.sh
bash scripts/slime/convert-qwen3.5-9B.sh
PROMPT_DATA=data/slime/sii_2wiki_train_512.jsonl \
EVAL_PROMPT_DATA=data/slime/sii_browsecomp_test_64.jsonl \
bash scripts/slime/run-sii-qwen3.5-9B-opd.sh
```

## 评分项映射
| 评分维度 | 代码位置 |
|---|---|
| 工具：search / browser | `tools/` |
| ReAct + Harness | `agent/react.py`, `harness/` |
| Reflection | `agent/reflection.py` |
| Memory | `memory/` |
| 评测 / 进化对比 | `evaluation/` |
