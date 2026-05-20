# 本地 Wiki 搜索工具接入说明

数据来源：[XLDDD/wiki25](https://huggingface.co/datasets/XLDDD/wiki25)

工具暴露给 LLM 的两个函数：
- `wiki_search(query, top_k)` — BM25 检索，返回候选 (title / id / 摘要)
- `wiki_get(title | doc_id, max_chars)` — 按精确 title 或 id 取全文

---

## 一、初始化（只需一次）

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载 + 合并 + 探测格式
```bash
bash tools/download_wiki.sh
```
默认输出到 `./data/wiki25/`：
- 23 个分片（`wiki25_part_aaa..aaw`，约 24 GB）
- 合并后的 `wiki25_full.bin`
- 末尾会用 `file` / `xxd` / `head` 探测真实格式

### 3. 根据探测结果决定下一步

| 探测结果 | 操作 |
| --- | --- |
| **JSONL** (`{` 开头，每行一篇) | `mv data/wiki25/wiki25_full.bin data/wiki25/wiki25_full.jsonl` |
| **gzip** (`1f 8b`) | `gunzip -c wiki25_full.bin > wiki25_full.jsonl` |
| **bzip2** (`42 5a 68`) | `bunzip2 -c wiki25_full.bin > wiki25_full.jsonl` |
| **Parquet** (`PAR1`) | 见下方"非 JSONL 适配"小节 |
| **其他** | 把 `head -c 500` 的输出贴给我，我适配解析 |

### 4. 构建 BM25 索引
```bash
# 全量（24GB，依内存约 30 分钟–几小时）
python tools/build_wiki_index.py \
    --corpus  data/wiki25/wiki25_full.jsonl \
    --out-dir data/wiki25/index \
    --summary-chars 1000

# 冒烟测试：先只索引 5 万篇
python tools/build_wiki_index.py \
    --corpus  data/wiki25/wiki25_full.jsonl \
    --out-dir data/wiki25/index_smoke \
    --max-docs 50000
```

产物：`data/wiki25/index/{bm25.pkl, meta.jsonl, config.json}`

> ⚠️ 全量 BM25 模型在内存里很大（>10GB 量级，取决于词表）。如内存吃紧，先用 `--max-docs` 跑通流程。

---

## 二、独立测试工具

```bash
# 检索
python -m tools.wiki_tool "Albert Einstein relativity"

# 取全文
python -m tools.wiki_tool --get "Albert Einstein"
```

环境变量：
- `WIKI_INDEX_DIR`（默认 `data/wiki25/index`）— 自定义索引目录

---

## 三、跑 Agent

```bash
python task_runner.py -i "爱因斯坦在 1905 年发表了哪些重要论文？" -t einstein01
```

注意：
- `DISABLE_TOOLS=1` 仍可关闭工具，仅做 LLM 通路调试
- 默认 `WIKI_INDEX_DIR=data/wiki25/index`，索引放在别处时：
  ```bash
  WIKI_INDEX_DIR=/path/to/index python task_runner.py -i "..."
  ```

---

## 四、非 JSONL 适配小指引

`build_wiki_index.py` 假定输入是 **JSONL**（一行一篇）。如果探测出来是 Parquet 等其他格式，最省事的做法是先转成 JSONL：

**Parquet → JSONL**：
```python
import pandas as pd, json
df = pd.read_parquet("data/wiki25/wiki25_full.bin")
with open("data/wiki25/wiki25_full.jsonl","w",encoding="utf-8") as f:
    for r in df.to_dict(orient="records"):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

字段名识别（见 `wiki_tool.py` / `build_wiki_index.py`）支持：
- title: `title` / `page_title` / `name`
- text:  `text` / `content` / `body` / `article` / `passage`
- id:    `id` / `page_id` / `wiki_id` / `doc_id`

如果你的字段名不在上面，告诉我，我加进去。
