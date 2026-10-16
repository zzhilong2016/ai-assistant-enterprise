# -*- coding: utf-8 -*-
"""
RAG 第一课: 最小可用知识库系统
流程: 加载文档 → 切块 → 向量化(智谱embedding) → 存入向量库 → 检索 → 喂给模型生成
说明: 向量库用numpy手写(余弦相似度), 不依赖chromadb——先懂原理, 框架后上
运行: python rag_demo.py
"""
import os
import glob
import json
import numpy as np
from openai import OpenAI

# ========== 配置区 ========
# 密钥脱敏: 从仓库根目录的 .env 读取(该文件已被 .gitignore 拦截, 不入库)
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE, encoding="utf-8"):
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

API_KEY = os.environ["ZHIPU_API_KEY"]
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
CHAT_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-2"
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "知识库")
STORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store.json")
# ==========================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)


# ---------- 第一步: 加载文档 ----------
def load_docs(kb_dir: str) -> list[dict]:
    """读取目录下所有md文件, 返回 [{'filename':..., 'content':...}]"""
    docs = []
    for path in glob.glob(os.path.join(kb_dir, "*.md")):
        with open(path, encoding="utf-8") as f:
            docs.append({"filename": os.path.basename(path), "content": f.read()})
    return docs


# ---------- 第二步: 切块(按标题/段落切, 保持语义完整) ----------
def split_chunks(doc: dict, max_chars: int = 200) -> list[dict]:
    """
    逐行扫描: 跳过标题行, 以空行/超长为边界切块, 保持语义完整。
    每个chunk带上来源文件名, 以后做引用溯源。
    """
    chunks, buf = [], ""
    for line in doc["content"].split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):      # 空行和标题行: 作为切分边界, 不入正文
            if buf:
                chunks.append({"text": buf, "source": doc["filename"]})
                buf = ""
            continue
        if len(buf) + len(line) > max_chars:      # 超长: 先落盘已有内容
            if buf:
                chunks.append({"text": buf, "source": doc["filename"]})
            buf = line
        else:
            buf = buf + "\n" + line if buf else line
    if buf:
        chunks.append({"text": buf, "source": doc["filename"]})
    return chunks


# ---------- 第三步: 构建向量库(离线, 只需跑一次) ----------
class SimpleVectorStore:
    """手写向量库: 存json, 检索用numpy余弦相似度。麻雀虽小五脏俱全。"""

    def __init__(self, store_file: str):
        self.store_file = store_file
        self.items = []          # [{"id", "text", "source", "embedding"}]
        if os.path.exists(store_file):
            with open(store_file, encoding="utf-8") as f:
                self.items = json.load(f)
            print(f"向量库已存在({len(self.items)}块), 如需重建请删除 {store_file}")

    def add(self, chunks: list[dict]):
        """向量化文本块并入库"""
        print(f"开始向量化 {len(chunks)} 个文本块...")
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            resp = client.embeddings.create(model=EMBED_MODEL, input=[c["text"] for c in batch])
            for j, d in enumerate(resp.data):
                self.items.append({
                    "id": f"chunk_{i + j}",
                    "text": batch[j]["text"],
                    "source": batch[j]["source"],
                    "embedding": d.embedding,
                })
            print(f"  已处理 {min(i + batch_size, len(chunks))}/{len(chunks)}")
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False)
        print("向量库构建完成")

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """问题向量化 → 计算与所有块的余弦相似度 → 返回top_k"""
        q_emb = np.array(client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding)
        matrix = np.array([it["embedding"] for it in self.items])       # (N, 1024)
        # 余弦相似度 = 点积 / (模长之积)
        sims = matrix @ q_emb / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_emb))
        top_idx = np.argsort(-sims)[:top_k]                              # 相似度降序
        return [
            {"text": self.items[i]["text"], "source": self.items[i]["source"], "score": float(sims[i])}
            for i in top_idx
        ]


# ---------- 第五步: 增强生成 ----------
def answer(vector_store, question: str, top_k: int = 3) -> str:
    hits = vector_store.search(question, top_k)

    # 检索结果拼成"参考资料", 要求模型只依据资料回答 + 标注来源
    context = "\n\n".join(f"【资料{i+1}】(出自{h['source']})\n{h['text']}" for i, h in enumerate(hits))
    prompt = f"""你是企业产品咨询助手。请严格根据以下参考资料回答用户问题。
要求:
1. 答案必须出自资料, 资料里没有的信息明确说"根据现有资料无法确定";
2. 回答末尾标注引用了哪份资料(资料编号);
3. 回答简明扼要。

参考资料:
{context}

用户问题: {question}"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的企业产品咨询助手, 只依据给定资料回答, 不编造。"},
            {"role": "user", "content": prompt},
        ],
    )
    # 控制台同时展示检索过程(教学观察用)
    print("-" * 50)
    print(f"[检索] 问题: {question}")
    for i, h in enumerate(hits):
        print(f"  命中{i+1}: {h['source']} (相似度 {h['score']:.4f}) {h['text'][:40]}...")
    print("-" * 50)
    return resp.choices[0].message.content


# ---------- 主流程 ----------
if __name__ == "__main__":
    docs = load_docs(KB_DIR)
    all_chunks = []
    for d in docs:
        all_chunks.extend(split_chunks(d))
    print(f"加载 {len(docs)} 份文档, 切成 {len(all_chunks)} 块")

    vector_store = SimpleVectorStore(STORE_FILE)
    if not vector_store.items:               # 首次运行: 构建向量库
        vector_store.add(all_chunks)

    # 命令行问答循环
    print("\n=== RAG 知识库问答 (输入quit退出) ===")
    while True:
        q = input("\n你的问题: ").strip()
        if q in ("quit", "退出"):
            break
        print(f"\n回答: {answer(q, vector_store)}")
