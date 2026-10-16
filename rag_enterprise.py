# -*- coding: utf-8 -*-
"""
RAG 企业版 v1: 权限隔离 + 混合检索 + Rerank精排 + 评估集
对照演示版(rag_demo.py)新增:
  ① 权限元数据: 检索前按用户身份过滤(企业命门, 项目一核心)
  ② 混合检索: 向量 + BM25关键词, RRF融合(治"价格漏检")
  ③ Rerank: 智谱官方rerank模型精排
  ④ 评估集: golden set 对比 纯向量 vs 混合+精排
运行: python rag_enterprise.py           # 交互问答
      python rag_enterprise.py eval      # 跑评估对比
"""
import os
import sys
import glob
import json
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ========== 配置区 ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv() -> None:
    """极简 .env 读取(不依赖第三方库): KEY=VALUE 逐行, 已存在的环境变量优先"""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

def _require_key(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if not key:
        raise RuntimeError(f"缺少环境变量 {name}: 请在 .env 文件或系统环境变量中配置")
    return key

API_KEY = _require_key("ZHIPU_API_KEY")   # 密钥不入库, 见 .env
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
CHAT_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-2"
RERANK_MODEL = "rerank"
KB_DIR = os.path.join(BASE_DIR, "知识库")
STORE_FILE = os.path.join(BASE_DIR, "vector_store_enterprise.json")

# 模拟已登录用户及其权限角色(真实系统来自SSO/组织架构)
CURRENT_USER = {"name": "张三", "roles": ["all", "政企事业部"]}
FINANCE_USER = {"name": "李财务", "roles": ["all", "财务部"]}   # 财务视角
# CURRENT_USER = FINANCE_USER   # 换成财务试试
# ==========================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)


# ---------- 加载与切块(带权限元数据) ----------
def load_docs_with_acl() -> list[dict]:
    manifest = json.load(open(os.path.join(KB_DIR, "kb_manifest.json"), encoding="utf-8"))
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "*.md")):
        fname = os.path.basename(path)
        if fname == "kb_manifest.json":
            continue
        acl = manifest.get(fname, {}).get("acl", ["all"])
        with open(path, encoding="utf-8") as f:
            docs.append({"filename": fname, "content": f.read(), "acl": acl})
    return docs

def split_chunks(doc: dict, max_chars: int = 200) -> list[dict]:
    chunks, buf = [], ""
    for line in doc["content"].split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            if buf:
                chunks.append({"text": buf, "source": doc["filename"], "acl": doc["acl"]})
                buf = ""
            continue
        if len(buf) + len(line) > max_chars:
            if buf:
                chunks.append({"text": buf, "source": doc["filename"], "acl": doc["acl"]})
            buf = line
        else:
            buf = buf + "\n" + line if buf else line
    if buf:
        chunks.append({"text": buf, "source": doc["filename"], "acl": doc["acl"]})
    return chunks


def has_permission(chunk_acl: list[str], user_roles: list[str]) -> bool:
    """权限判断: 用户任一角色命中文档ACL即可见。检索前调用——企业版第一原则"""
    return bool(set(chunk_acl) & set(user_roles))


# ---------- 企业级向量库: 向量 + BM25 双索引 ----------
class EnterpriseVectorStore:
    def __init__(self, store_file: str):
        self.store_file = store_file
        self.items = []
        self.bm25 = None
        if os.path.exists(store_file):
            self.items = json.load(open(store_file, encoding="utf-8"))
            self._build_bm25()
            print(f"向量库已加载({len(self.items)}块, 双索引就绪)")

    def _build_bm25(self):
        tokenized = [list(it["text"]) for it in self.items]   # 中文按字切, 简单有效
        self.bm25 = BM25Okapi(tokenized)

    def add(self, chunks: list[dict]):
        print(f"开始向量化 {len(chunks)} 个文本块...")
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            resp = client.embeddings.create(model=EMBED_MODEL, input=[c["text"] for c in batch])
            for j, d in enumerate(resp.data):
                self.items.append({"id": f"chunk_{i+j}", "text": batch[j]["text"],
                                   "source": batch[j]["source"], "acl": batch[j]["acl"],
                                   "embedding": d.embedding})
            print(f"  已处理 {min(i+batch_size, len(chunks))}/{len(chunks)}")
        json.dump(self.items, open(self.store_file, "w", encoding="utf-8"), ensure_ascii=False)
        self._build_bm25()
        print("向量库构建完成(向量+BM25双索引)")

    # ★ 核心: 混合检索 = 权限过滤 → 双路召回 → RRF融合 → 可选Rerank
    def search(self, question: str, user_roles: list[str], top_k: int = 3,
               use_hybrid: bool = True, use_rerank: bool = True, recall_n: int = 10) -> list[dict]:
        # 第一原则: 先按权限过滤, 被过滤的块不参与任何检索
        visible = [i for i, it in enumerate(self.items) if has_permission(it["acl"], user_roles)]

        # 路1: 向量检索 top_recall
        q_emb = np.array(client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding)
        vis_matrix = np.array([self.items[i]["embedding"] for i in visible])
        sims = vis_matrix @ q_emb / (np.linalg.norm(vis_matrix, axis=1) * np.linalg.norm(q_emb))
        vec_ranked = [visible[i] for i in np.argsort(-sims)[:recall_n]]

        candidates = {idx: None for idx in vec_ranked}

        # 路2: BM25关键词检索(向量漏掉的关键词, 这里能捞回来)
        if use_hybrid:
            # get_scores 返回全库得分, 先按 visible 全局下标取出可见块的得分再排序
            bm_scores = np.array(self.bm25.get_scores(list(question)))[visible]
            bm_ranked = [visible[i] for i in np.argsort(-bm_scores)[:recall_n]]
            for idx in bm_ranked:
                candidates.setdefault(idx, None)

        # RRF融合: 每路排名倒数求和, 名次越靠前分越高, 天然平衡两路尺度
        rrf = {}
        for rank, idx in enumerate(vec_ranked):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)
        if use_hybrid:
            for rank, idx in enumerate(bm_ranked):
                rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)
        fused = sorted(rrf, key=rrf.get, reverse=True)[:recall_n]

        # 精排: 重算"问题与候选块"的相关性, 取最终top_k
        if use_rerank:
            # ★ 关键细节: 块文本前拼上来源文档名。裸块缺乏主题信号,
            #   实测不加标题时"价格"类问题会被资费文档抢走top3
            docs = [f"[出自{self.items[i]['source']}] {self.items[i]['text']}" for i in fused]
            try:
                r = requests.post(f"{BASE_URL}/rerank",
                                  headers={"Authorization": f"Bearer {API_KEY}"},
                                  json={"model": RERANK_MODEL, "query": question, "documents": docs},
                                  timeout=30).json()
                fused = [fused[x["index"]] for x in sorted(r["results"], key=lambda x: -x["relevance_score"])][:top_k]
            except Exception:
                fused = fused[:top_k]   # rerank失败降级: 直接用融合结果
        else:
            fused = fused[:top_k]

        return [{"text": self.items[i]["text"], "source": self.items[i]["source"],
                 "acl": self.items[i]["acl"]} for i in fused]


import requests   # rerank调用用(放这里避免与上面逻辑混淆)

# ---------- 生成(带引用溯源) ----------
def answer(vector_store, question: str, user: dict, **search_kw) -> str:
    hits = vector_store.search(question, user["roles"], **search_kw)
    print("-" * 50)
    print(f"[检索] 用户={user['name']} 可见块过滤后, 命中:")
    for i, h in enumerate(hits):
        print(f"  {i+1}. [{h['source']}] {h['text'][:40]}...")
    print("-" * 50)

    context = "\n\n".join(f"【资料{i+1}】(出自{h['source']})\n{h['text']}" for i, h in enumerate(hits))
    prompt = f"""你是企业产品咨询助手。请严格根据以下参考资料回答用户问题。
要求: 1.答案必须出自资料, 资料没有就明确说"根据现有资料无法确定"; 2.末尾标注引用的资料编号; 3.简明扼要。

参考资料:
{context}

用户问题: {question}"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": "你是严谨的企业咨询助手, 只依据给定资料回答, 不编造。"},
                  {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ---------- 评估集: golden set ----------
# 前5条是召回测试(文档对所有人可见); 第6条是★权限测试题★——
# 内部结算价.md 只对财务/总监可见, 张三根本不该检索到它, 检索不到=权限过滤生效
GOLDEN_SET = [
    {"q": "5G专网有哪几种模式, 分别什么价格?", "expect_source": "5G专网产品.md"},
    {"q": "商务快线500M多少钱, 包含什么?",     "expect_source": "政企宽带资费.md"},
    {"q": "云视讯最多支持多少方在线?",          "expect_source": "云视讯产品.md"},
    {"q": "云视讯专业版怎么收费?",              "expect_source": "云视讯产品.md"},
    {"q": "办理企业专线需要什么材料?",          "expect_source": "政企宽带资费.md"},
]
PERMISSION_CASE = {"q": "5G专网的内部结算价是多少?", "expect_source": "内部结算价.md",
                   "allowed_roles": ["财务部", "政企事业部总监"]}

def _recall_at_3(vector_store, user, kw):
    hits_cnt = 0
    for case in GOLDEN_SET:
        hits = vector_store.search(case["q"], user["roles"], top_k=3, **kw)
        if any(case["expect_source"] == h["source"] for h in hits):
            hits_cnt += 1
    return hits_cnt, len(GOLDEN_SET)

def evaluate(vector_store, user):
    """指标: recall@3 = 期望文档出现在top3命中里的比例 + 权限隔离验证"""
    for mode, kw in [("纯向量", dict(use_hybrid=False, use_rerank=False)),
                     ("混合+精排", dict(use_hybrid=True, use_rerank=True))]:
        hits_cnt, total = _recall_at_3(vector_store, user, kw)
        print(f"{mode}: recall@3 = {hits_cnt}/{total} ({hits_cnt/total*100:.0f}%)")

    # 权限测试: 张三问结算价, 应检索不到; 财务问, 应能检索到
    print("\n--- 权限隔离验证 ---")
    hits = vector_store.search(PERMISSION_CASE["q"], user["roles"], top_k=3)
    blocked = not any(PERMISSION_CASE["expect_source"] == h["source"] for h in hits)
    can_see = bool(set(PERMISSION_CASE["allowed_roles"]) & set(user["roles"]))
    print(f"{user['name']}: {'✓ 正确拦截(无权不可见)' if blocked else '✗ 泄露了!'}")
    hits = vector_store.search(PERMISSION_CASE["q"], FINANCE_USER["roles"], top_k=3)
    got = any(PERMISSION_CASE["expect_source"] == h["source"] for h in hits)
    print(f"{FINANCE_USER['name']}: {'✓ 正常检索到(有权可见)' if got else '✗ 该看到却没看到!'}")


if __name__ == "__main__":
    docs = load_docs_with_acl()
    chunks = []
    for d in docs:
        chunks.extend(split_chunks(d))
    print(f"加载 {len(docs)} 份文档(含权限元数据), 切成 {len(chunks)} 块")

    vs = EnterpriseVectorStore(STORE_FILE)
    if not vs.items:
        vs.add(chunks)

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        print(f"\n=== 评估 (当前用户: {CURRENT_USER['name']}, 角色: {CURRENT_USER['roles']}) ===")
        evaluate(vs, CURRENT_USER)
    else:
        print(f"\n=== 企业知识库问答 (当前用户: {CURRENT_USER['name']}, quit退出) ===")
        while True:
            q = input("\n你的问题: ").strip()
            if q in ("quit", "退出"):
                break
            print(f"\n回答: {answer(vs, q, CURRENT_USER)}")
