# 全能助手 · 企业级 LLM 应用

基于大模型 API 的全能助手，覆盖 **对话 + 工具调用 + 企业知识库问答（RAG）** 三大能力。
学习路线：LLM API → Prompt 工程 → Function Calling → RAG（权限隔离/混合检索/精排/评估）→ 后续 Agent 编排。

## 功能亮点

| 模式 | 能力 | 关键技术 |
|---|---|---|
| 💬 自由对话 | 股票实时行情查询（真实数据·新浪财经接口） | Function Calling、流式输出、幻觉护栏 |
| 📚 知识库问答 | 基于企业文档回答，无权文档检索不到 | 权限隔离（ACL）、混合检索、Rerank 精排 |

**工程化特性**（不限于 Demo）：

- **多通道切换**：智谱官方 / CCSWIC 中转，一键切换容灾
- **权限隔离**：检索前按用户角色过滤文档块——张三问"内部结算价"得到"无法确定"，财务问同一问题得到真实数据
- **混合检索**：向量（embedding-2）+ BM25 关键词，RRF 融合，专治"纯向量漏关键词"
- **Rerank 精排**：智谱 rerank 模型二次排序（块文本带文档标题前缀，避免主题信号丢失）
- **可观测**：右侧调试面板实时显示工具调用记录 / 检索命中明细；黑匣子日志 `llm_debug.log` 落盘全部请求
- **兜底机制**：超时容错（断线保半截）、工具调用 5 轮上限防死循环、幻觉护栏（无工具调用却声称有查询结果 → 自动打回）
- **评估集**：golden set 对比"纯向量 vs 混合+精排"recall@3，含权限隔离双断言

## 快速开始

```bash
# 1. 安装依赖
pip install openai gradio numpy rank_bm25 requests

# 2. 配置密钥(不入库): 仓库根目录新建 .env
#    ZHIPU_API_KEY=你的智谱key
#    CCSWIC_API_KEY=你的中转key(可选)

# 3. 启动应用
python unified_app.py        # 打开 http://127.0.0.1:7860

# 4. (可选) 跑 RAG 评估对比
python rag_enterprise.py eval
```

预置体验：

1. 切到 **📚 知识库问答**，身份选"张三"问 `5G专网的内部结算价是多少?` → 回答"无法确定"
2. 身份换成"李财务"再问同一句话 → 给出结算价明细——权限隔离即刻可见
3. 切回 **💬 自由对话** 问 `600519现在多少钱?` → 调用真实行情接口

## 项目结构

```
├── unified_app.py          # 主应用: Gradio 界面 + 双模式(对话/知识库) + 工具调用
├── rag_enterprise.py       # 企业版 RAG: ACL权限 + 混合检索 + RRF融合 + rerank + 评估集
├── rag_demo.py             # RAG 最小教学版(numpy 手写向量库, 讲原理用)
├── 知识库/                  # 4篇运营商政企文档 + kb_manifest.json(权限清单)
├── vector_store_enterprise.json  # 预构建向量库(克隆即用, 无需重新花钱embedding)
├── 教学分步存档/            # 各课独立脚本: 第一个LLM应用/流式/Prompt实验/Function Calling
├── 我的AI-Agent自学路线v2.md
└── 排错方法论复盘.md        # 真实排错记录: 超时/幻觉护栏误伤/检索不准
```

## 技术栈

Python · OpenAI SDK · Gradio · 智谱 GLM-4-Flash / embedding-2 / rerank · BM25(rank_bm25) · NumPy

## 踩坑记录（真实复盘，详见 排错方法论复盘.md）

- **检索反而变差**：裸块送精排丢了文档标题信号，"价格"类问题被资费文档抢走 → 块拼标题前缀
- **评估集设计错误**：把权限拦截误判为召回失败 → 召回与权限断言分开度量
- **BM25 下标错位**：权限过滤后索引空间变化，按原下标取数越界
- **中转站不稳定**：免费/中转通道频繁宕机 → 默认官方通道 + 断线保半截 + 黑匣子日志

## Roadmap

- [x] LLM API + Prompt 工程 + Function Calling
- [x] RAG：权限隔离 / 混合检索 / Rerank / Eval
- [ ] Query 改写（多轮追问的指代补全）
- [ ] LangGraph 多步编排 + 人工干预节点
- [ ] MCP Server（对接数据库/API）
- [ ] Docker + FastAPI 部署
