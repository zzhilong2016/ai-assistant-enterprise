# AI Agent 开发 · 个人自学路线（优化版 v2）

> 合并依据：原路线图（HTML，8 阶段，配 B 站 94 集）+ 新执行路线（按 4–6 小时/天设计，运营商背景作品集）
> 整理日期：2026-10-12 · 当前进度：阶段 0 已完成 ✅，阶段 1 进行中

---

## 与原版相比的四个关键调整

1. **节奏**：每天 4–6 小时 → 全周期压缩为 **8–10 周**（原版 10–12 周是按每天 2–3 小时）
2. **阶段 1 扩容**：新路线把 Function Calling 提前到阶段 1（原版放在 LangChain 阶段），这是对的——它是 Agent 的地基，越早碰越好
3. **RAG 权重最高**：两个版本一致认定 RAG 是面试追问最密集区，安排 3 周并配"过关三问"
4. **作品集换题**：用你的运营商/政企背景做选题（区别于普通转行者），第 6 周起并行推进

---

## 阶段 0：认知校准（已完成 ✅）

- LLM 本质、Token、上下文窗口、Temperature、RAG 补哪类知识缺口
- ✅ 已交付：Python 环境（Anaconda 3.12）、API 调通（CCSWIC）、Gradio 聊天应用

> ⚠️ 遗留问题：CCSWIC 不稳定（一天挂两次），**尽快注册 DeepSeek 备用**

## 阶段 1：LLM API + Prompt 工程（第 1–2 周）【当前位置】

**知识点**：对话/流式输出/结构化输出（JSON Schema）、四要素框架、Few-shot、CoT、Function Calling（2–3 个自定义工具）

**对应视频**：04–05、47–53 集

**过关标准**：
- [ ] 模型稳定输出合法 JSON（程序可 `json.loads`）
- [ ] 能讲清一个 Prompt 失效的具体原因
- [ ] 手写 Function Calling：让模型自己决定调用"查话费"和"查套餐"两个工具

**本周练习**：
- [x] 四要素模板套用（已完成）
- [ ] `prompt_demo.py` 对比实验（等接口恢复跑通）
- [ ] 流式输出改造 `first_llm_app.py`（`stream=True`）

## 阶段 2：大模型原理 · Transformer（穿插，不单独占周）

新路线砍掉了这块的整块时间，改为**穿插学习**：每天 30 分钟看 The Illustrated Transformer + HuggingFace 免费课，重点只盯四个概念：**Tokenization、Embedding、Self-Attention、上下文窗口**。

理由：应用开发不需推导数学，但阶段 2 的 embedding 和 RAG 调参会直接用到。

## 阶段 3：RAG（第 3–5 周，全路线最重）

**主线**（LangChain v1 + Chroma 起步，了解 Milvus）：文档解析 → 切块 → embedding → 入库 → 检索 → 生成

**逐级加料**：Top-K 调参 → 混合检索（向量+BM25）→ Rerank → Query 改写 → RAGAS 评估

**过关三问（面试最高频）**：
1. 切片大小怎么选？（答得出"按语义边界+300–500 token 起步做消融"）
2. 为什么纯向量检索会漏关键词匹配？（答得出"向量捕捉语义不保证精确词面，需 BM25 互补"）
3. 检索结果不相关怎么办？（答得出排查路径：query 改写 → 调 K → rerank → 换 embedding 模型）

**实操**：用真实文档建知识库（建议直接用你熟悉的政企方案模板/产品手册做语料，顺便为作品集储备）

## 阶段 4：Agent 核心（第 6–8 周）

1. **手写最小 Agent loop**（不依赖框架）：模型选动作 → 执行 → 观察 → 再决策
2. 两种范式：ReAct vs Plan-and-Execute，能讲清选型依据
3. **LangGraph**：有状态、可中断恢复、带人工干预节点的多步 Agent
4. **MCP**：写 2 个 MCP server（查数据库 / 调 API），理解"Agent 的 USB 接口"

**过关标准**：能讲清 **"什么时候根本不需要 Agent"**——这比会写十个 Demo 更值钱。

> 对应视频：32–46 集（LangChain 1.0 + LangGraph）＋ 06–12 集（方法论）

## 阶段 5：工程化（第 9–10 周，全程穿插）

- **Eval**：50–100 条评测集（golden set），Ragas / LLM-as-judge
- **可观测**：接 Langfuse（追踪 prompt、工具参数、token 消耗）
- **兜底**：超时、重试、最大步数、死循环检测
- **部署**：Docker + FastAPI + Streamlit 前端

**过关两问**：Agent 死循环怎么办？怎么证明一次改版是变好还是变差？

## 阶段 6：作品集（第 6 周起并行，全程）

用运营商场景，这是你区别于其他转行者的**唯一资产**：

**项目一 · 政企知识库 Agent（完整工程向）**
RAG + 权限隔离（不同客户看不同文档）+ LangGraph 编排 + Eval 报告
→ 要求：能讲 30 分钟，每个选型有理由、有踩坑记录

**项目二 · 工单处理 Agent 组（多 Agent 向）**
分类 Agent → 查询 Agent → 起草回复 Agent → 审核 Agent
→ 要求：体现状态管理与异常处理

**验收**：两个项目都部署上线、README、可体验链接。GitHub 上"能跑+有评估数据+有踩坑记录"的项目 > 简历十行技能关键词。

---

## 本周行动清单（阶段 1）

1. [ ] 注册 DeepSeek 备用 Key（platform.deepseek.com）
2. [ ] 跑通 `prompt_demo.py` 三组对比实验
3. [ ] 给 `first_llm_app.py` 加流式输出（`stream=True`，逐字打印效果）
4. [ ] 手写第一个 Function Calling：定义"查话费/查套餐"两个工具，让模型自主调用
5. [ ] 每天 30 分钟：The Illustrated Transformer（jalammar.github.io）
