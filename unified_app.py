# -*- coding: utf-8 -*-
"""
全能助手 · 最终交付版
统合阶段1全部成果:
  多通道切换 + 流式输出 + 超时容错 + 黑匣子日志 + 幻觉护栏 + Function Calling
工具: 查股票行情(真实·新浪财经公开接口)
+RAG: 知识库问答模式(复用 rag_enterprise.py: 权限隔离/混合检索/精排)
运行: python unified_app.py
"""
from openai import OpenAI
import gradio as gr
import json
import time
import os
import requests

# ---------- 企业知识库 RAG(复用阶段2的企业版) ----------
from rag_enterprise import EnterpriseVectorStore, STORE_FILE, load_docs_with_acl, split_chunks

# ========== 配置区: 多通道切换 ========
# 密钥说明: 两个 key 都从环境变量/.env读取(rag_enterprise导入时已自动加载.env)
PROVIDERS = {
    "CCSWIC (gpt-6-astra)": {
        "api_key": os.environ.get("CCSWIC_API_KEY", ""),
        "base_url": "https://xiaoni-ai.top/v1",
        "model": "gpt-6-astra",
    },
    "智谱 (glm-4-flash)": {
        "api_key": os.environ.get("ZHIPU_API_KEY", ""),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
}
DEFAULT_PROVIDER = "智谱 (glm-4-flash)"   # 默认走稳定的官方通道
# =====================================

SYSTEM_PROMPT = (
    "你是用户的全能助手。规则："
    "1. 涉及股价/行情/涨跌, 调用股票工具查询后再回答, 数据以工具返回为准;"
    "2. 严禁在未调用工具的情况下编造查询结果;"
    "3. 工具返回'未找到'或错误时, 如实转告用户;"
    "4. 用户提出你没有工具支撑的需求(如查话费/订机票)时, 明确告知'暂不支持该功能',"
    "不得编造回答, 不得答非所问;"
    "5. 其他问题(如编程、概念解释)直接回答, 不需要调工具。"
)


# ---------- 工具函数 ----------
def get_stock_price(stock_code: str) -> str:
    """查询A股实时行情(真实数据: 新浪财经公开接口)"""
    try:
        code = stock_code.strip().lower()
        if code.startswith("6"):
            full = f"sh{code}"
        elif code.startswith(("0", "3")):
            full = f"sz{code}"
        else:
            return f"代码 {stock_code} 无法识别, A股代码应为6位数字(如 601006)"
        r = requests.get(
            f"https://hq.sinajs.cn/list={full}",
            headers={"Referer": "https://finance.sina.com.cn"},
            timeout=10,
        )
        r.encoding = "gbk"
        data = r.text.split('"')[1].split(",")
        if not data[0]:
            return f"未找到代码 {stock_code} 的行情, 请确认代码正确"
        name, open_p, prev_close, price = data[0], float(data[1]), float(data[2]), float(data[3])
        change = price - prev_close
        pct = change / prev_close * 100 if prev_close else 0
        arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
        return f"{name}({full}): 最新价 {price:.2f}元 {arrow} {change:+.2f} ({pct:+.2f}%)  今开 {open_p:.2f}"
    except Exception as e:
        return f"行情查询失败({type(e).__name__}), 请稍后重试"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询A股股票的实时行情(最新价/涨跌幅), 当用户问股价/行情/涨跌/某支股票时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string", "description": "6位A股代码, 例如 601006、600519、000001"}
                },
                "required": ["stock_code"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
}

# ---------- 知识库问答: 用户身份(决定ACL过滤结果) ----------
KB_USERS = {
    "张三 (政企事业部)": ["all", "政企事业部"],
    "李财务 (财务部)":   ["all", "财务部"],
}

_KB_STORE = None   # 惰性加载: 首次进入知识库模式时才构建/载入向量库

def get_kb_store() -> EnterpriseVectorStore:
    global _KB_STORE
    if _KB_STORE is None:
        _KB_STORE = EnterpriseVectorStore(STORE_FILE)
        if not _KB_STORE.items:   # 首次运行: 建库
            chunks = []
            for d in load_docs_with_acl():
                chunks.extend(split_chunks(d))
            _KB_STORE.add(chunks)
    return _KB_STORE

KB_SYSTEM_PROMPT = (
    "你是企业产品咨询助手。规则："
    "1.答案必须出自给你的参考资料, 资料没有就明确说'根据现有资料无法确定';"
    "2.末尾标注引用的资料编号(如【资料1】);"
    "3.简明扼要, 不编造资料中不存在的信息。"
)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_debug.log")


# ---------- 基础工具 ----------
def _extract_text(content):
    """从 Gradio 的 content 提取纯文本, 兼容 str/对象块/字典块"""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        parts.append(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
    return "".join(parts)


def _blackbox(messages, provider_name):
    """黑匣子: 每次请求落盘, 灵异问题发生后可查证"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%H:%M:%S')} 通道={provider_name} =====\n")
        f.write(json.dumps(messages, ensure_ascii=False, indent=1) + "\n")


# ---------- 核心: 流式 Agent 循环 ----------
def chat(message, history, provider_name):
    """完全体对话函数: 流式 + 工具调用 + 幻觉护栏 + 容错"""
    cfg = PROVIDERS[provider_name]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": _extract_text(item["content"])})
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    called_tool = False      # 本回合是否调过工具(幻觉护栏依据)
    guardrail_fired = False  # 护栏只打回一次, 防与拒绝话术互相矛盾造成死循环
    used_tools = []          # 给用户展示调用了哪些工具
    debug_calls = []         # 侧边栏调试面板: 工具调用记录
    reply = ""

    try:
        for _ in range(5):  # 最多5轮工具调用, 防死循环(工程兜底)
            _blackbox(messages, provider_name)
            stream = client.chat.completions.create(
                model=cfg["model"], messages=messages,
                tools=TOOLS, stream=True,          # 流式 + 工具能力同时开
            )

            # 流式累积: 文本逐块yield, 工具调用增量拼完整
            content = ""
            tool_acc = {}    # index -> 累积中的工具调用
            for chunk in stream:
                if not chunk.choices:             # 跳过纯统计块(usage)
                    continue
                delta = chunk.choices[0].delta
                if delta.content:                 # 情况A: 正常文本 → 边收边显示
                    content += delta.content
                    yield content, {"通道": provider_name, "工具调用": debug_calls}
                if delta.tool_calls:              # 情况B: 工具调用意图(分块来的) → 静默累积
                    for tc in delta.tool_calls:
                        acc = tool_acc.setdefault(tc.index, {"id": "", "function": {"name": "", "arguments": ""}})
                        if tc.id:
                            acc["id"] += tc.id
                        if tc.function and tc.function.name:
                            acc["function"]["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            acc["function"]["arguments"] += tc.function.arguments
            reply = content

            if not tool_acc:                      # 分支B: 最终答复
                # 幻觉护栏: 声称有具体查询结果但没调工具 → 打回
                # 注意: ①"查询"一词太宽泛, 拒绝话术里也含它, 不算幻觉;
                #       ②只打回一次, 避免护栏与拒绝规则互相拉扯造成死循环
                CLAIM_WORDS = ("最新价", "余额为", "套餐为", "未找到", "查询到")
                REFUSAL_WORDS = ("无法", "不支持", "不能", "抱歉")
                claims_result = any(k in content for k in CLAIM_WORDS) and \
                                not any(r in content for r in REFUSAL_WORDS)
                if not called_tool and claims_result and not guardrail_fired:
                    guardrail_fired = True
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "system", "content":
                        "警告: 你刚才没有调用任何工具就给出了查询结论, 这是编造的。"
                        "请立即调用正确的工具获取真实数据后再回答。"})
                    yield content + "\n\n*(正在核实数据...)*", {"通道": provider_name, "工具调用": debug_calls}
                    continue
                yield content, {"通道": provider_name, "工具调用": debug_calls}
                return

            # 分支A: 执行工具, 结果回喂
            called_tool = True
            messages.append({"role": "assistant", "content": content, "tool_calls": [
                {"id": tool_acc[i]["id"], "type": "function",
                 "function": {"name": tool_acc[i]["function"]["name"],
                              "arguments": tool_acc[i]["function"]["arguments"]}}
                for i in sorted(tool_acc.keys())
            ]})
            for i in sorted(tool_acc.keys()):
                call = tool_acc[i]
                fn_name = call["function"]["name"]
                fn_args = json.loads(call["function"]["arguments"] or "{}")
                print(f"[工具调用] {fn_name}({fn_args})")   # 控制台留痕
                result = TOOL_FUNCTIONS[fn_name](**fn_args)  # 真正执行
                used_tools.append(fn_name)
                debug_calls.append({"函数": fn_name, "参数": fn_args, "返回": result})   # 调试面板
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            # 工具调用过程对用户可见化
            yield reply + f"\n\n*(已调用: {'、'.join(used_tools)}, 查询中...)*", \
                  {"通道": provider_name, "工具调用": debug_calls}

        yield reply + "\n\n⚠️ 工具调用轮数过多, 已中断", {"通道": provider_name, "工具调用": debug_calls}

    except Exception as e:
        # 容错: 中途断线保住已有内容, 彻底失败给人话提示
        if reply:
            yield reply + f"\n\n---\n⚠️ 连接中断({type(e).__name__}), 以上为已生成的部分。可重发或切换通道。", \
                  {"通道": provider_name, "工具调用": debug_calls}
        else:
            yield f"⚠️ 调用失败: {type(e).__name__}\n可能原因: 该通道宕机或网络不通, 请切换模型通道后重试。", \
                  {"通道": provider_name, "工具调用": debug_calls}


# ---------- 知识库问答: 检索(权限过滤+混合+精排) → 组装证据 → 流式生成 ----------
REWRITE_PROMPT = """将用户的问题改写为独立的搜索查询, 用于在企业知识库中检索资料。
规则:
1. 如果问题本身自包含, 不需要上下文就能理解, 原样返回, 一个字都不改;
2. 如果问题含指代词(那/这个/它/上面/还有呢/专业版等), 结合对话历史补全成完整问句;
3. 只输出改写后的查询文本, 不要任何解释、引号或前缀。

对话历史:
{history}

用户问题: {question}

改写后的搜索查询:"""


def rewrite_query(client, cfg, message, history):
    """Query改写: 把带指代的追问补全成独立查询。失败/无历史时原样返回"""
    recent = [h for h in (history or []) if isinstance(h, dict)][-6:]   # 取最近3轮
    if not recent:
        return message
    hist_text = "\n".join(f"{'用户' if h['role'] == 'user' else '助手'}: {_extract_text(h['content'])[:100]}"
                          for h in recent)
    try:
        resp = client.chat.completions.create(
            model=cfg["model"], timeout=30,
            messages=[{"role": "user", "content":
                       REWRITE_PROMPT.format(history=hist_text, question=message)}],
        )
        rewritten = resp.choices[0].message.content.strip().strip('"“”')
        # 护栏: 改写结果不能是空话/过长, 否则视为失败退回原问题
        return rewritten if 0 < len(rewritten) <= 100 else message
    except Exception:
        return message   # 改写失败不阻塞主流程, 用原问题检索


def kb_chat(message, history, provider_name, user_label):
    """RAG模式: 与chat()相同的双输出生成器约定 (yield 文本, 调试dict)"""
    cfg = PROVIDERS[provider_name]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120)
    roles = KB_USERS[user_label]
    debug = {"模式": "知识库问答", "当前用户": user_label, "角色": roles, "检索命中": []}

    yield "*(正在检索知识库...)*", debug
    # Query改写: 追问("那云视讯呢")先补全成独立查询("云视讯怎么收费")再检索
    query = rewrite_query(client, cfg, message, history)
    if query != message:
        debug["查询改写"] = {"原始问题": message, "改写后": query}
    try:
        store = get_kb_store()
        hits = store.search(query, roles, top_k=3)
    except Exception as e:
        yield f"⚠️ 检索失败: {type(e).__name__}\n(embedding/精排固定走智谱通道)", debug
        return

    # 调试面板: 让用户看到"检索到底捞回了什么"(RAG可观测的雏形)
    debug["检索命中"] = [
        {"来源": h["source"], "权限": "、".join(h["acl"]),
         "内容": h["text"][:60] + ("..." if len(h["text"]) > 60 else "")}
        for h in hits
    ]
    yield "*(检索完成, 正在生成回答...)*", debug

    context = "\n\n".join(
        f"【资料{i+1}】(出自{h['source']})\n{h['text']}" for i, h in enumerate(hits))
    messages = [{"role": "system", "content": KB_SYSTEM_PROMPT}]
    for item in history or []:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": _extract_text(item["content"])})
    messages.append({"role": "user", "content":
        f"参考资料:\n{context}\n\n用户问题: {message}"})

    reply = ""
    try:
        _blackbox(messages, provider_name)
        stream = client.chat.completions.create(model=cfg["model"], messages=messages, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                reply += delta.content
                yield reply, debug
    except Exception as e:
        if reply:   # 断线保半截
            yield reply + f"\n\n---\n⚠️ 连接中断({type(e).__name__}), 以上为已生成部分。", debug
        else:
            yield f"⚠️ 生成失败: {type(e).__name__}\n请切换模型通道后重试。", debug


# ---------- Blocks 界面: 主对话区 + 侧边栏调试面板 ----------
CSS = """
footer {display: none !important;}
.gradio-container {max-width: 1100px !important; margin: auto;}
"""

def respond(message, history, provider_name, mode, user_label):
    """包装生成器: 按模式分流, 对话内容与调试面板双输出"""
    history = history or []
    gen = kb_chat(message, history, provider_name, user_label) \
          if mode.startswith("📚") else chat(message, history, provider_name)
    for partial, debug_info in gen:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": partial},
        ], debug_info


if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="全能助手") as demo:
        gr.Markdown("# 📈 全能助手 · 最终版\n"
                    "多通道切换 + 流式输出 + 工具调用 + 容错 + 幻觉护栏 + **知识库RAG**")

        with gr.Row():
            with gr.Column(scale=3):      # 左侧: 对话区
                mode = gr.Dropdown(choices=["💬 自由对话(股票工具)", "📚 知识库问答(RAG)"],
                                   value="💬 自由对话(股票工具)", label="模式")
                chatbot = gr.Chatbot(avatar_images=("👤", "🤖"), height=520, show_label=False)
                msg = gr.Textbox(placeholder="输入消息, 例如: 600519现在多少钱 / 5G专网怎么收费",
                                 show_label=False, lines=2)
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot], value="🗑 清空对话")
                    send = gr.Button("🚀 发送", variant="primary")
                gr.Examples(
                    examples=["600519现在多少钱", "000001行情如何",
                              "5G专网有哪几种模式, 分别什么价格?",
                              "5G专网的内部结算价是多少?"],
                    inputs=msg, label="试试这些(后两条是知识库题)",
                )

            with gr.Column(scale=1):      # 右侧: 侧边栏
                provider = gr.Dropdown(choices=list(PROVIDERS.keys()),
                                       value=DEFAULT_PROVIDER, label="模型通道")
                kb_user = gr.Dropdown(choices=list(KB_USERS.keys()),
                                      value=list(KB_USERS.keys())[0],
                                      label="知识库身份(决定可见范围)")
                debug_panel = gr.JSON(label="⚙️ 调用记录(工具/检索)")
                gr.Markdown("---\n### 可用工具\n- **股票实时行情**（真实·新浪财经）\n\n"
                            "### 知识库\n- 4篇产品文档, 15个块\n- **权限隔离**: 结算价仅财务/总监可见\n"
                            "- 混合检索 + 精排(embedding/精排固定走智谱)\n\n"
                            "### 排障提示\n日志见 `llm_debug.log`（黑匣子）")

        # 三种触发方式: 回车 / 点按钮
        msg.submit(respond, [msg, chatbot, provider, mode, kb_user], [chatbot, debug_panel]) \
            .then(lambda: "", None, [msg])          # 发送后清空输入框
        send.click(respond, [msg, chatbot, provider, mode, kb_user], [chatbot, debug_panel]) \
            .then(lambda: "", None, [msg])

    demo.launch()
