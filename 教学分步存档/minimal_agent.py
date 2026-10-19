# -*- coding: utf-8 -*-
"""
第3阶段 第1课: 最小 Agent Loop —— 不依赖任何框架
====================================================
Agent 的本质就是一个循环:
    模型决策(回答 or 调工具) → 执行动作 → 观察结果 → 带着新信息再决策
对照: unified_app.py 里的 chat() 就是这个循环的"流式+工程化"版本。
本课用最朴素的写法, 让你看清循环本身。

运行: python 教学分步存档/minimal_agent.py
"""
import os
import json
import requests
from openai import OpenAI

# ---------- 配置区(密钥从 .env 读取, 不入库) ----------
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE, encoding="utf-8"):
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

API_KEY = os.environ["ZHIPU_API_KEY"]
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4-flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)


# ---------- 工具(就是普通 Python 函数) ----------
def get_stock_price(stock_code: str) -> str:
    """查询A股实时行情(真实数据: 新浪财经公开接口)"""
    try:
        code = stock_code.strip().lower()
        full = f"sh{code}" if code.startswith("6") else f"sz{code}"
        r = requests.get(f"https://hq.sinajs.cn/list={full}",
                         headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        r.encoding = "gbk"
        data = r.text.split('"')[1].split(",")
        if not data[0]:
            return f"未找到代码 {stock_code} 的行情"
        return f"{data[0]}({full}) 最新价 {float(data[3]):.2f}元"
    except Exception as e:
        return f"行情查询失败({type(e).__name__})"


def calculator(expression: str) -> str:
    """安全计算器: 只接受数字和 +-*/.() , 其余字符一律拒绝"""
    if not all(c in "0123456789+-*/.() " for c in expression):
        return "表达式含非法字符, 仅支持数字和 +-*/.()"
    try:
        return f"{expression} = {eval(expression)}"
    except Exception:
        return "计算失败: 表达式无法求值"


TOOLS = [
    {"type": "function", "function": {
        "name": "get_stock_price",
        "description": "查询A股实时行情(最新价), 用户问股价/行情/多少钱一股时调用",
        "parameters": {"type": "object", "properties": {
            "stock_code": {"type": "string", "description": "6位A股代码"}}, "required": ["stock_code"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "四则运算计算器, 需要算钱/算数时调用",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "算式, 如 1328*100"}}, "required": ["expression"]}}},
]
TOOL_FUNCTIONS = {"get_stock_price": get_stock_price, "calculator": calculator}

SYSTEM_PROMPT = ("你是会用工具的助手。规则: "
                 "1.需要数据时先调用工具获取真实结果, 严禁编造;"
                 "2.涉及金额计算时, 用工具返回的精确数字计算, 不要自己心算;"
                 "3.每次计算只解决一个明确的小问题(先算总价, 再算差额);"
                 "4.比较预算与总价得出结论时, 若预算大于总价则回答'钱够', 差额是剩余; "
                 "反之回答'不够', 差额是还差多少。")


# ---------- ★核心: Agent 循环(去掉所有花哨装饰的最小版) ----------
def run_agent(question: str, max_steps: int = 5, verbose: bool = True) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n===== 第 {step} 步: 模型决策 =====")
        # ① 模型看完全部历史(含之前的工具结果), 做出决策
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message

        # 决策点A: 不调工具, 直接回答 → 循环结束
        if not msg.tool_calls:
            if verbose:
                print("决策: 直接回答(信息已足够)")
            return msg.content

        # 决策点B: 要调工具 → 记录决策, 执行, 把结果喂回历史
        messages.append(msg)   # assistant 的 tool_calls 必须原样回传(API 要求)
        for call in msg.tool_calls:
            fn = TOOL_FUNCTIONS[call.function.name]
            args = json.loads(call.function.arguments)
            if verbose:
                print(f"决策: 调用工具 {call.function.name}({args})")
            result = fn(**args)                      # ② 真正执行(模型只是"提议", 代码来动手)
            if verbose:
                print(f"观察: 返回 -> {result}")
            messages.append({"role": "tool",          # ③ 观察结果写进历史
                             "tool_call_id": call.id, "content": result})
        # ④ 循环回到①: 模型带着新观察, 做下一轮决策

    return "⚠️ 达到最大步数, 强制停止(防死循环兜底)"


if __name__ == "__main__":
    # 这个问题需要"两个工具串行": 先查价, 再按最新价算账
    q = "600519现在多少钱? 帮我算一下买200股需要多少钱。"
    print("问题:", q)
    answer = run_agent(q)
    print("\n===== 最终回答 =====")
    print(answer)
