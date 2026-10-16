# -*- coding: utf-8 -*-
"""
阶段1收官: Function Calling 实战
给聊天机器人装上"查话费/查套餐"的手
运行: python tools_chat.py
"""
from openai import OpenAI
import gradio as gr
import json
import os

# 密钥脱敏: 从仓库根目录的 .env 读取(该文件已被 .gitignore 拦截, 不入库)
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE, encoding="utf-8"):
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

# ========== 配置区 ========
API_KEY = os.environ["ZHIPU_API_KEY"]
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4-flash"
# ==========================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)


# ---------- 第一步: 定义工具函数(假装是真实业务系统) ----------
import hashlib

def _is_valid_phone(phone: str) -> bool:
    return phone.isdigit() and len(phone) == 11 and phone.startswith("1")

def _pseudo_random(phone: str, salt: str) -> int:
    """用号码本身算出确定性伪随机数: 同一号码永远得到同一结果, 模拟真实数据库"""
    return int(hashlib.md5((phone + salt).encode()).hexdigest(), 16)

_PLANS = [
    "5G畅享套餐 99元/月: 30GB流量+500分钟通话",
    "4G飞享套餐 38元/月: 5GB流量+100分钟通话",
    "5G尊享套餐 159元/月: 100GB流量+1500分钟通话",
    "青春卡套餐 29元/月: 20GB流量+200分钟通话",
]

def get_phone_balance(phone_number: str) -> str:
    """查询手机话费余额"""
    if not _is_valid_phone(phone_number):
        return f"号码 {phone_number} 格式不合法, 应为11位数字且以1开头"
    n = _pseudo_random(phone_number, "balance")
    balance = f"{n % 50000 / 100:.2f}"   # 0.00 ~ 500.00 元
    return f"话费余额: {balance} 元"


def get_phone_plan(phone_number: str) -> str:
    """查询手机套餐信息"""
    if not _is_valid_phone(phone_number):
        return f"号码 {phone_number} 格式不合法, 应为11位数字且以1开头"
    n = _pseudo_random(phone_number, "plan")
    return _PLANS[n % len(_PLANS)]


# ---------- 真实工具: 调新浪财经公开行情接口(免Key, 国内直连) ----------
import requests

def get_stock_price(stock_code: str) -> str:
    """查询A股股票实时行情(真实数据源: 新浪财经)"""
    try:
        code = stock_code.strip().lower()
        # 智能判断交易所: 6开头=沪(sh), 0/3开头=深(sz)
        if code.startswith("6"):
            full = f"sh{code}"
        elif code.startswith(("0", "3")):
            full = f"sz{code}"
        else:
            return f"代码 {stock_code} 无法识别, A股代码应为6位数字(如 601006)"

        r = requests.get(
            f"https://hq.sinajs.cn/list={full}",
            headers={"Referer": "https://finance.sina.com.cn"},  # 该接口要求来源页
            timeout=10,
        )
        r.encoding = "gbk"   # 新浪接口用GBK编码
        data = r.text.split('"')[1].split(",")
        if not data[0]:
            return f"未找到代码 {stock_code} 的行情, 请确认代码正确"

        name, open_p, prev_close, price = data[0], float(data[1]), float(data[2]), float(data[3])
        change = price - prev_close
        change_pct = change / prev_close * 100 if prev_close else 0
        arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
        return (f"{name}({full}): 最新价 {price:.2f}元 {arrow} "
                f"{change:+.2f} ({change_pct:+.2f}%)  今开 {open_p:.2f}")
    except Exception as e:
        return f"行情查询失败({type(e).__name__}), 请稍后重试"


# ---------- 第二步: 用 JSON Schema 向模型"说明书式"地描述工具 ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_phone_balance",
            "description": "查询用户手机的话费余额, 当用户问余额/欠费/还剩多少钱时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "11位手机号, 例如 13800138000",
                    }
                },
                "required": ["phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_phone_plan",
            "description": "查询用户手机当前的套餐内容和资费, 当用户问套餐/流量/资费时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "11位手机号, 例如 13800138000",
                    }
                },
                "required": ["phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询A股股票的实时行情(最新价/涨跌幅), 当用户问股价/行情/涨跌/某支股票时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "6位A股代码, 例如 601006(大秦铁路)、600519(贵州茅台)、000001(平安银行)",
                    }
                },
                "required": ["stock_code"],
            },
        },
    },
]

# 函数字典: 模型说名字, 我们找函数
TOOL_FUNCTIONS = {
    "get_phone_balance": get_phone_balance,
    "get_phone_plan": get_phone_plan,
    "get_stock_price": get_stock_price,
}


# ---------- 第三步: Agent 循环(本课核心) ----------
def run_agent(message: str, history: list) -> str:
    """一次完整的'工具调用循环'"""
    messages = [{"role": "system", "content":
        "你是运营商客服助手。规则："
        "1. 凡是涉及用户号码的信息(余额/套餐/资费), 你必须先调用工具获取真实数据, 再回答；"
        "2. 严禁在不调用工具的情况下编造查询结果；"
        "3. 如果工具返回'未找到', 如实转告用户并建议核对号码。"}]
    for item in history:
        if isinstance(item, dict):
            messages.append(item)
    messages.append({"role": "user", "content": message})

    # 循环: 模型可能连续调用多个工具, 直到给出最终答复
    called_tool = False   # 本回合是否真正调过工具(防'假装查过'的校验依据)
    for _ in range(5):  # 最多5轮, 防死循环(工程兜底)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:  # 模型要求调工具
            called_tool = True
            messages.append(msg)  # 把模型的"调用意图"存入历史
            for call in msg.tool_calls:
                fn_name = call.function.name
                fn_args = json.loads(call.function.arguments)  # 参数是JSON字符串
                print(f"[工具调用] {fn_name}({fn_args})")        # 控制台留痕

                result = TOOL_FUNCTIONS[fn_name](**fn_args)      # 真正执行

                messages.append({                                # 结果喂回模型
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
        else:  # 模型给出最终答复
            # ===== 护栏: 声称查过但没调工具 = 幻觉, 打回重来 =====
            if not called_tool and any(k in msg.content for k in ("查询", "未找到", "余额为", "套餐为")):
                messages.append({"role": "system", "content":
                    "警告：你刚才没有调用任何工具就给出了查询结论，这是编造的。"
                    "请立即调用正确的工具获取真实数据后再回答。"})
                continue
            # =======================================================
            return msg.content

    return "⚠️ 工具调用次数过多, 已中断"


# ---------- 第四步: 命令行测试 ----------
def cli_mode():
    print("=== 运营商客服 Agent (命令行版) ===")
    print("(试试: 我话费还剩多少? 13800138000)")
    while True:
        q = input("\n你: ").strip()
        if q in ("退出", "quit"):
            break
        a = run_agent(q, [])
        print(f"客服: {a}")


# ---------- 第五步: Gradio 界面 ----------
def gradio_chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    answer = run_agent(message, history)
    return answer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_mode()
    else:
        gr.ChatInterface(
            fn=gradio_chat,
            title="运营商客服 Agent · Function Calling",
            description="阶段1收官: 模型自动调用查话费/查套餐工具",
        ).launch()
