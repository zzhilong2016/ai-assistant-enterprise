# -*- coding: utf-8 -*-
"""
阶段 0 实操：第一个 LLM 应用
步骤：
  1. 把下面三行配置改成你自己的
  2. 运行: python first_llm_app.py
  3. 浏览器会自动打开 http://127.0.0.1:7860
"""
from openai import OpenAI
import gradio as gr
import os

# 密钥脱敏: 从仓库根目录的 .env 读取(该文件已被 .gitignore 拦截, 不入库)
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE, encoding="utf-8"):
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

# ========== 配置区（只改这里）==========
API_KEY = os.environ["CCSWIC_API_KEY"]
BASE_URL = "https://xiaoni-ai.top/v1"
MODEL = "gpt-6-astra"   # 已对照该服务的模型列表修正（原名多了个空格）
# =====================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(message, history):
    """把 Gradio 的对话历史转成 OpenAI 的 messages 格式，再调用模型"""
    messages = [{"role": "system", "content": "你是一个耐心的编程助教，回答简洁清楚。"}]
    for item in history:
        if isinstance(item, dict):
            # Gradio 6 格式: {"role": "user"/"assistant", "content": "..."}
            content = item["content"]
            if not isinstance(content, str):   # 多模态内容取文本部分
                content = "".join(getattr(b, "text", "") for b in content)
            messages.append({"role": item["role"], "content": content})
        else:
            # 兼容旧版元组格式: (用户的话, 机器人的话)
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        title="我的第一个 LLM 应用",
        description="阶段 0 成果：LLM API + Gradio 界面",
    ).launch()
