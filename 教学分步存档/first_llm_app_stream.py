# -*- coding: utf-8 -*-
"""
阶段1练习：流式输出版聊天机器人
与 first_llm_app.py 的区别：回答逐字出现，不用干等
运行: python first_llm_app_stream.py
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

# ========== 配置区：多通道切换 ========
# 格式: "显示名": {"api_key": ..., "base_url": ..., "model": ...}
PROVIDERS = {
    "CCSWIC (gpt-6-astra)": {
        "api_key": os.environ["CCSWIC_API_KEY"],
        "base_url": "https://xiaoni-ai.top/v1",
        "model": "gpt-6-astra",
    },
    "智谱 (glm-4-flash)": {
        "api_key": os.environ["ZHIPU_API_KEY"],
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
}
DEFAULT_PROVIDER = "智谱 (glm-4-flash)"   # 默认用稳定的智谱
# =====================================


def _extract_text(content):
    """从 Gradio 的 content 里提取纯文本: 兼容 str / 对象块 / 字典块 三种格式"""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):            # {"type": "text", "text": "..."}
            parts.append(block.get("text", ""))
        else:                                  # 带 .text 属性的对象
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def chat(message, history, provider_name):
    """流式输出：模型每生成一段就吐一段，边生成边显示"""
    cfg = PROVIDERS[provider_name]                    # 按界面选中的通道取配置
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120)

    messages = [{"role": "system", "content": "你是一个耐心的编程助教，回答简洁清楚。"}]
    for item in history:
        if isinstance(item, dict):
            # Gradio 6 格式: {"role": "user"/"assistant", "content": "..."}
            content = _extract_text(item["content"])  # content 可能是 str 或内容块列表
            messages.append({"role": item["role"], "content": content})
        else:
            # 兼容旧版元组格式: (用户的话, 机器人的话)
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # ===== 黑匣子: 把每次实际发出去的请求落盘, 灵异问题发生后可查证 =====
    import json, time, os
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_debug.log"),
              "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%H:%M:%S')} 通道={provider_name} =====\n")
        f.write(json.dumps(messages, ensure_ascii=False, indent=1) + "\n")
    # ==================================================================

    # stream=True: 返回值变成迭代器，逐块吐出
    reply = ""
    try:
        stream = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            stream=True,          # ← 关键改动就这一个参数
        )
        for chunk in stream:
            if not chunk.choices:                # 有的块不含内容（仅.usage统计），跳过
                continue
            delta = chunk.choices[0].delta.content   # 这一小块新增的文本
            if delta:                                # 有的块是空的（控制信息），跳过
                reply += delta
                yield reply          # 每收到一块就更新界面，实现"打字机"效果
    except Exception as e:
        # 容错：中途断线不丢已有内容，并给出可读的原因
        if reply:
            yield reply + f"\n\n---\n⚠️ 连接中断（{type(e).__name__}），以上为已生成的部分。可重发或切换通道。"
        else:
            yield f"⚠️ 调用失败：{type(e).__name__}\n可能原因：该通道宕机或网络不通，请切换模型通道后重试。"


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        title="我的第一个 LLM 应用 · 流式多通道版",
        description="阶段 1 成果：stream=True 打字机效果 + 多模型通道切换",
        additional_inputs=[   # 在输入框下方加一个下拉框，切换模型通道
            gr.Dropdown(
                choices=list(PROVIDERS.keys()),
                value=DEFAULT_PROVIDER,
                label="模型通道",
            )
        ],
    ).launch()
