# -*- coding: utf-8 -*-
"""探针: 抓取 Gradio 6 真实传给 fn 的 history 结构"""
import json
import gradio as gr

DUMP_FILE = r"D:\测试项目1\待办事项应用\history_dump.json"


def probe(message, history):
    # 把 history 原样落盘(对象不可序列化时转字符串)
    with open(DUMP_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(history, default=str, ensure_ascii=False) + "\n")
    return f"收到: {message}"


if __name__ == "__main__":
    gr.ChatInterface(fn=probe, title="history探针").launch(server_port=7863)
