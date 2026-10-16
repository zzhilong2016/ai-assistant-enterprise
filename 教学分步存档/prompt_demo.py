# -*- coding: utf-8 -*-
"""阶段1实验：模糊提示词 vs 结构化提示词 vs Few-Shot，对比输出质量"""
from openai import OpenAI
import os

# 密钥脱敏: 从仓库根目录的 .env 读取(该文件已被 .gitignore 拦截, 不入库)
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE, encoding="utf-8"):
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

API_KEY = os.environ["CCSWIC_API_KEY"]
BASE_URL = "https://xiaoni-ai.top/v1"
MODEL = "gpt-6-astra"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90)


def ask(messages, label):
    r = client.chat.completions.create(model=MODEL, messages=messages)
    print(f"\n{'='*20} {label} {'='*20}")
    print(r.choices[0].message.content.strip())


# 实验1: 模糊提示词 vs 四要素结构化提示词
ask([{"role": "user", "content": "介绍一下Python"}], "实验1a: 模糊提示词")
ask([
    {"role": "system", "content": "你是一位面向零基础学员的资深编程讲师，擅长用生活类比解释概念。"},
    {"role": "user", "content": "任务：介绍Python这门语言\n"
                                "上下文：学员是完全没有编程经验的大学生，只关心学了能干什么\n"
                                "输出要求：分三段：一句话定义 / 三个实际用途（每点不超过20字）/ 一个生活类比。总长不超过150字。"}
], "实验1b: 四要素结构化提示词")

# 实验2: Few-Shot（给示例）控制输出为固定 JSON 格式
ask([
    {"role": "system", "content": "你是信息抽取助手。只输出JSON，不要任何多余文字。"},
    {"role": "user", "content": "从下面的评论中抽取：情感（正面/负面）、关注点、是否推荐购买。\n"
                                "评论：\n"
                                "例1: '电池很耐用，就是有点重，总体来说推荐！'\n"
                                '-> {"sentiment":"正面","focus":"电池续航","recommend":true}\n'
                                "例2: '屏幕不错但信号太差了，后悔买了'\n"
                                '-> {"sentiment":"负面","focus":"信号质量","recommend":false}\n'
                                "现在抽取：'拍照效果惊艳，夜景模式特别棒，已经安利给同事了'"}
], "实验2: Few-Shot 控制JSON输出")
