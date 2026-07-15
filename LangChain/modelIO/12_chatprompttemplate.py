from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import base64

load_dotenv()
prefix = "SILICONFLOW"
llm = init_chat_model(
    model_provider="openai",
    configurable_fields=["model", "api_key", "base_url"],
    config_prefix=prefix,
    temperature=0.5,
    max_tokens=1024,
)
config = {
    "configurable": {
        f"{prefix}_model": os.getenv(f"{prefix}_MODEL"),
        f"{prefix}_api_key": os.getenv(f"{prefix}_API_KEY"),
        f"{prefix}_base_url": os.getenv(f"{prefix}_BASE_URL"),
    }
}
# prompt = ChatPromptTemplate([
#     {"role": "system", "content": "你是AI开发助手{name}，擅长编程开发。"},
#     {"role": "user", "content": "你能帮我做什么？"},
#     {"role": "ai", "content": "我能帮你写代码，改bug，设计技术方案。"},
#     {"role": "user", "content": "{user_question}"}
# ]
# )
# prompt_partial = prompt.partial(name="freeZH")
# prompt_value = prompt_partial.invoke({
#     "user_question": "用python写一个简单的计算器程序"
# })
# print(prompt_value)
# resp = llm.invoke(prompt_value, config=config)
# print(resp.content)

# 通过placeholder输入消息列表
# prompt = ChatPromptTemplate([
#     {"role": "system", "content": "你是AI开发助手{name}，擅长编程开发。"},
#     {"role": "placeholder","content":"{history}"},
#     {"role": "user", "content": "你能帮我做什么？"},
#     {"role": "ai", "content": "我能帮你写代码，改bug，设计技术方案。"},
#     {"role": "user", "content": "{user_question}"}
# ]
# )
# prompt_partial = prompt.partial(name="freeZH")
# prompt_value = prompt_partial.invoke({
#     "user_question": "用python写一个简单的计算器程序",
#     "history":[
#         {"role": "user", "content": "我想做个编程工具"},
#         {"role":"ai","content":"我来帮你实现"}
#     ]
# })
# resp = llm.invoke(prompt_value, config=config)
# print(resp.content)


# 新增：图片转base64工具函数
def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    base64_str = base64.b64encode(bytes_data).decode("utf-8")
    # OpenAI兼容格式必须带前缀
    return f"data:image/jpeg;base64,{base64_str}"

# 多模态消息格式化
prompt_template = ChatPromptTemplate([
    {"role":"system","content":"你是专业的多模态内容分析助手。"},
    {"role":"user","content":[
        {"type":"text","text":"用中文简短描述图片内容"},
        {"type":"image_url","image_url":{"url":"{image_url}"}}
    ]}
])
prompt_value = prompt_template.invoke(
    {
        "image_url": file_to_base64("./image.jpg")
    }
)
resp = llm.invoke(prompt_value, config=config)
print(resp.content)