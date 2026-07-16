from html import parser

import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
prefix = "SILICONFLOW"
model = init_chat_model(
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

# 保存为JSON文件
def save_json(data, filename="output.json"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"JSON数据表已成功保存到{filename}")
    except Exception as e:
        print(f"保存JSON文件时出错：{e}")
    return data

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是一个10年经验的资深软件工程师。{format_instructions}"},
        {"role": "user", "content": "{new_input}"},
    ]
)

chain = ({
             "new_input": RunnablePassthrough(),
             "format_instructions": lambda _: parser.get_format_instructions()
         }
         | prompt
         | model
         | parser
         | save_json
         )
res = chain.invoke({"new_input": "请描述一下张三这个人，包括姓名，年龄，职业和兴趣爱好"}, config=config)
print(res)
