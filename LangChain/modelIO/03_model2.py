import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
llm = init_chat_model(model="qwen3.6-plus",
                      model_provider="openai",
                      base_url=os.getenv("QWEN_BASE_URL"),
                      api_key=os.getenv("QWEN_API_KEY"),
                      temperature=0.5,
                      max_tokens=1024,
                      streaming=True,
                      )
# llm = init_chat_model(model="qwen3.6-plus", model_provider="openai")
# print(type(llm))
# print(llm.root_client.base_url)
# # print(llm.root_client.api_key)
# result = llm.invoke("你是谁啊？")
# print(result)

# 流式输出
for chunk in llm.stream("介绍一下LangChain"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
