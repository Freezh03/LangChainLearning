import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
llm = init_chat_model(model="qwen3.7-plus",
                      model_provider="openai",
                      base_url=os.getenv("DASHSCOPE_BASE_URL"),
                      api_key=os.getenv("DASHSCOPE_API_KEY"),
                      configurable_fields=["model","temperature","max_tokens"],
                      temperature=0.5,
                      max_tokens=1024,
                      )
result = llm.invoke("你是谁啊？")
print(result.content)
print("-"*60)

llm_plus = llm.with_config(
    configurable={"model":"deepseek-v4-pro","temperature":0.4,"max_tokens":100}
)
result = llm_plus.invoke("你是谁啊？")
print(result.content)