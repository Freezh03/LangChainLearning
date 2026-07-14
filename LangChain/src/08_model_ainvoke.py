import asyncio
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
prefix = "SILICONFLOW"
llm = init_chat_model(
    model_provider="openai",
    configurable_fields=["model","api_key","base_url"],
    config_prefix=prefix,
    temperature=0.5,
    max_tokens=1024,
)
config = {
    "configurable":{
        f"{prefix}_model":os.getenv(f"{prefix}_MODEL"),
        f"{prefix}_api_key":os.getenv(f"{prefix}_API_KEY"),
        f"{prefix}_base_url":os.getenv(f"{prefix}_BASE_URL"),
    }
}
prompt = [
    "生成Python打印Hello World的代码，只提供一种最标准的写法"
]
async def async_invoke():
    response = await llm.ainvoke(prompt, config=config)
    print(response.content)

asyncio.run(async_invoke())