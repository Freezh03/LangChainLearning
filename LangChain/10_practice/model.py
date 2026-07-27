import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
prefix = "QWEN"

model = init_chat_model(
    model_provider="openai",
    configurable_fields=["model", "api_key", "base_url"],
    config_prefix=prefix,
).with_config({
    "configurable": {
        f"{prefix}_model": os.getenv(f"{prefix}_MODEL"),
        f"{prefix}_base_url": os.getenv(f"{prefix}_BASE_URL"),
        f"{prefix}_api_key": os.getenv(f"{prefix}_API_KEY"),
    }
})

prompt = "你好"

res = model.invoke(prompt)
print(res.content)


res = model.stream(prompt)
for chunk in res:
    print(chunk.content,end="",flush=True)
