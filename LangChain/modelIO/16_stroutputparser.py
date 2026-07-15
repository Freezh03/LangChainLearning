import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages([
    {"role":"system","content":"你是一个笑话大王"},
    {"role":"user","content":"{new_input}"},
])

# 输入内容
prompt_value = chat_prompt.invoke(
    {
        "new_input":"讲一个英文冷笑话"
    }
)
res = llm.invoke(prompt_value, config=config)
print(res.content)
print("+"*60)
parser = StrOutputParser()
result = parser.invoke(res)
print(result)