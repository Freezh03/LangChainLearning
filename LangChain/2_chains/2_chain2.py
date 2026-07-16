from html import parser

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

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
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是一个笑话大王"},
        {"role": "user", "content": "{new_input}"},
    ]
)

chain = prompt | model | parser
res = chain.invoke({"new_input": "讲一个笑话"}, config=config)
print(res)
