from langchain_core.prompts import PromptTemplate
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
prompt = PromptTemplate.from_template(
    "将{word}翻译成{lang}"
)
prompt_value = prompt.invoke({"word":"good","lang":"中文"})
# res = llm.invoke(prompt_value,config=config)
# print(res.content)
prompt_str = prompt_value.to_string()
print(prompt_value)
print(prompt_str)