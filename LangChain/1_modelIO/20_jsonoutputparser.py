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
from langchain_core.output_parsers import  JsonOutputParser

chat_prompt = ChatPromptTemplate.from_messages([
    {"role":"system","content":"你是一个10年经验的资深软件工程师。{format_instructions}"},
    {"role":"user","content":"{new_input}"},
])
parser = JsonOutputParser()
# 输入内容
prompt_value = chat_prompt.invoke(
    {
        "format_instructions":parser.get_format_instructions(),
        "new_input":"请描述一下张三这个人，包括姓名，性别，年龄，兴趣爱好。"
    }
)
print(prompt_value.to_string())
res = llm.invoke(prompt_value, config=config)
result = parser.invoke(res)
print(result)
# 一行一行打印
# for step_num, step_content in enumerate(result,1):
#     print(f"{step_num}.{step_content}")
