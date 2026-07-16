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
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


chat_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "你是一个10年经验的资深软件工程师。{format_instructions}"},
    {"role": "user", "content": "{new_input}"},
])


class Persion(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    post: str = Field(description="岗位")
    hobbies: list[str] = Field(description="兴趣爱好")


parser = PydanticOutputParser(pydantic_object=Persion)
# 输入内容
prompt_value = chat_prompt.invoke(
    {
        "format_instructions": parser.get_format_instructions(),
        "new_input": "请描述一下张三这个人，包括姓名，年龄，职业和兴趣爱好。"
    }
)
print(prompt_value.to_string())
res = llm.invoke(prompt_value, config=config)
persion = parser.invoke(res)
print(f"姓名：{persion.name}")
print(f"姓名：{persion.age}")
print(f"姓名：{persion.post}")
print(f"姓名：{persion.hobbies}")
