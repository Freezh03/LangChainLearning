from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
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
examples = [
    {"input": "Hello", "output": "你好"},
    {"input": "Thank you", "output": "谢谢"},
    {"input": "Good morning", "output": "早上好"}
]

example_prompt = ChatPromptTemplate.from_messages([
    {"role":"user","content":"{input}"},
    {"role":"ai","content":"{output}"},
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)
chat_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "请把以下英文翻译成中文："},
    few_shot_prompt,
    {"role": "user", "content": "{new_input}"},
])
prompt_value = chat_prompt.invoke(
    {
        "new_input": "What a beautiful day",
    }
)
print(prompt_value.to_string())
res = llm.invoke(prompt_value,config=config)
print(res.content)