from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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

example_prompt = PromptTemplate.from_template("{input}->{output}")
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请把以下英文翻译成中文：",
    suffix="{new_input}->",
    input_variables=["new_input"],
    example_separator="\n",
)
prompt_value = few_shot_prompt.invoke(
    {
        "new_input": "Good bye"
    }
)
print(prompt_value.to_string())
res = llm.invoke(prompt_value,config=config)
print(res.content)