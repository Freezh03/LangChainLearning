import os
import dotenv
from langchain_openai import OpenAI

# 设置环境变量
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

llm = OpenAI(
    model="gpt-4o-mini",
    streaming=False,
)

from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "请评价{product}的优缺点，包括{aspect1}和{aspect2}。"
)
prompt = prompt_template.format(product="电脑",aspect1="性能",aspect2="电池")
print(type(prompt))
print(prompt)

llm.invoke(prompt)  #使用非对话模型调用

#print(llm.invoke(prompt))   #阻塞时输出

# 流式输出
for chunk in llm.stream(prompt):
    print(chunk, end='', flush=True)