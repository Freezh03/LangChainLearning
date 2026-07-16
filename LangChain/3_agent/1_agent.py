import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()
prefix = "SILICONFLOW"
model = init_chat_model(
    model_provider="openai",
    configurable_fields=["model", "api_key", "base_url"],
    config_prefix=prefix,
    temperature=0.5,
    max_tokens=1024,
).with_config({
    "configurable": {
        f"{prefix}_model": os.getenv(f"{prefix}_MODEL"),
        f"{prefix}_api_key": os.getenv(f"{prefix}_API_KEY"),
        f"{prefix}_base_url": os.getenv(f"{prefix}_BASE_URL"),
    }
})
@tool(description="计算数学表达式的值")
# 定义calculate函数，接收字符串表达式，用于完成数学计算
def calculate(expression:str)->str:
    # 函数搭配异常捕获，用eval执行算式，正常返回计算值，出错就返回报错信息
    try:
        return f"计算结果：{eval(expression)}"
    except Exception as e:
        return f"计算错误：{str(3)}"

agent = create_agent(
    model=model,
    tools=[calculate],
    system_prompt="你是一个专业的计算器助手"
)
response = agent.invoke(
    {"messages":[{"role":"user","content":"计算（123+456)*789/10"}]}
)
print(f"最终计算结果：{response["messages"][-1].content}")