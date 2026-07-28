import asyncio
import os
import time
import dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量
dotenv.load_dotenv()
prefix = "SILICON"
# 初始化大模型
try:
    chat_model = init_chat_model(
        model_provider="openai",
        configurable_fields=["model","api_key","base_url"],
        config_prefix=prefix,
    ).with_config({
        "configurable":{
            f"{prefix}_model":os.getenv(f"{prefix}_MODEL"),
            f"{prefix}_api_key":os.getenv(f"{prefix}_API_KEY"),
            f"{prefix}_base_url":os.getenv(f"{prefix}_BASE_URL"),            
        }
    })
    print(f"LLM初始化成功！")
except Exception as e:
    print(f"LLM初始化失败，失败原因：{e}")
    chat_model=None

# 同步调用（对比组）
async def demo_async_invoke():
    print("+++演示：ainvoke 的异步（非阻塞）效果+++")
    start_time = time.perf_counter()  # 记录开始时间

    print("程序开始...")

    # 1. 创建任务（task）
    print(">>>发起异步模型调用（ainvoke）...")
    async_task = asyncio.create_task(chat_model.ainvoke("用一句话解释人工智能。"))

    # 2. 并行执行其他任务
    print(">>>模型请求已在后台发送，继续执行本地逻辑...")
    for i in range(3):
        await asyncio.sleep(1)  # 异步等待，释放控制权
        print(f">>>正在执行第{i+1}个任务...(已耗时{time.perf_counter() - start_time:.2f}s)")

    # 3. 获取模型结果
    print(">>>本地任务完成，检查模型状态...")
    response = await async_task

    end_time = time.perf_counter()
    print(f">>>模型返回：{response.content}")
    print(f"++++总运行耗时：{end_time- start_time:.2f}s+++++")

# 入口：执行异步主函数
if __name__ == "__main__":
    asyncio.run(demo_async_invoke())