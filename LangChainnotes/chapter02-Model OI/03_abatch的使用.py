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

questions = [
    "用一句话介绍机器学",
    "中国的首都在哪里",
    "上海的简称是什么"
]

# 同步调用（对比组）
async def demo_async_batch():
    print("+++演示：abatch 的异步（非阻塞）效果+++")
    start_time = time.perf_counter()  # 记录开始时间
    print("程序开始...")

    # 1. 发起异步批量请求
    # 关键修改：使用 create_task 让协程立即在后台执行
    print(">>>发起异步批量调用（abatch）...")
    batch_task = asyncio.create_task(chat_model.abatch(questions))

    # 2. 在等待批量处理的同时，执行其他任务
    print(">>>批量任务已在后台运行，主程序无需等待，继续执行其他任务...")
    for i in range(3):
        # 使用 asyncio.sleep允许后台任务获取 CPU 时间片进行网络请求
        await asyncio.sleep(1)  # 异步等待，释放控制权
        print(f">>>正在执行第{i+1}个任务...(已耗时{time.perf_counter() - start_time:.2f}s)")

    # 3. 等待批量处理结果
    print(">>>其他任务已完成，现在获取后台批量任务的结果...")
    # 此时 batch_task 可能已经完成，或者我们在这里等它完成
    responses = await batch_task
    
    end_time = time.perf_counter()

    for response in responses:
        content=response.content if hasattr(response, 'content') else str(response)
        print(f">>>响应内容：{content}")

    print(f"++++总运行耗时：{end_time- start_time:.2f}s+++++")

# 入口：执行异步主函数
if __name__ == "__main__":
    asyncio.run(demo_async_batch())