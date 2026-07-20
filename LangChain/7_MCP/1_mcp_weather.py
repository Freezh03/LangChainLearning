"""FastMCP 天气服务"""
from fastmcp import FastMCP

mcp = FastMCP("天气服务")


@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气信息
    Args:
        city：城市名称，如"北京"、"上海""、"广州"
    """

    # 模拟天气数据
    weather_data = {
        "北京": "晴天，气温25摄氏度，湿度 40%",
        "上海": "多云，气温28摄氏度，湿度 65%",
        "广州": "小雨，气温29摄氏度，湿度 80%",
        "深圳": "阴天，气温30摄氏度，湿度 75%",
    }

    if city in weather_data:
        return f"{city}天气：{weather_data[city]}"
    else:
        return f"{city}天气：晴，气温 22摄氏度，湿度50%"


mcp.run(transport="streamable-http", host="127.0.0.1", port=8080)
# mcp.run(transport="stdio") #本地模式
