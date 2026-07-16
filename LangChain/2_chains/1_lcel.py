class Runnable:
    def run(self, data):
        raise NotImplementedError("子类必须实现run方法")

    # 重载 | 运算符，两个Runnable拼接成Chain
    def __or__(self, other):
        return Chain(self, other)


class Chain(Runnable):
    def __init__(self, *runnables):
        self.steps = list(runnables)

    def run(self, data):
        # 串行依次执行每一步，数据向后传递
        for step in self.steps:
            data = step.run(data)
        return data

    # Chain再拼接其他算子，合并步骤列表
    def __or__(self, other):
        return Chain(*self.steps, other)


# 自定义算子1：输入数字 +1
class AaaOne(Runnable):
    def run(self, data):
        # print(f"AaaOne处理前: {data}")
        res = data + 1
        # print(f"AaaOne处理后: {res}")
        return res


# 自定义算子2：输入数字 *2
class MulTwo(Runnable):
    def run(self, data):
        # print(f"MulTwo处理前: {data}")
        res = data * 2
        # print(f"MulTwo处理后: {res}")
        return res


# 自定义算子3：转为字符串
class ToStr(Runnable):
    def run(self, data):
        # print(f"ToStr处理前: {data}")
        res = str(data)
        # print(f"ToStr处理后: {res}")
        return res


# 链式拼接，使用 | 运算符
chain = AaaOne() | MulTwo() | ToStr()

# 执行流水线，输入初始数字5
result = chain.run(5)
print("\n流水线最终输出：", result)
print("输出类型：", type(result))