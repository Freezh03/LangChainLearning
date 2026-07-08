from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import SequentialChain
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")

schainA_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位精通各领域知识的知名教授"),
        ("human", "请你先尽可能详细的解释一下：{knowledge}，并且{action}")
    ]
)

schainA_chains = LLMChain(llm=llm,
    prompt=schainA_template,
    verbose=True,
    output_key="schainA_chains_key"
    )

# schainA_chains.invoke({
# "knowledge": "中国的篮球怎么样？",
# "action": "举一个实际的例子"
# }
# )
schainB_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你非常善于提取文本中的重要信息，并做出简短的总结"),
        ("human", "这是针对一个提问完整的解释说明内容：{schainA_chains_key}"),
        ("human", "请你根据上述说明，尽可能简短的输出重要的结论，请控制在100个字以内"),
    ]
)

schainB_chains = LLMChain(llm=llm,
    prompt=schainB_template,
    verbose=True,
    output_key='schainB_chains_key'
    )


Seq_chain = SequentialChain(
    chains=[schainA_chains, schainB_chains],
    input_variables=["knowledge", "action"],
    output_variables=["schainA_chains_key","schainB_chains_key"],
    verbose=True
    )

response = Seq_chain.invoke({
    "knowledge":"中国足球为什么踢得烂",
    "action":"举一个实际的例子"
    }
    )
print(response)
