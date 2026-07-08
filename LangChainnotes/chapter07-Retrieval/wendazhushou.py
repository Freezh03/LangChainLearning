#导入必须的包
from langchain_community.document_loaders import UnstructuredExcelLoader,Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
#导入聊天所需的模块
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
#定义chatdoc
class ChatDoc():
    def __init__(self):
        self.doc = None
        self.splitText = [] #分割后的文本
        self.template = [
        ("system",
        "你是一个处理文档的秘书,你从不说自己是一个大模型或者AI助手,你会根据下面提供的上下文内容来继续回答问题.\n 上下文内容\n {context} \n"),
        ("human", "你好！"),
        ("ai", "您好，我是智能小智"),
        ("human", "{question}"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(self.template)

    def getFile(self):
        doc = self.doc
        loaders = {
            "docx": Docx2txtLoader
        }
        file_extension = doc.split(".")[-1]
        loader_class = loaders.get(file_extension)
        if loader_class:
            try:
                loader = loader_class(doc)
                text = loader.load()
                return text
            except Exception as e:
                print(f"Error loading {file_extension} files:{e}")
        else:
            print(f"Unsupported file extension: {file_extension}")
            return None
    
    #处理文档的函数
    def splitSentences(self):
        full_text = self.getFile() #获取文档内容
        if full_text != None:
            #对文档进行分割
            # Split the document
            text_split = CharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=10,
                separator="\n\n",
                length_function=len,
                is_separator_regex=False
            )
            texts = text_split.split_documents(full_text)
            self.splitText = texts

    #向量化与向量存储
    def embeddingAndVectorDB(self):
        # embeddings = OpenAIEmbeddings(
        # model="BAAI/bge-m3",
        # api_key=os.getenv("SILICON_API_KEY"),
        # base_url="https://api.siliconflow.cn/v1"
        # )
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(
            documents=self.splitText,
            embedding=embeddings,
        )
        return db
    
    #提问并找到相关的文本块
    def askAndFindFiles(self, question):
        db = self.embeddingAndVectorDB()
        print(db._collection.count())
        #retriever = db.as_retriever(search_type="mmr")
        retriever = db.as_retriever()
        return retriever.invoke(input=question)
    
    
    #用自然语言和文档聊天
    def chatWithDoc(self, question):
        _content = ""
        context = self.askAndFindFiles(question)
        for i in context:
            _content += i.page_content
        print(f"{_content}", "_content")
        messages = self.prompt.format_messages(context=_content, question=question)
        print("message:", messages)
        llm = ChatOpenAI(model="gpt-4",
            temperature=0,)
        return llm.invoke(messages)
    
chat_doc = ChatDoc()
chat_doc.doc = "./asset/load/12-langchain.docx"
chat_doc.splitSentences()
response = chat_doc.chatWithDoc("langchain的核心模块有哪些")
print(response.content)