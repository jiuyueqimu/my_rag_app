import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "sk-05556d0f00434ff8a4e61d5202578062"          # 替换成新生成的
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

llm = ChatOpenAI(model="deepseek-chat", temperature=0)
response = llm.invoke("你好，请简单介绍一下你自己")
print(response.content)