import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ---------- 密钥配置（正确版）----------
os.environ["OPENAI_API_KEY"] = "sk-62f62426ef14428b8ab2098626475b91"           # 填你重新生成的
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
# -----------------------------------------

@tool
def multiply(first_int: int, second_int: int) -> int:
    """计算两个整数的乘积。"""
    return first_int * second_int

llm = ChatOpenAI(model="deepseek-chat", temperature=0)
tools = [multiply]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个有用的助手，可以使用工具。"
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "1234乘以5678等于多少？"}]}
    )
    print("最终答案：", result["messages"][-1].content)