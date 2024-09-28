from langchain_community.llms import Ollama

# 初始化 Ollama 模型
llm = Ollama(model="llama3.2")

# 生成文本
response = llm.invoke("你跟我聊聊天看看怎么样？")
print(response)
