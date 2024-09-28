from langchain_community.llms import Ollama


class LangChainOutputParser:
    def __init__(self, model_name):
        self.llm = Ollama(model=model_name)

    def extract_information_as_json_for_context(self, context, IE_query, output_data_structure):
        # 构建原始输入
        prompt = f"Context: {context}\nQuery: {IE_query}\nOutput format: {output_data_structure}\nExtract information in JSON format."

        # 打印或记录输入
        print(f"Raw input to Llama model:\n{prompt}\n")

        # 调用模型并获得原始输出
        response = self.llm.invoke(prompt)

        # 打印或记录输出
        print(f"Raw output from Llama model:\n{response}\n")

        return response


# 模拟的文档
documents = [
    "LangChain 是一个用于构建大型语言模型驱动的应用的框架。",
    "Ollama 模型在本地和远程环境下都能运行大规模语言模型。",
    "AI 正在改变现代商业的各个方面，从自动化到数据分析。"
]

# 你的问题和输出结构定义
IE_query = "Extract the key points about how AI is influencing businesses."
output_data_structure = "{'key_influences': []}"

# 初始化输出解析器
parser = LangChainOutputParser(model_name="llama3.2")

# 对每个文档调用并提取 JSON 格式信息，同时捕获原始输入和输出
output_jsons = list(
    map(
        lambda context: parser.extract_information_as_json_for_context(
            context=context,
            IE_query=IE_query,
            output_data_structure=output_data_structure
        ),
        documents
    )
)

# 输出结果
for idx, output in enumerate(output_jsons):
    print(f"Document {idx + 1} Output:")
    print(output)
