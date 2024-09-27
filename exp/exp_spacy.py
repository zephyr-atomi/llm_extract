import spacy

# 加载 spaCy 模型
nlp = spacy.load("en_core_web_sm")


def generate_ngrams(text, max_n=3):
    """
    使用生成器逐个生成 n-grams。

    :param text: 输入的文本字符串
    :param max_n: n-gram 的最大 n 值，默认为 3
    :yield: 每次返回一个生成的 n-gram
    """
    # Step 1: 使用 spaCy 进行分词
    doc = nlp(text)

    # 提取tokens
    tokens = [token.text for token in doc]

    # Step 2: 逐个生成 n-grams 并 yield
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = tokens[i:i + n]
            yield " ".join(ngram)


# 示例：用生成器逐个生成 n-grams
text = "Steve Jobs founded Apple in 1976. He was born in San Francisco."

# 逐个获取 n-grams
for ngram in generate_ngrams(text):
    print(f"Generated n-gram: {ngram}")

from sentence_transformers import SentenceTransformer

# 加载 Sentence-BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')


def process_ngrams(ngrams):
    """
    对每个 n-gram 生成嵌入向量（embedding）

    :param ngrams: n-grams 生成器或列表
    :return: 每个 n-gram 的向量表示
    """
    # 将生成器 ngrams 转换为列表
    ngram_list = list(ngrams)

    # 使用 Sentence-BERT 生成 embedding
    embeddings = model.encode(ngram_list)

    # 输出每个 n-gram 的 embedding
    for ngram, embedding in zip(ngram_list, embeddings):
        print(f"n-gram: {ngram}, Embedding shape: {embedding.shape}")
        # 如果你希望进一步处理，可以在此对 embedding 做其他操作
        # 例如，你可以将 embedding 存储到某个地方，或者传递到后续模型中


# 示例：生成 n-grams
ngrams = generate_ngrams("Steve Jobs founded Apple in 1976. He was born in San Francisco.")

# 获取并处理 n-grams 的 embedding
process_ngrams(ngrams)


