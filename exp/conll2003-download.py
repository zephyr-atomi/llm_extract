from datasets import load_dataset

dataset = load_dataset("conll2003")
print(dataset)

# 访问训练集
train_dataset = dataset['train']

# 查看训练集中的前几个样本
for sample in train_dataset[:5]:
    print(sample)

# 获取标签列表
ner_tags = dataset['train'].features['ner_tags'].feature.names

# 查看标签名
print(ner_tags)

print("查看训练集中的第一个样本，并解码NER标签")
sample = dataset['train'][100]
tokens = sample['tokens']
ner_tag_ids = sample['ner_tags']
ner_labels = [ner_tags[id] for id in ner_tag_ids]

print("Tokens:", tokens)
print("NER Tags:", ner_labels)
