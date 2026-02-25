#!/usr/bin/env python3
"""
条件随机场(CRF)词性标注 - 教学实现
使用相同数据集(Brown news)与HMM对比
"""
import nltk
import random
from nltk.corpus import brown
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

# 1. 数据准备（保持与HMM实验一致）
print("1. 准备数据...")
tagged_sentences = list(brown.tagged_sents(categories='news', tagset='universal'))

# 随机划分（使用相同种子保证可比性）
random.seed(42)
random.shuffle(tagged_sentences)
split_idx = int(len(tagged_sentences) * 0.9)
train_data = tagged_sentences[:split_idx]
test_data = tagged_sentences[split_idx:]
print(f"   训练集: {len(train_data)}句, 测试集: {len(test_data)}句")

# 2. 特征提取函数（CRF的核心）
def extract_features(sentence, i):
    """提取第i个词的特征"""
    word = sentence[i][0]

    # 基础特征
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],  # 后缀
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    # 上下文特征（前一个词）
    if i > 0:
        prev_word = sentence[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # 句子开头

    # 上下文特征（后一个词）
    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word.istitle()': next_word.istitle(),
        })
    else:
        features['EOS'] = True  # 句子结尾

    return features

def prepare_crf_data(tagged_sentences):
    """将数据转换为CRF需要的格式"""
    X, y = [], []
    for sentence in tagged_sentences:
        X_sent = [extract_features(sentence, i) for i in range(len(sentence))]
        y_sent = [tag for _, tag in sentence]
        X.append(X_sent)
        y.append(y_sent)
    return X, y


print("\n2. 提取特征...")
X_train, y_train = prepare_crf_data(train_data)
X_test, y_test = prepare_crf_data(test_data)
print(f"   特征示例: {list(X_train[0][0].keys())[:8]}...")

# 3. 训练CRF模型
print("\n3. 训练CRF模型...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',  # 使用L-BFGS优化算法
    c1=0.1,  # L1正则化系数（控制特征稀疏性）
    c2=0.1,  # L2正则化系数（控制权重幅度）
    max_iterations=100,
    all_possible_transitions=True  # 允许所有可能的转移
)
crf.fit(X_train, y_train)
print("   训练完成!")

# 4. 评估模型
print("\n4. 评估模型性能...")
y_pred = crf.predict(X_test)

# 扁平化标签用于计算准确率
y_test_flat = [tag for sent in y_test for tag in sent]
y_pred_flat = [tag for sent in y_pred for tag in sent]

# 计算准确率
accuracy = sum(y_t == y_p for y_t, y_p in zip(y_test_flat, y_pred_flat)) / len(y_test_flat)
print(f"   整体准确率: {accuracy:.4f} ({accuracy:.2%})")

# 详细分类报告
print("\n5. 详细分类报告:")
labels = sorted(set(y_test_flat + y_pred_flat))
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))

# 6. 与HMM性能对比分析
print("\n" + "=" * 60)
print("6. CRF 特征分析")
print("=" * 60)

# 统计特征数量（展示CRF的特征丰富性）
feature_counts = Counter()
for sent_features in X_train[:100]:  # 抽样查看
    for word_features in sent_features:
        feature_counts.update(word_features.keys())

print(f"   CRF使用的特征类型数: {len(feature_counts)}")
print(f"   最常使用的特征: {feature_counts.most_common(5)}")

# 性能对比预期
print(f"   - CRF (实际测得): {accuracy:.2%} 准确率")

# 7. 新句子预测演示
print("\n7. 新句子预测演示:")

def predict_sentence(sentence, crf_model):
    """对单个句子进行预测"""
    words = sentence.split()
    # 构建假标注用于特征提取（预测时我们不知道标签）
    fake_tags = ['UNK'] * len(words)
    fake_sentence = list(zip(words, fake_tags))

    # 提取特征并预测
    X_sent = [extract_features(fake_sentence, i) for i in range(len(words))]
    tags = crf_model.predict_single(X_sent)
    return list(zip(words, tags))


examples = [
    "The quick brown fox jumps over the lazy dog .",
    "Natural language processing is an exciting field ."
]

for sent in examples:
    result = predict_sentence(sent, crf)
    print(f"\n   '{sent}'")
    print(f"   结果: {' '.join([f'{w}/{t}' for w, t in result])}")

print("\n" + "=" * 60)
print("演示完成!")
print("=" * 60)