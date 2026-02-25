#!/usr/bin/env python3
"""
最大熵词性标注 - 使用NLTK的MaxEnt实现
保持与CRF相同的特征提取和评估流程
"""
import nltk
import random
from nltk.corpus import brown
from nltk.classify import MaxentClassifier
from sklearn.metrics import classification_report
from collections import Counter

# 1. 数据准备（保持与CRF一致）
print("1. 准备数据...")
tagged_sentences = list(brown.tagged_sents(categories='news', tagset='universal'))

random.seed(42)
random.shuffle(tagged_sentences)
split_idx = int(len(tagged_sentences) * 0.9)
train_data = tagged_sentences[:split_idx]
test_data = tagged_sentences[split_idx:]
print(f"   训练集: {len(train_data)}句, 测试集: {len(test_data)}句")

# 2. 特征提取函数（与CRF相同）
def extract_features(sentence, i):
    """提取第i个词的特征"""
    word = sentence[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if i > 0:
        prev_word = sentence[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word.istitle()': next_word.istitle(),
        })
    else:
        features['EOS'] = True

    return features

# 3. 准备训练数据（格式化为NLTK所需格式）
print("\n2. 准备训练数据...")

def prepare_training_examples(tagged_sentences):
    """将数据转换为NLTK MaxEnt所需的格式"""
    training_examples = []
    for sentence in tagged_sentences:
        for i, (word, tag) in enumerate(sentence):
            # 特征字典 -> 特征集合（NLTK需要字典键）
            features = extract_features(sentence, i)
            # NLTK要求特征是简单字典（值转换为字符串）
            nltk_features = {}
            for k, v in features.items():
                if isinstance(v, bool):
                    nltk_features[k] = v
                else:
                    nltk_features[k] = str(v)
            training_examples.append((nltk_features, tag))
    return training_examples

train_examples = prepare_training_examples(train_data[:500])  # 限制数据量以加速训练
print(f"   训练样本数: {len(train_examples)} (为加速只使用前500句)")

# 4. 训练NLTK最大熵模型
print("\n3. 训练NLTK最大熵模型...")
print("   (训练可能需要几分钟，使用GIS算法)...")
maxent_tagger = MaxentClassifier.train(
    train_examples,
    algorithm='gis',  # 通用迭代缩放算法
    max_iter=10,  # 迭代次数（减少以加速）
    trace=3  # 显示训练进度
)
print("   训练完成!")

# 5. 评估模型
print("\n4. 评估模型性能...")
y_true, y_pred = [], []

for sentence in test_data:
    for i, (word, true_tag) in enumerate(sentence):
        # 提取特征（格式化为NLTK格式）
        features = extract_features(sentence, i)
        nltk_features = {k: str(v) for k, v in features.items()}

        # 预测
        pred_tag = maxent_tagger.classify(nltk_features)

        y_true.append(true_tag)
        y_pred.append(pred_tag)

# 计算准确率
accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
print(f"   整体准确率: {accuracy:.4f} ({accuracy:.2%})")

# 6. 与CRF对比
print("\n" + "=" * 60)
print("5. MaxEnt vs CRF 对比分析")
print("=" * 60)

# 显示最常见的错误
print("\n   最常见的预测错误:")
error_counts = Counter()
for true, pred in zip(y_true, y_pred):
    if true != pred:
        error_counts[(true, pred)] += 1

for (true, pred), count in error_counts.most_common(5):
    print(f"     {true} -> {pred}: {count}次")

# 7. 新句子预测演示
print("\n6. 新句子预测演示:")

def predict_sentence(sentence, classifier):
    """预测单个句子的词性"""
    words = sentence.split()
    fake_sentence = [(w, 'UNK') for w in words]

    results = []
    for i in range(len(words)):
        features = extract_features(fake_sentence, i)
        nltk_features = {k: str(v) for k, v in features.items()}
        tag = classifier.classify(nltk_features)
        results.append((words[i], tag))

    return results

examples = [
    "The quick brown fox jumps over the lazy dog .",
    "Natural language processing is an exciting field ."
]

for sent in examples:
    result = predict_sentence(sent, maxent_tagger)
    print(f"\n   '{sent}'")
    print(f"   结果: {' '.join([f'{w}/{t}' for w, t in result])}")

print("\n" + "=" * 60)
print("NLTK最大熵实现完成!")
print("=" * 60)

# 附加：特征重要性（显示前10个最重要的特征）
print("\n" + "=" * 60)
print("附加：模型特征分析")
print("=" * 60)
print("   最重要的10个特征权重:")
for feat, weight in list(maxent_tagger.show_most_informative_features(10)):
    print(f"     {feat}")