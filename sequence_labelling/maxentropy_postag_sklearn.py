#!/usr/bin/env python3
"""
最大熵词性标注 - 修复版
"""
import nltk, random, numpy as np
from nltk.corpus import brown
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from collections import Counter

# 1. 数据准备
print("准备数据...")
tagged_sentences = list(brown.tagged_sents(categories='news', tagset='universal'))
random.seed(42)
random.shuffle(tagged_sentences)
train_data, test_data = tagged_sentences[:int(len(tagged_sentences) * 0.9)], tagged_sentences[
                                                                             int(len(tagged_sentences) * 0.9):]
print(f"训练集: {len(train_data)}句, 测试集: {len(test_data)}句")

# 2. 特征提取
def extract_features(sentence, i):
    word = sentence[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:] if len(word) >= 3 else word,
        'word.isupper()': int(word.isupper()),
        'word.istitle()': int(word.istitle()),
    }
    if i > 0:
        features['prev_word.lower()'] = sentence[i - 1][0].lower()
    else:
        features['BOS'] = 1
    if i < len(sentence) - 1:
        features['next_word.lower()'] = sentence[i + 1][0].lower()
    else:
        features['EOS'] = 1
    return features

# 3. 构建特征词汇表
print("构建特征词汇表...")
feature_counter = Counter()
for sentence in train_data[:500]:
    for i in range(len(sentence)):
        features = extract_features(sentence, i)
        for feat_name, feat_value in features.items():
            feature_str = f"{feat_name}={feat_value}"
            feature_counter[feature_str] += 1

feature_vocab = {feat: idx for idx, (feat, _) in enumerate(feature_counter.most_common(2000))}

# 4. 正确的特征向量化函数
def word_to_features(sentence, i):
    """为单个词生成特征向量"""
    features_vec = np.zeros(len(feature_vocab), dtype=np.float32)
    extracted = extract_features(sentence, i)
    for feat_name, feat_value in extracted.items():
        feature_str = f"{feat_name}={feat_value}"
        if feature_str in feature_vocab:
            features_vec[feature_vocab[feature_str]] = 1.0
    return features_vec


# 5. 准备训练数据
print("准备训练数据...")
X_train, y_train = [], []
for sentence in train_data[:1000]:
    for i in range(len(sentence)):
        X_train.append(word_to_features(sentence, i))
        y_train.append(sentence[i][1])

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
X_train = np.array(X_train)

# 6. 训练模型
print("训练最大熵模型...")
model = LogisticRegression(max_iter=50, C=0.1, multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train_encoded)
print(f"训练完成! 特征数: {len(feature_vocab)}, 类别数: {len(label_encoder.classes_)}")

# 7. 评估
print("\n评估模型...")
X_test, y_test = [], []
for sentence in test_data[:200]:  # 使用部分测试数据
    for i in range(len(sentence)):
        X_test.append(word_to_features(sentence, i))
        y_test.append(sentence[i][1])

y_pred = label_encoder.inverse_transform(model.predict(np.array(X_test)))
accuracy = np.mean(y_pred == y_test)
print(f"准确率: {accuracy:.2%}")


# 8. 正确的预测函数
def predict_sentence(sentence_text):
    """预测新句子 - 修复版"""
    words = sentence_text.split()
    fake_sentence = [(w, 'UNK') for w in words]

    # 为每个词单独生成特征向量
    features_list = []
    for i in range(len(words)):
        features_list.append(word_to_features(fake_sentence, i))

    # 预测每个词
    tags = label_encoder.inverse_transform(model.predict(np.array(features_list)))
    return list(zip(words, tags))


# 9. 演示
examples = [
    "The quick brown fox jumps over the lazy dog .",
    "Natural language processing is an exciting field ."
]

print("\n预测演示:")
for sent in examples:
    result = predict_sentence(sent)
    print(f"\n'{sent}'")
    print(f"结果: {' '.join([f'{w}/{t}' for w, t in result])}")

print("\n" + "=" * 60)
print("最大熵词性标注完成!")
print("=" * 60)