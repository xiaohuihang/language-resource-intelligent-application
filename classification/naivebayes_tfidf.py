import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(pos_file, neg_file):
    """加载语料数据，返回文本列表和标签列表"""
    texts, labels = [], []

    # 加载正面情感文本
    with open(pos_file, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            labels.append(1)  # 正面标签为1

    # 加载负面情感文本
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            labels.append(0)  # 负面标签为0

    return texts, labels

def chinese_tokenizer(text):
    """中文分词器，使用结巴分词"""
    return list(jieba.cut(text))

def extract_tfidf_features(texts):
    """提取TF-IDF特征"""
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,  # 使用中文分词
        max_features=5000,  # 最大特征数
        min_df=2,  # 最小文档频率
        max_df=0.8,  # 最大文档频率
        ngram_range=(1, 2)  # 使用1-2元语法
    )

    # 将文本转换为TF-IDF特征矩阵
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def train_nb_classifier(X_train, y_train):
    """训练朴素贝叶斯分类器"""
    # 创建多项式朴素贝叶斯模型
    nb_model = MultinomialNB(
        alpha=1.0,  # 平滑参数
        fit_prior=True  # 学习先验概率
    )

    # 训练模型
    nb_model.fit(X_train, y_train)
    return nb_model

def evaluate_model(model, X_test, y_test):
    """评估模型并输出报告"""
    # 预测测试集
    y_pred = model.predict(X_test)

    # 生成分类报告
    report = classification_report(y_test, y_pred,
                                   target_names=['负面', '正面'],
                                   digits=4)

    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    return report, y_pred, cm

def plot_confusion_matrix(cm):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()

def predict_example(model, vectorizer, text):
    """对单个样例进行预测"""
    # 分词
    tokens = chinese_tokenizer(text)
    print(f"分词结果: {'/'.join(tokens)}")

    # 提取特征
    features = vectorizer.transform([text])

    # 预测
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    # 输出结果
    sentiment = "正面" if prediction == 1 else "负面"
    pos_prob = proba[1] if len(proba) > 1 else proba[0]

    print(f"预测情感: {sentiment}")
    print(f"正面概率: {pos_prob:.4f}")
    print(f"负面概率: {1 - pos_prob:.4f}")
    return prediction

def get_top_features(model, vectorizer, n=10):
    """获取最重要的特征（词汇）"""
    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()

    # 朴素贝叶斯使用特征的对数概率
    # 这里我们计算每个特征在两类中的概率差异
    log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    # 找出差异最大的n个特征（正面）
    top_positive_indices = log_prob_diff.argsort()[-n:][::-1]
    top_positive_features = [(feature_names[i], log_prob_diff[i])
                             for i in top_positive_indices]

    # 找出差异最小的n个特征（负面）
    top_negative_indices = log_prob_diff.argsort()[:n]
    top_negative_features = [(feature_names[i], log_prob_diff[i])
                             for i in top_negative_indices]

    return top_positive_features, top_negative_features

def analyze_feature_probabilities(model, vectorizer, word):
    """分析特定词汇在朴素贝叶斯模型中的概率"""
    if word not in vectorizer.vocabulary_:
        print(f"词汇 '{word}' 不在特征中")
        return

    idx = vectorizer.vocabulary_[word]

    # 获取该特征的对数概率
    neg_log_prob = model.feature_log_prob_[0][idx]
    pos_log_prob = model.feature_log_prob_[1][idx]

    # 转换为实际概率（取指数）
    neg_prob = np.exp(neg_log_prob)
    pos_prob = np.exp(pos_log_prob)

    print(f"词汇: {word}")
    print(f"负面情感对数概率: {neg_log_prob:.4f} (实际概率: {neg_prob:.6f})")
    print(f"正面情感对数概率: {pos_log_prob:.4f} (实际概率: {pos_prob:.6f})")
    print(f"差异(正面-负面): {pos_log_prob - neg_log_prob:.4f}")
    print(f"更可能属于: {'正面' if pos_log_prob > neg_log_prob else '负面'}情感")

def main():
    # 1. 加载数据
    print("正在加载数据...")
    texts, labels = load_data('实验数据/pos.txt', '实验数据/neg.txt')
    print(f"数据加载完成，共{len(texts)}条样本")
    print(f"正面样本: {sum(labels)}条，负面样本: {len(labels) - sum(labels)}条")

    # 2. 提取TF-IDF特征
    print("\n正在提取TF-IDF特征...")
    X, vectorizer = extract_tfidf_features(texts)
    y = np.array(labels)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape[0]}条，测试集: {X_test.shape[0]}条")
    print(f"特征维度: {X_train.shape[1]}")

    # 4. 训练朴素贝叶斯分类器
    print("\n正在训练朴素贝叶斯分类器...")
    nb_model = train_nb_classifier(X_train, y_train)

    # 5. 评估模型并输出报告
    print("\n模型评估结果:")
    report, y_pred, cm = evaluate_model(nb_model, X_test, y_test)
    print(report)

    # 6. 输出混淆矩阵
    print("混淆矩阵:")
    print(cm)

    # 可视化混淆矩阵（可选）
    # plot_confusion_matrix(cm)

    # 7. 显示最重要的特征
    print("\n最重要的情感特征:")
    top_pos, top_neg = get_top_features(nb_model, vectorizer, n=10)
    print("正面情感词汇（正面概率远高于负面概率）:")
    for word, diff in top_pos:
        print(f"  {word}: {diff:.4f}")

    print("\n负面情感词汇（负面概率远高于正面概率）:")
    for word, diff in top_neg:
        print(f"  {word}: {diff:.4f}")

    # 8. 分析特定词汇的概率
    print("\n" + "=" * 50)
    print("特定词汇概率分析:")
    test_words = ["好", "坏", "不错", "失望"]
    for word in test_words:
        analyze_feature_probabilities(nb_model, vectorizer, word)
        print("-" * 40)

    # 9. 样例测试
    print("\n" + "=" * 50)
    print("样例数据测试:")

    # 测试样例
    test_samples = [
        "这部电影真的太棒了，演员演技出色，剧情扣人心弦！",
        "非常失望，完全浪费了我的时间和金钱。",
        "产品一般般，没什么特别的感觉。",
        "服务态度很好，下次还会再来。"
    ]

    for i, sample in enumerate(test_samples, 1):
        print(f"\n样例{i}: {sample}")
        predict_example(nb_model, vectorizer, sample)
        print("-" * 40)

    # 10. 输出模型先验概率
    print("\n先验概率:")
    print(f"负面情感先验概率: {np.exp(nb_model.class_log_prior_[0]):.4f}")
    print(f"正面情感先验概率: {np.exp(nb_model.class_log_prior_[1]):.4f}")

if __name__ == "__main__":
    main()