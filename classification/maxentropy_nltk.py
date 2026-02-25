import jieba
import nltk
import random
from nltk.classify import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.metrics import ConfusionMatrix

def load_and_prepare_data(pos_file='./实验数据/pos.txt', neg_file='./实验数据/neg.txt', test_ratio=0.2):
    """
    加载数据并准备为NLTK格式。
    NLTK MaxentClassifier需要(特征字典, 标签)格式的训练数据。
    """
    data = []
    # 加载正面评价
    with open(pos_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:  # 确保不是空行
                data.append((text, 'pos'))  # 正面标签为'pos'

    # 加载负面评价
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                data.append((text, 'neg'))  # 负面标签为'neg'

    print(f"数据加载完成，共{len(data)}条样本")
    print(f"正面样本: {sum(1 for _, label in data if label == 'pos')}条")
    print(f"负面样本: {sum(1 for _, label in data if label == 'neg')}条")

    # 打乱数据并划分训练集和测试集
    random.seed(42)
    random.shuffle(data)

    split_point = int(len(data) * (1 - test_ratio))
    train_data = data[:split_point]
    test_data = data[split_point:]

    print(f"训练集: {len(train_data)}条，测试集: {len(test_data)}条")

    return train_data, test_data

def extract_features(text, use_bigrams=True):
    """
    将文本转换为NLTK分类器所需的特征字典格式。
    默认使用词袋模型（二元特征）。
    """
    # 使用jieba进行中文分词
    words = list(jieba.cut(text))

    # 基础特征：词的存在性（词袋模型）
    features = {f"word_{word}": True for word in words if len(word) > 1}  # 过滤单字

    # 可选的：添加二元组特征（捕捉词语共现）
    if use_bigrams and len(words) > 1:
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        for bigram in bigrams:
            features[f"bigram_{bigram[0]}_{bigram[1]}"] = True

    # 添加文本长度特征（连续特征离散化）
    text_len = len(text)
    if text_len < 20:
        features['length_short'] = True
    elif text_len < 50:
        features['length_medium'] = True
    else:
        features['length_long'] = True

    return features

def prepare_featuresets(train_data, test_data):
    """准备训练和测试的特征集"""
    train_featuresets = [(extract_features(text), label) for text, label in train_data]
    test_featuresets = [(extract_features(text), label) for text, label in test_data]

    # 收集所有特征用于分析
    all_features = set()
    for features, _ in train_featuresets:
        all_features.update(features.keys())

    print(f"特征提取完成，共{len(all_features)}个唯一特征")
    return train_featuresets, test_featuresets

def train_maxent_classifier(train_featuresets, algorithm, max_iter):
    """
    训练NLTK最大熵分类器[citation:10]。
    参数说明：
    - algorithm: 训练算法，可选'GIS'（通用迭代缩放）或'IIS'（改进的迭代缩放）
    - max_iter: 最大迭代次数
    """
    print(f"\n正在使用{algorithm}算法训练最大熵分类器（最大迭代次数: {max_iter}）...")

    # 训练最大熵分类器
    classifier = MaxentClassifier.train(
        train_featuresets,
        algorithm=algorithm,
        max_iter=max_iter,
        trace=3  # 显示训练进度
    )

    return classifier

def evaluate_classifier(classifier, test_featuresets):
    """评估分类器性能"""
    print("\n模型评估结果:")

    # 计算准确率
    acc = accuracy(classifier, test_featuresets)
    print(f"准确率: {acc:.4f}")

    # 生成预测标签和真实标签
    refsets = {'pos': set(), 'neg': set()}
    testsets = {'pos': set(), 'neg': set()}

    for i, (feats, label) in enumerate(test_featuresets):
        refsets[label].add(i)
        predicted = classifier.classify(feats)
        testsets[predicted].add(i)

    # 计算精确率、召回率和F1值
    for label in ['pos', 'neg']:
        precision = nltk.precision(refsets[label], testsets[label])
        recall = nltk.recall(refsets[label], testsets[label])
        f1 = nltk.f_measure(refsets[label], testsets[label])

        print(f"{label}类别:")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1值: {f1:.4f}")

    # 生成混淆矩阵
    actual = [label for _, label in test_featuresets]
    predicted = [classifier.classify(feats) for feats, _ in test_featuresets]
    cm = ConfusionMatrix(actual, predicted)
    print("\n混淆矩阵:")
    print(cm)

    return acc

def analyze_features(classifier, n=10):
    """显示最有信息量的特征"""
    print(f"\n最有信息量的前{n}个特征:")
    try:
        # NLTK最大熵分类器提供show_most_informative_features方法
        classifier.show_most_informative_features(n)
    except AttributeError:
        print("该分类器不支持特征重要性分析")
        #负数权重：该特征的出现降低该类别的概率
        #正数权重：该特征的出现提高该类别的概率

def predict_example(classifier, text):
    """对单个样例进行预测和解释"""
    print(f"\n样例: {text}")

    # 分词
    words = list(jieba.cut(text))
    print(f"分词结果: {'/'.join(words)}")

    # 提取特征
    features = extract_features(text)

    # 预测
    predicted = classifier.classify(features)

    # 获取概率分布
    prob_dist = classifier.prob_classify(features)
    pos_prob = prob_dist.prob('pos')
    neg_prob = prob_dist.prob('neg')

    print(f"预测情感: {predicted} (正面概率: {pos_prob:.4f}, 负面概率: {neg_prob:.4f})")

    # 分析关键特征对决策的影响
    print("关键特征影响分析:")
    for feat in list(features.keys())[:5]:  # 只显示前5个特征
        if hasattr(classifier, 'weights'):
            # 注意：NLTK内部权重结构复杂，这里简化处理
            print(f"  特征 '{feat}' 出现在样本中")

    return predicted, pos_prob

def main():
    """主函数：完整的NLTK最大熵分类流程"""
    # 1. 加载数据
    print("=" * 50)
    print("步骤1: 加载数据")
    print("=" * 50)
    train_data, test_data = load_and_prepare_data()

    # 2. 提取特征
    print("\n" + "=" * 50)
    print("步骤2: 提取特征")
    print("=" * 50)
    train_featuresets, test_featuresets = prepare_featuresets(train_data, test_data)

    # 3. 训练最大熵分类器
    print("\n" + "=" * 50)
    print("步骤3: 训练最大熵分类器")
    print("=" * 50)
    classifier = train_maxent_classifier(train_featuresets, algorithm='GIS', max_iter=50)

    # 4. 评估模型
    print("\n" + "=" * 50)
    print("步骤4: 评估模型")
    print("=" * 50)
    accuracy_score = evaluate_classifier(classifier, test_featuresets)

    # 5. 特征分析
    print("\n" + "=" * 50)
    print("步骤5: 特征分析")
    print("=" * 50)
    analyze_features(classifier, n=10)

    # 6. 样例测试
    print("\n" + "=" * 50)
    print("步骤6: 样例测试")
    print("=" * 50)

    test_samples = [
        "这部电影真的太棒了，演员演技出色，剧情扣人心弦！",
        "非常失望，完全浪费了我的时间和金钱。",
        "产品一般般，没什么特别的感觉。",
        "服务态度很好，下次还会再来。"
    ]

    for i, sample in enumerate(test_samples, 1):
        print(f"\n样例 {i}:")
        predict_example(classifier, sample)
        print("-" * 40)

    # 7. 模型保存（可选）
    print("\n" + "=" * 50)
    print("附加功能: 模型保存与加载")
    print("=" * 50)

    # 保存模型（使用pickle）
    import pickle
    with open('maxent_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("模型已保存为 'maxent_classifier.pkl'")

    # 加载模型示例
    # with open('maxent_classifier.pkl', 'rb') as f:
    #     loaded_classifier = pickle.load(f)
    # print("模型加载成功！")

if __name__ == "__main__":
    # 确保必要的库已安装
    main()