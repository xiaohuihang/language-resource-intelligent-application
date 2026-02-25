import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tag import HiddenMarkovModelTagger
from nltk.probability import LidstoneProbDist

def load_data(file_path):
    """加载数据"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    sentences.append(json.loads(line.strip()))
                except:
                    continue
    print(f"加载数据: {len(sentences)} 条")
    return sentences

def extract_features_and_labels(sentences):
    """提取特征和标签"""
    X, y = [], []
    entity_types = ['name', 'company', 'game', 'organization', 'movie',
                    'address', 'position', 'government', 'scene', 'book']

    for data in sentences:
        text, labels = data['text'], data['label']
        char_labels = ['O'] * len(text)

        # 将标签转换为BIO格式
        for entity_type in entity_types:
            if entity_type in labels:
                for entity, positions in labels[entity_type].items():
                    for start, end in positions:
                        if 0 <= start < end <= len(text):
                            char_labels[start] = f'B-{entity_type}'
                            for i in range(start + 1, end):
                                char_labels[i] = f'I-{entity_type}'

        # 为HMM准备观察序列（字符本身作为特征）
        observations = list(text)
        X.append(observations)
        y.append(char_labels)

    print(f"特征提取完成: {len(X)}个句子")
    return X, y

def prepare_hmm_training_data(X_train, y_train):
    """准备HMM训练数据格式"""
    train_data = []
    for observations, states in zip(X_train, y_train):
        train_data.append(list(zip(observations, states)))
    return train_data

def train_hmm_model(train_data):
    """训练HMM模型"""
    print("\n训练HMM模型...")

    # 使用nltk的HMM实现
    # 设置gamma值用于平滑（Lidstone平滑）
    hmm_tagger = HiddenMarkovModelTagger.train(
        train_data,
        estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins),
        verbose=True
    )

    print("模型训练完成!")
    return hmm_tagger

def evaluate_and_show_report(y_true, y_pred, dataset_name="测试集"):
    """评估并显示标签级别的精确率、召回率和F1值"""
    y_true_flat = [label for seq in y_true for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]

    print(f"\n{dataset_name}标签级别评估:")
    print("-" * 60)
    report = classification_report(y_true_flat, y_pred_flat, digits=4, zero_division=0)
    print(report)

    correct = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == p)
    total = len(y_true_flat)
    accuracy = correct / total if total > 0 else 0
    print(f"标签总体准确率: {accuracy:.4f} ({correct}/{total})")

def extract_entities(text, labels):
    """从标签序列中提取完整实体"""
    entities = []
    current_entity, start_idx, entity_text = None, 0, ""

    for i, (char, label) in enumerate(zip(text, labels)):
        if label.startswith('B-'):
            if current_entity is not None:
                entities.append({'text': entity_text, 'start': start_idx,
                                 'end': i, 'type': current_entity})
            current_entity = label[2:]
            start_idx, entity_text = i, char
        elif label.startswith('I-'):
            if current_entity is not None and label[2:] == current_entity:
                entity_text += char
        elif label == 'O' and current_entity is not None:
            entities.append({'text': entity_text, 'start': start_idx,
                             'end': i, 'type': current_entity})
            current_entity, entity_text = None, ""

    if current_entity is not None:
        entities.append({'text': entity_text, 'start': start_idx,
                         'end': len(text), 'type': current_entity})
    return entities

def evaluate_entities(X, y_true, y_pred, dataset_name="测试集"):
    """评估完整实体的精确率、召回率和F1值"""
    true_entities, pred_entities = [], []

    for i in range(len(X)):
        text = ''.join(X[i])  # 恢复原始文本
        true_ents = extract_entities(text, y_true[i])
        pred_ents = extract_entities(text, y_pred[i])

        true_entities.append(true_ents)
        pred_entities.append(pred_ents)

    # 计算实体级别的指标
    tp, fp, fn = 0, 0, 0

    for true_ents, pred_ents in zip(true_entities, pred_entities):
        true_set = set((ent['start'], ent['end'], ent['type']) for ent in true_ents)
        pred_set = set((ent['start'], ent['end'], ent['type']) for ent in pred_ents)

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    # 计算精确率、召回率和F1值
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{dataset_name}实体级别评估:")
    print("-" * 60)
    print(f"精确率: {precision:.4f} (TP: {tp}, FP: {fp})")
    print(f"召回率: {recall:.4f} (TP: {tp}, FN: {fn})")
    print(f"F1值: {f1:.4f}")

    # 计算实体总体准确率
    correct_sentences = 0
    total_sentences = len(true_entities)

    for true_ents, pred_ents in zip(true_entities, pred_entities):
        true_set = set((ent['start'], ent['end'], ent['type'], ent['text']) for ent in true_ents)
        pred_set = set((ent['start'], ent['end'], ent['type'], ent['text']) for ent in pred_ents)

        if true_set == pred_set:
            correct_sentences += 1

    entity_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0
    print(f"实体总体准确率: {entity_accuracy:.4f} ({correct_sentences}/{total_sentences})")

    return precision, recall, f1

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """评估模型性能"""
    print("=" * 60)

    # 测试集预测
    y_test_pred = []
    for observations in X_test:
        tagged = model.tag(observations)
        pred_labels = [tag for _, tag in tagged]
        y_test_pred.append(pred_labels)

    evaluate_and_show_report(y_test, y_test_pred, "测试集")
    evaluate_entities(X_test, y_test, y_test_pred, "测试集")

    # 训练集预测（可选）
    if X_train is not None and y_train is not None:
        y_train_pred = []
        for observations in X_train:
            tagged = model.tag(observations)
            pred_labels = [tag for _, tag in tagged]
            y_train_pred.append(pred_labels)

        evaluate_and_show_report(y_train, y_train_pred, "训练集")
        evaluate_entities(X_train, y_train, y_train_pred, "训练集")

    return y_test_pred

def predict_and_display(model, text, title="预测结果"):
    """预测并显示结果"""
    observations = list(text)
    tagged = model.tag(observations)
    labels = [tag for _, tag in tagged]

    entities = extract_entities(text, labels)

    print(f"\n{title}:")
    print(f"文本: {text}")

    if entities:
        print("实体识别结果:")
        for entity in entities:
            print(f"  [{entity['start']}:{entity['end']}] {entity['type']}: {entity['text']}")
    else:
        print("  未识别到命名实体")
    return entities

def display_hmm_model_info(model):
    """显示HMM模型信息"""
    print("\n" + "=" * 60)
    print("HMM模型信息")
    print("=" * 60)

    try:
        # 获取状态和符号信息
        if hasattr(model, '_states'):
            states = list(model._states)
            print(f"状态（标签）数量: {len(states)}")
            print(f"状态列表: {states}")

        if hasattr(model, '_symbols'):
            symbols = list(model._symbols)
            print(f"观察符号（字符）数量: {len(symbols)}")
            print(f"观察符号示例（前20个）: {symbols[:20]}")

        # 尝试显示一些转移概率
        print(f"\n状态转移概率示例:")
        if hasattr(model, '_transitions'):
            transition_samples = []
            # 获取前3个状态作为源状态
            states_list = list(model._states)
            for from_state in states_list[:3]:
                # 获取前3个目标状态
                for to_state in states_list[:3]:
                    try:
                        prob = model._transitions[from_state].prob(to_state)
                        transition_samples.append((from_state, to_state, prob))
                    except:
                        continue

            for from_state, to_state, prob in transition_samples[:10]:
                print(f"  {from_state} -> {to_state}: {prob:.6f}")

        # 尝试显示一些发射概率
        print(f"\n观察符号发射概率示例:")
        if hasattr(model, '_outputs'):
            emission_samples = []
            # 获取前3个状态
            states_list = list(model._states)
            symbols_list = list(model._symbols)

            for state in states_list[:3]:
                # 获取前5个符号
                for symbol in symbols_list[:5]:
                    try:
                        prob = model._outputs[state].prob(symbol)
                        emission_samples.append((state, symbol, prob))
                    except:
                        continue

            for state, symbol, prob in emission_samples[:10]:
                # 对特殊字符进行转义显示
                if symbol == '\n':
                    symbol_display = "\\n"
                elif symbol == '\t':
                    symbol_display = "\\t"
                elif symbol == ' ':
                    symbol_display = "' '"
                else:
                    symbol_display = symbol
                print(f"  状态{state} -> 符号'{symbol_display}': {prob:.6f}")

        # 显示初始状态概率
        print(f"\n初始状态概率示例:")
        if hasattr(model, '_priors'):
            states_list = list(model._states)
            for state in states_list[:5]:
                try:
                    prob = model._priors.prob(state)
                    print(f"  初始状态 {state}: {prob:.6f}")
                except:
                    continue

    except Exception as e:
        print(f"获取模型信息时出错: {e}")
        print("模型信息显示不完整，但模型可以正常使用。")

def main():
    print("=" * 60)
    print("HMM命名实体识别演示程序")
    print("=" * 60)

    # 1. 加载数据
    sentences = load_data("data/ner_data.json")

    # 2. 特征提取
    X, y = extract_features_and_labels(sentences)

    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n数据集划分: 训练集{len(X_train)}个句子, 测试集{len(X_test)}个句子")

    # 4. 准备HMM训练数据
    train_data = prepare_hmm_training_data(X_train, y_train)

    # 5. 训练HMM模型
    hmm_model = train_hmm_model(train_data)

    # 6. 评估模型
    print("\n模型评估结果:")
    y_test_pred = evaluate_model(hmm_model, X_test, y_test, X_train[:1000], y_train[:1000])

    # 7. 演示新样例预测
    print("\n" + "=" * 60)
    print("新样例预测演示")
    print("=" * 60)

    demo_texts = [
        "腾讯公司的马化腾在深圳总部发表了演讲。",
        "王者荣耀和英雄联盟是目前最受欢迎的游戏。",
        "阿里巴巴集团与工商银行合作推出了新的金融服务。",
        "北京的李明教授参加了国际人工智能会议。",
        "电影《流浪地球》和《战狼2》在中国取得了巨大成功。"
    ]

    for i, text in enumerate(demo_texts):
        predict_and_display(hmm_model, text, f"样例{i + 1}")

    # 8. 显示HMM模型信息
    display_hmm_model_info(hmm_model)

if __name__ == "__main__":
    main()