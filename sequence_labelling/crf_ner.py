import json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report

def load_data(file_path):
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
    X, y = [], []
    entity_types = ['name', 'company', 'game', 'organization', 'movie',
                    'address', 'position', 'government', 'scene', 'book']
    for data in sentences:
        text, labels = data['text'], data['label']
        char_labels = ['O'] * len(text)
        for entity_type in entity_types:
            if entity_type in labels:
                for entity, positions in labels[entity_type].items():
                    for start, end in positions:
                        if 0 <= start < end <= len(text):
                            char_labels[start] = f'B-{entity_type}'
                            for i in range(start + 1, end):
                                char_labels[i] = f'I-{entity_type}'

        features = []
        for i, char in enumerate(text):
            feat = {'char': char, 'char.lower': char.lower(), 'is_first': i == 0,
                    'is_last': i == len(text) - 1, 'is_digit': char.isdigit(),
                    'is_alpha': char.isalpha()}
            if i > 0:
                feat['-1:char'] = text[i - 1]
                feat['-1:char.lower'] = text[i - 1].lower()
            if i > 1:
                feat['-2:char'] = text[i - 2]
            if i < len(text) - 1:
                feat['+1:char'] = text[i + 1]
                feat['+1:char.lower'] = text[i + 1].lower()
            if i < len(text) - 2:
                feat['+2:char'] = text[i + 2]
            features.append(feat)
        X.append(features)
        y.append(char_labels)

    print(f"特征提取完成: {len(X)}个句子")
    return X, y

def train_crf_model(X_train, y_train):
    print("\n训练CRF模型...")
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50,
              all_possible_transitions=True, verbose=True)
    crf.fit(X_train, y_train)
    print("模型训练完成!")
    return crf

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
        # 从特征中恢复原始文本
        text = ''.join([feat['char'] for feat in X[i]])
        true_ents = extract_entities(text, y_true[i])
        pred_ents = extract_entities(text, y_pred[i])

        true_entities.append(true_ents)
        pred_entities.append(pred_ents)

    # 计算实体级别的指标
    tp, fp, fn = 0, 0, 0

    for true_ents, pred_ents in zip(true_entities, pred_entities):
        # 将实体转换为元组以便比较
        true_set = set((ent['start'], ent['end'], ent['type']) for ent in true_ents)
        pred_set = set((ent['start'], ent['end'], ent['type']) for ent in pred_ents)

        tp += len(true_set & pred_set)  # 真正例
        fp += len(pred_set - true_set)  # 假正例
        fn += len(true_set - pred_set)  # 假反例

    # 计算精确率、召回率和F1值
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{dataset_name}实体级别评估:")
    print("-" * 60)
    print(f"精确率: {precision:.4f} (TP: {tp}, FP: {fp})")
    print(f"召回率: {recall:.4f} (TP: {tp}, FN: {fn})")
    print(f"F1值: {f1:.4f}")

    # 计算实体总体准确率（所有实体都正确预测的句子比例）
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
    y_test_pred = model.predict(X_test)
    evaluate_and_show_report(y_test, y_test_pred, "测试集")
    evaluate_entities(X_test, y_test, y_test_pred, "测试集")

    # 训练集预测（可选）
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        evaluate_and_show_report(y_train, y_train_pred, "训练集")
        evaluate_entities(X_train, y_train, y_train_pred, "训练集")
    return y_test_pred

def predict_and_display(model, text, title="预测结果"):
    """预测并显示结果"""
    features = []
    for i, char in enumerate(text):
        feat = {'char': char, 'char.lower': char.lower(), 'is_first': i == 0,
                'is_last': i == len(text) - 1, 'is_digit': char.isdigit(),
                'is_alpha': char.isalpha()}
        if i > 0:
            feat['-1:char'] = text[i - 1]
            feat['-1:char.lower'] = text[i - 1].lower()
        if i < len(text) - 1:
            feat['+1:char'] = text[i + 1]
            feat['+1:char.lower'] = text[i + 1].lower()
        features.append(feat)

    labels = model.predict([features])[0]
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

def main():
    print("=" * 60)
    print("CRF命名实体识别演示程序")
    print("=" * 60)

    # 1. 加载数据
    sentences = load_data("data/ner_data.json")

    # 2. 特征提取
    X, y = extract_features_and_labels(sentences)

    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n数据集划分: 训练集{len(X_train)}个句子, 测试集{len(X_test)}个句子")

    # 4. 训练模型
    crf_model = train_crf_model(X_train, y_train)

    # 5. 评估模型（显示训练集和测试集评估结果）
    y_test_pred = evaluate_model(crf_model, X_test, y_test, X_train, y_train)

    # 6. 演示新样例预测
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
        predict_and_display(crf_model, text, f"样例{i + 1}")

if __name__ == "__main__":
    main()