#!/usr/bin/env python3
"""
HMM词性标注完整教学演示
完全兼容新版NLTK (所有概率分布均为LidstoneProbDist对象)
"""

import nltk
from nltk.corpus import brown
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.metrics import accuracy, precision, recall, f_measure
import numpy as np
import random
# ===================== 1. 数据准备 =====================
print("1. 准备数据...")
# 关键：使用 list() 将惰性加载的视图转换为可操作的列表
tagged_sentences = list(brown.tagged_sents(categories='news', tagset='universal'))

# 划分训练集和测试集 (9:1)
random.seed(42)  # 固定随机种子，使结果可复现
random.shuffle(tagged_sentences) #思考，如果如加入随机划分，其性能会如何？
split_idx = int(len(tagged_sentences) * 0.9)
train_data = tagged_sentences[:split_idx]
test_data = tagged_sentences[split_idx:]
print(f"   训练集: {len(train_data)}句, 测试集: {len(test_data)}句")

# ===================== 2. 训练HMM模型 =====================
print("\n2. 训练HMM模型...")
hmm_tagger = HiddenMarkovModelTagger.train(train_data)

# ===================== 3. 评估模型性能 =====================
print("\n3. 评估模型性能...")
test_sentences = [[word for word, _ in sent] for sent in test_data]
gold_tags = [[tag for _, tag in sent] for sent in test_data]
predicted_tags = [hmm_tagger.tag(sent) for sent in test_sentences]

# 扁平化标签用于评估
gold_flat = [tag for sent in gold_tags for tag in sent]
pred_flat = [tag for sent in predicted_tags for _, tag in sent]

# 计算整体准确率
overall_accuracy = accuracy(gold_flat, pred_flat)
print(f"   整体准确率: {overall_accuracy:.4f} ({overall_accuracy:.2%})")

# 计算各标签的精确率、召回率、F1
all_tags = sorted(set(gold_flat + pred_flat))
tag_metrics = {}
for tag in all_tags:
    ref_set = {i for i, t in enumerate(gold_flat) if t == tag}
    test_set = {i for i, t in enumerate(pred_flat) if t == tag}
    p = precision(ref_set, test_set) if test_set else 0.0
    r = recall(ref_set, test_set) if ref_set else 0.0
    f = f_measure(ref_set, test_set) if (p + r) > 0 else 0.0
    tag_metrics[tag] = {'precision': p, 'recall': r, 'f1': f, 'support': len(ref_set)}

# 计算宏平均
macro_p = np.mean([m['precision'] for m in tag_metrics.values()])
macro_r = np.mean([m['recall'] for m in tag_metrics.values()])
macro_f = np.mean([m['f1'] for m in tag_metrics.values()])
print(f"   宏平均: P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f:.4f}")

# ===================== 4. 展示HMM核心参数 (已统一修复) =====================
print("\n4. HMM核心参数示例:")
print("   - 隐藏状态(词性标签):", list(hmm_tagger._states)[:8], "...")
print("   - 观测符号(词汇表大小):", len(hmm_tagger._symbols), "个单词")

# 通用辅助函数：从LidstoneProbDist对象中获取最高概率的k个项
def get_top_probabilities(prob_dist, all_items, top_k=2, threshold=1e-6):
    """从概率分布对象中提取概率最高的top_k个项"""
    probs = [(item, prob_dist.prob(item)) for item in all_items]
    filtered = [(item, p) for item, p in probs if p > threshold]
    sorted_items = sorted(filtered, key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]

# 转移概率示例 (A矩阵)
print("\n   转移概率示例 (A矩阵):")
sample_tags = ['NOUN', 'VERB', 'DET', 'ADJ'][:3]
all_states = list(hmm_tagger._states)

for from_tag in sample_tags:
    if from_tag in hmm_tagger._transitions:
        prob_dist = hmm_tagger._transitions[from_tag]
        top_trans = get_top_probabilities(prob_dist, all_states, top_k=2)
        if top_trans:
            trans_str = "，".join([f"{to}:{prob:.3f}" for to, prob in top_trans])
            print(f"     {from_tag} → {trans_str}")

# 发射概率示例 (B矩阵)
print("\n   发射概率示例 (B矩阵):")
for tag in sample_tags:
    if tag in hmm_tagger._outputs:
        prob_dist = hmm_tagger._outputs[tag]
        # 从词汇表中抽样一部分单词来查询，提高效率
        sample_words = list(hmm_tagger._symbols)[:200]
        top_words = get_top_probabilities(prob_dist, sample_words, top_k=2)
        if top_words:
            words_str = "，".join([f"{w}:{p:.3f}" for w, p in top_words])
            print(f"     {tag} 生成: {words_str}")

# 初始概率示例 (π分布) - 关键修复点
print("\n   初始概率示例 (π分布):")
if hasattr(hmm_tagger, '_priors') and hmm_tagger._priors:
    # _priors现在也是LidstoneProbDist对象，需要使用.prob()方法
    prob_dist = hmm_tagger._priors
    top_states = get_top_probabilities(prob_dist, all_states, top_k=3)
    for state, prob in top_states:
        print(f"     句子以 {state} 开头的概率: {prob:.4f}")
else:
    print("     未找到初始概率分布信息")

# ===================== 5. 各标签性能简表 =====================
print("\n5. 主要词性标签性能 (按支持度排序):")
sorted_tags = sorted(tag_metrics.items(), key=lambda x: x[1]['support'], reverse=True)[:8]
print("   标签    精确率  召回率  F1分数  支持度")
for tag, met in sorted_tags:
    print(f"   {tag:<6}  {met['precision']:.4f}  {met['recall']:.4f}  {met['f1']:.4f}  {met['support']:>6}")

# ===================== 6. 新句子预测演示 =====================
print("\n6. 新句子预测演示:")
examples = [
    "The quick brown fox jumps over the lazy dog .",
    "Natural language processing is an exciting field ."
]
for sent in examples:
    words = sent.split()
    tagged = hmm_tagger.tag(words)
    print(f"\n   '{sent}'")
    result = " ".join([f"{w}/{t}" for w, t in tagged])
    print(f"   结果: {result}")

print("\n" + "="*50)
print(f"演示完成! HMM词性标注器准确率: {overall_accuracy:.2%}")
print("="*50)