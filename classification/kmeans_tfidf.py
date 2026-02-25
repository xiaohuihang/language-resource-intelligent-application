import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from collections import Counter

def load_and_merge_data(pos_file='./实验数据/pos.txt', neg_file='./实验数据/neg.txt'):
    all_texts, labels = [], []
    with open(pos_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text: all_texts.append(text); labels.append('pos')
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text: all_texts.append(text); labels.append('neg')
    print(f"数据加载完成: {len(all_texts)}条样本 (正面:{labels.count('pos')}, 负面:{labels.count('neg')})")
    return all_texts, labels

def extract_tfidf_features(texts):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)), max_features=4000, min_df=2, max_df=0.8,
                                 norm=None)
    features = vectorizer.fit_transform(texts)
    features_norm = normalize(features, norm='l2')
    print(f"TF-IDF特征提取完成: {features_norm.shape}")
    return features_norm, vectorizer

def find_optimal_clusters(features, max_k=6):
    silhouette_scores, k_values = [], range(2, max_k + 1)
    print("寻找最佳聚类数量...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        score = silhouette_score(features, kmeans.fit_predict(features))
        silhouette_scores.append(score);
        print(f"K={k}: 轮廓系数={score:.4f}")

    plt.figure(figsize=(8, 4));
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('聚类数(K)');
    plt.ylabel('轮廓系数');
    plt.title('最佳K值选择');
    plt.grid(True);
    plt.show()

    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"最佳聚类数量: K={best_k}")
    return best_k

def perform_clustering(features, n_clusters):
    print(f"\n使用K={n_clusters}进行KMeans聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300, init='k-means++')
    cluster_labels = kmeans.fit_predict(features)
    inertia, silhouette = kmeans.inertia_, silhouette_score(features, cluster_labels)
    print(f"聚类完成 - 惯性值: {inertia:.2f}, 平均轮廓系数: {silhouette:.4f}")

    for cluster_id, count in Counter(cluster_labels).items():
        print(f"簇{cluster_id}: {count}个样本")
    return kmeans, cluster_labels

def analyze_clusters(texts, labels, cluster_labels, n_clusters):
    print("\n聚类分析结果:")
    for cluster_id in range(n_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_labels_list = [labels[i] for i in indices]
        pos_count, neg_count = cluster_labels_list.count('pos'), cluster_labels_list.count('neg')
        print(
            f"\n簇{cluster_id} ({len(indices)}个样本): 正面{pos_count}({pos_count / len(indices) * 100:.1f}%), 负面{neg_count}({neg_count / len(indices) * 100:.1f}%)")
        for i in indices[:2]:  # 显示2个示例
            print(f"  示例: {texts[i][:30]}...")

def recommend_within_cluster(kmeans, vectorizer, texts, cluster_labels, query_text, top_n=3):
    print(f"\n查询: {query_text}")
    query_tokens = list(jieba.cut(query_text))
    print(f"分词: {'/'.join(query_tokens)}")

    query_features = normalize(vectorizer.transform([query_text]), norm='l2')
    query_cluster = kmeans.predict(query_features)[0]
    print(f"所属簇: {query_cluster}")

    cluster_indices = np.where(cluster_labels == query_cluster)[0]
    cluster_features = normalize(vectorizer.transform([texts[i] for i in cluster_indices]), norm='l2')
    similarities = cosine_similarity(query_features, cluster_features).flatten()

    top_indices = similarities.argsort()[-(top_n + 1):][::-1]
    print(f"推荐同一簇内最相似的{top_n}个文本:")
    for i, idx in enumerate(top_indices):
        if i == 0 and similarities[idx] > 0.999: continue
        if i > top_n: break
        text_idx = cluster_indices[idx]
        preview = texts[text_idx][:50] + "..." if len(texts[text_idx]) > 50 else texts[text_idx]
        print(f"  {i}. 相似度{similarities[idx]:.4f}: {preview}")

def main():
    print("=" * 50 + "\n中文文本聚类与推荐系统\n" + "=" * 50)

    # 1. 加载数据
    texts, labels = load_and_merge_data()

    # 2. 提取特征
    features, vectorizer = extract_tfidf_features(texts)

    # 3. 寻找最佳聚类数
    optimal_k = find_optimal_clusters(features)

    # 4. 执行聚类
    kmeans, cluster_labels = perform_clustering(features, optimal_k)

    # 5. 分析结果
    analyze_clusters(texts, labels, cluster_labels, optimal_k)

    # 6. 推荐演示
    print("\n" + "=" * 50 + "\n簇内文本推荐演示\n" + "=" * 50)
    test_samples = [
        "这部电影真的太棒了，演员演技出色，剧情扣人心弦！",
        "非常失望，完全浪费了我的时间和金钱。",
        "产品一般般，没什么特别的感觉。",
        "服务态度很好，下次还会再来。"
    ]

    for i, sample in enumerate(test_samples, 1):
        print(f"\n--- 测试样例{i} ---")
        recommend_within_cluster(kmeans, vectorizer, texts, cluster_labels, sample)

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    main()