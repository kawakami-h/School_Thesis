from sklearn.cluster import KMeans
import numpy as np

# 1. サンプルデータ
data = [
    [1.0, 2.0], [1.5, 1.8], [2.0, 2.5], [1.2, 1.5],  # 塊A用
    [8.0, 9.0], [8.5, 8.5], [9.0, 9.5], [8.2, 8.8],  # 塊B用
    [3.0, 1.0], [10.0, 9.0], [6.0, 2.0]
]

X = np.array(data)

# 2. AI(K-meansモデル)の準備
# n_clusters=2 でグループ数を指定
# n_init="auto" は初期化の警告を消す
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")

# 3. 学習（fitするだけで、中のループや中心点計算をやってくれる）
kmeans.fit(X)

# 4. 結果の取得
centroids = kmeans.cluster_centers_ # 求まった中心点
labels = kmeans.labels_             # 各データがどのグループ（0か1）に属するか

# 5. 結果の表示
print("=== 学習完了 (scikit-learn) ===")
for i in range(2):
    belonging_data = X[labels == i].tolist()

    print(f"グループ {i+1} の中心点: {centroids[i].tolist()}")
    print(f"所属するデータ: {belonging_data}")