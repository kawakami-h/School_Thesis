import math
import random

# 1. サンプルデータを作成（2次元の座標データ：正解ラベルなし）
# 見た目がわかりやすいように、2つの塊を意識してデータを作成する
data = [
    [1.0, 2.0], [1.5, 1.8], [2.0, 2.5], [1.2, 1.5],  # 塊A用
    [8.0, 9.0], [8.5, 8.5], [9.0, 9.5], [8.2, 8.8],  # 塊B用
    [3.0, 1.0], [10.0, 9.0], [6.0, 2.0]
]

# 2. 初期設定（グループ数を2にする）
k = 2
# ランダムに二つの点を最初の「中心点（セントロイド）」として選ぶ
centroids = random.sample(data, k)

# 3. K-meansのメインループ（学習）
for iteration in range(10):
    # 各データをどの中心に一番近いかでグループ分け
    clusters = [[] for _ in range(k)]

    for point in data:
        # 各中心からの距離を計算して、一番近いものを探す
        distances = []
        for c in centroids:
            # ユークリッド距離の計算（三平方の定理）
            dist = math.sqrt((point[0] - c[0])**2 + (point[1] - c[1])**2)
            distances.append(dist)

        nearest_index = distances.index(min(distances))
        clusters[nearest_index].append(point)

    # 4. グループの新しい中心点を計算（平均をとる）
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            mean_x = sum(p[0] for p in cluster) / len(cluster)
            mean_y = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append([mean_x, mean_y])
        else:
            new_centroids.append(centroids[i])# 空っぽならそのまま

    centroids = new_centroids

# 5. 結果の表示
print("=== 学習完了 ===")
for i, c in enumerate(centroids):
    print(f"グループ {i+1} の中心点* {c}")
    print(f"所属するデータ: {clusters[i]}")