"""
計測CSV(data/*.csv)をクラスタリングして「通常/疲労っぽい状態」の候補を探す。
    python analyze.py                 # k=2〜5をシルエット係数で自動選択
    python analyze.py --k 3           # クラスタ数を指定
    python analyze.py --no-per-user   # 被験者ごとの標準化をしない
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=None)
ap.add_argument("--no-per-user", action="store_true")
ap.add_argument("--files", default="data/*.csv")
args = ap.parse_args()

files = glob.glob(args.files)
if not files:
    raise SystemExit("data/ にCSVがありません")
df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

# 区間の長さが違っても比較できるよう、回数系は1分あたりにそろえる
m = df["duration_sec"] / 60.0
df["scroll_per_min"] = df["scroll_count"] / m
df["mouse_dist_per_min"] = df["mouse_distance_px"] / m
df["window_switch_per_min"] = df["window_switch_count"] / m
df["idle_ratio"] = df["key_idle_total_sec"] / df["duration_sec"]
# 姿勢の回数は、顔を検出できた秒数あたりにそろえる(旧CSVにはface_secがないのでobs_secで代用)
face_sec = df["face_sec"] if "face_sec" in df.columns else df["obs_sec"]
df["posture_change_per_min"] = df["posture_change_count"] / (face_sec / 60.0)

FEATURES = [
    "typing_per_min", "correction_ratio", "continuous_work_min", "idle_ratio",
    "scroll_per_min", "mouse_dist_per_min", "window_switch_per_min",
    "blink_per_min", "eye_open_rel", "fixation_ratio", "gaze_shift_per_min",
    "posture_change_per_min", "posture_max_hold_sec", "face_y_dev", "face_width_rel",
]
feats = [c for c in FEATURES if c in df.columns and df[c].notna().any()]
df = df.dropna(subset=feats).reset_index(drop=True)
print(f"使用データ: {len(df)}区間 / 特徴量 {len(feats)}個 / 被験者 {df['user'].nunique()}人")
if len(df) < 6:
    raise SystemExit("データが少なすぎます(もっと計測してください)")

X = df[feats].copy()
if not args.no_per_user:                      # 人ごとの基準の違いを除く
    X = X.groupby(df["user"]).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
else:
    X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)

if args.k:
    best_k = args.k
else:
    scores = {}
    for k in range(2, min(5, len(df) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        scores[k] = silhouette_score(X, lab)
    print("シルエット係数:", {k: round(v, 3) for k, v in scores.items()})
    best_k = max(scores, key=scores.get)

df["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(X)
print(f"\nクラスタ数 k={best_k}  (各区間数: {df['cluster'].value_counts().sort_index().to_dict()})")
print("\n各クラスタの特徴(標準化後の平均。+は多い/高い、-は少ない/低い):")
print(X.groupby(df["cluster"]).mean().round(2).T.to_string())

import os; os.makedirs("results", exist_ok=True)
df.to_csv("results/clustered.csv", index=False)
p = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6, 5))
for c in sorted(df["cluster"].unique()):
    plt.scatter(p[df["cluster"] == c, 0], p[df["cluster"] == c, 1], label=f"cluster {c}", alpha=0.7)
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("results/clusters.png", dpi=120)
print("\n保存: results/clustered.csv, results/clusters.png")
