# 疲労計測アプリ(ローカル版)

PCの入力(キーボード/マウス)とWebカメラ(MediaPipe Face Landmarker)から、卒論READMEの指標を
**5分単位**で集計してCSVに保存する。解析は `analyze.py`(KMeansクラスタリング)。

## 1. セットアップ
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
- Python 3.10〜3.12 を推奨。
- 初回起動時に顔ランドマークのモデル(約4MB)を `models/` へ自動ダウンロードする。
  失敗したら `config.py` の `MODEL_URL` を手動で保存し `models/face_landmarker.task` に置く。
- **macOS**: 「システム設定 > プライバシーとセキュリティ」で、使うターミナルに
  *入力監視*・*アクセシビリティ*・*カメラ* を許可する。
- **Linux**: X11のみ対応(Wayland上ではキーボード/マウスの全体監視ができない)。

## 2. 実行
```bash
python main.py --user taro                 # カメラ + キーボード/マウス
python main.py --user taro --no-camera     # キーボード/マウスのみ
python main.py --window-sec 60             # 1分単位(動作確認用)
python main.py --no-preview                # プレビュー窓なし
```
- 起動後の最初の **20秒** は「普通に目を開けて画面を見る」キャリブレーション(眼の開き具合の基準値を測る)。
- カメラは画面上部、顔が正面から写る位置に置く。プレビュー窓の黄色い点が目・顔に乗っているか確認。
- 終了はプレビュー窓で `q`、またはターミナルで `Ctrl+C`。途中終了でも1分以上たまっていれば最後の区間を保存する。
- 出力: `data/<user>_<日時>.csv`(1行 = 1区間)

## 3. 解析
```bash
python analyze.py              # k=2〜5をシルエット係数で選ぶ
python analyze.py --k 3
python analyze.py --no-per-user
```
被験者ごとに標準化 → KMeans → 各クラスタの特徴を表示し、`results/` にラベル付きCSVとPCA散布図を保存。

## 4. プライバシー(被験者への説明用)
- 入力した**文字の内容は記録しない**(回数・種類・時刻のみ)。
- **映像・画像は保存しない**(フレームごとの数値を集計するだけ)。
- 保存されるのは `data/*.csv` の集計値だけ。

## 5. 出力列と定義
| 列 | 内容 |
|---|---|
| `work_elapsed_min` | 作業時間。キーボード/マウスが `SESSION_RESET_IDLE_SEC`(10分)以上無操作でリセット |
| `typing_count`, `typing_per_min` | キー入力回数(修飾キー単体・長押しのオートリピートは除く) |
| `correction_*` | 修正入力(Backspace, Delete, Ctrl+Z/X/Y, Ctrl+Shift+Z, Ctrl+Backspace)。重みは `config.py` で変更。`correction_ratio` = 重み付き修正数 / 入力回数 |
| `net_input` | 入力回数 − 修正回数(作業量の粗い目安) |
| `max_streak_sec` | 連続入力時間: 間隔が `STREAK_GAP_SEC`(3秒)以内で続いた入力の最長継続秒数 |
| `key_idle_total_sec`, `key_max_idle_sec` | 入力しない時間(5秒以上の空白の合計 / 最長の空白) |
| `scroll_count` | スクロールイベント数 |
| `mouse_*` | クリック数、移動距離(px)、マウスを動かしていた秒の割合 |
| `window_switch_count` | Alt+Tab / Cmd+Tab の回数(タスクバーのクリック等は数えない) |
| `face_detected_ratio` | 顔を検出できたフレームの割合。`MIN_FACE_RATIO` 未満の区間はカメラ指標が空欄 |
| `blink_count`, `blink_per_min` | 瞬き回数(開眼度が基準値×0.65を下回り、0.8秒以内に戻ったもの) |
| `eye_open_mean`, `eye_open_rel` | 眼の開き具合(上下まぶたの距離÷目の幅、瞬き中を除く平均)と、キャリブレーション基準に対する比 |
| `fixation_ratio`, `max_fixation_sec` | 視線の一点集中: 虹彩中心が基準点から半径内に1秒以上留まった時間の割合 / 最長滞在 |
| `gaze_shift_per_min` | 視線が基準点から半径外に動いた回数(1分あたり) |
| `posture_change_count` | 顔の中心y座標が基準から4%以上動いた回数(3秒のクールダウン付き) |
| `posture_max_hold_sec` | 姿勢を直さずに維持した最長時間 |
| `face_y_mean`, `face_width_mean` | 顔の高さ位置・顔の大きさ(猫背や画面への近さの目安) |

## 6. 解釈を置いた点・既知の限界
- READMEの「連続入力時間」(規定値に達したら5分追加)は意図が読み切れなかったため、上記の「最長連続入力秒数」で実装した。変更したい場合は `input_monitor.py` の `snapshot` を直す。
- 顔の向きは未実装(READMEどおり)。
- 視線は目の幅を1とした虹彩の相対位置で見ているだけで、画面上の注視点は求めていない。
- キャリブレーション中(最初の20秒)は瞬き等を数えない。比率系の指標は実際に観測できた秒数で割っている(`obs_sec`)。
- 閾値(`config.py`)はすべて初期値。実測しながら調整する。
- `analyze.py` では `work_elapsed_min` が時間経過だけでクラスタを分けてしまうことがある。結果が「前半/後半」になるなら特徴量から外して試す。
- クラスタリングは正解ラベルがないため、作業後の自己申告(疲労度1〜5など)を別に記録しておくと、結果の妥当性を後から検証しやすい。
