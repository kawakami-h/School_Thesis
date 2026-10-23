"""
計測パラメータ。実験しながらここの値を調整する。
(README.mdの「データの細かい手法」に対応)
"""

# ---------- 共通 ----------
WINDOW_SEC = 300          # 集計単位(秒)。5分 = 300。テスト時は --window-sec 60 など

# ---------- キーボード / マウス ----------
STREAK_GAP_SEC = 3.0      # この秒数以内に次の入力があれば「連続入力」とみなす
KEY_IDLE_GAP_SEC = 5.0    # この秒数以上キー入力が無い時間を「入力しない時間」に加算

# 連続作業時間: 区間(WINDOW_SEC)のうち、キーボードまたはマウスを操作した秒の割合がこの値以上なら
# 「作業していた区間」とみなして連続作業時間に加算する。下回る区間があるとリセット(休憩扱い)
WORK_ACTIVE_RATIO = 0.3

# ウィンドウ切り替え(Windows): 前面ウィンドウをこの間隔で監視し、新しいウィンドウが
# SWITCH_CONFIRM_SEC 秒以上前面にあったら「切り替え」と数える
FOREGROUND_POLL_SEC = 0.2
SWITCH_CONFIRM_SEC = 0.5

# 修正入力と重要度(重み)。重要度を変えたい場合はここを編集
CORRECTION_WEIGHTS = {
    "backspace": 1.0,
    "delete": 1.0,
    "ctrl+z": 1.0,
    "ctrl+x": 1.0,
    "ctrl+y": 1.0,
    "ctrl+shift+z": 1.0,
    "ctrl+backspace": 1.0,
}

# ---------- カメラ ----------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280       # 解像度が低いと瞬き/視線の精度が落ちる
CAMERA_HEIGHT = 720
TARGET_FPS = 20           # 処理するフレームレート(瞬き検出のため15以上を推奨)
CALIB_SEC = 20            # 起動直後、目を普通に開けた状態を基準として測る秒数
MIN_FACE_RATIO = 0.5      # 顔検出できたフレームの割合がこれ未満の区間はカメラ指標を空欄にする

# 瞬き: 眼の開き具合が「基準値 × この比率」を下回ったら閉眼
BLINK_THRESHOLD_RATIO = 0.65
BLINK_MAX_SEC = 0.8       # これより長い閉眼は瞬きに数えない(目をつぶっているだけ)

# 視線の一点集中: 虹彩中心が、基準点から(目の幅を1としたとき)この半径内に留まっている状態
GAZE_RADIUS = 0.06
FIXATION_MIN_SEC = 1.0    # この秒数以上留まったら「一点集中」とみなす

# 姿勢: 顔の中心のy座標(画面の高さを1とする)がこれ以上動いたら「姿勢を直した」
POSTURE_MOVE_THR = 0.04
POSTURE_COOLDOWN_SEC = 3.0  # 連続カウント防止

# ---------- MediaPipeモデル ----------
MODEL_PATH = "models/face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# ---------- 映像の保存(--save-video / 実験モードの既定) ----------
VIDEO_SCALE = 0.5            # 保存時の縮小率(0.5 → 1280x720 を 640x360 に。容量を抑える)
VIDEO_TIMESTAMP = True       # 映像に日時を焼き込む(CSVの区間と突き合わせるため)
VIDEO_CODECS = [("mp4v", ".mp4"), ("MJPG", ".avi")]   # 先頭から順に試す
