import cv2
import mediapipe as mp
import random
import math
import time
import urllib.request
import os

# ---  最新のMediaPipeモデル（タスクファイル）のダウンロード ---
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("モデルファイルをダウンロード中... (初回のみ)")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("ダウンロード完了！")

# MediaPipe Handsのセットアップ
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# リアルタイム映像処理用設定
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
landmarker = HandLandmarker.create_from_options(options)

# スコアの初期化
score = 0

# ランダムに赤い丸を表示するための初期位置
circle_x = random.randint(100, 500)
circle_y = random.randint(100, 400)
circle_radius = 20

# カメラからの映像をキャプチャ
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# openCVのウィンドウを作成
cv2.namedWindow('Hand Game', cv2.WINDOW_NORMAL)
is_fullscreen = False

# ゲーム制限時間（秒）
game_duration = 30
start_time = time.time()

def is_hand_touching_circle(hand_x, hand_y, circle_x, circle_y, circle_radius):
    distance = math.sqrt((hand_x - circle_x) ** 2 + (hand_y - circle_y) ** 2)
    return distance < circle_radius

def generate_new_circle_position(frame_width, frame_height, circle_radius, exclude_area):
    while True:
        new_x = random.randint(circle_radius, frame_width - circle_radius)
        new_y = random.randint(circle_radius, frame_height - circle_radius)
        if not (exclude_area[0] <= new_x <= exclude_area[2] and exclude_area[1] <= new_y <= exclude_area[3]):
            return new_x, new_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像を水平方向に反転
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 画像をRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 手のランドマークの検出
    frame_timestamp_ms = int ((time.time() - start_time) * 1000)
    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # スコアとタイマー表示領域の背景を描画
    cv2.rectangle(frame, (0, 0), (200, 100), (0, 0, 0), -1)

    # 人差し指の先端（インデックス: 8）の座標を取得して処理
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # 一つずつ間接（ランドマーク）を描画
            for landmark in hand_landmarks:
                lx = int(landmark.x * w)
                ly = int(landmark.y * h)
                cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

            # 人差し指の先端の座標（正規化座標からピクセル座標に変換）
            index_finger_tip = hand_landmarks[8]
            hand_x = int(index_finger_tip.x * w)
            hand_y = int(index_finger_tip.y * h)

            # 当たり判定
            if is_hand_touching_circle(hand_x, hand_y, circle_x, circle_y, circle_radius):
                score += 1
                # 新しい位置に赤い丸を再配置
                circle_x, circle_y = generate_new_circle_position(frame.shape[1], frame.shape[0], circle_radius, (0, 0, 200, 100))

    # 赤い丸を描画
    cv2.circle(frame, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)

    # スコアを表示
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 残り時間を計算
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - int(elapsed_time))
    cv2.putText(frame, f'Time: {remaining_time}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # ゲーム終了
    if remaining_time <= 0:
        break

    # 画像を表示
    cv2.imshow('Hand Game', frame)

    # キー入力をチェック
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty('Hand Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Hand Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# リソースの解放
landmarker.close()
cap.release()
cv2.destroyAllWindows()

# ゲーム終了メッセージ
print(f'Game Over! Your score is {score}')