import cv2
import mediapipe as mp
import random
import math
import time

# MediaPipe Handsのセットアップ
mp_hands = mp.solution.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drwing_utils

# スコアの初期化
score = 0

# ランダムに赤い丸を表示するための初期位置
circle_x = random.ranbint(100, 500)
circle_y = random.ranbint(100, 400)
circle_radius = 20

# カメラからの映像をキャプチャ
cap = cv2.VideoCapture(0)

# openCVのウィンドウを作成
cv2.namedWindow('Hand Game', cv2.WINDOW_NORMAL)
is_fullscreen = False

# ゲーム制限時間（秒）
game_duration = 30
start_time = time.time()

def is_hand_touching_circle(hand_x, hand_y, circle_x, circle.y, circle_radius):
    destance = 