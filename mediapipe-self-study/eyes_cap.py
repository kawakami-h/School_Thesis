import os
import urllib.request
import cv2
import mediapipe as mp

# 最新のFace Landmarkerモデルをダウンロード
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Face Landmarker モデルファイルをダウンロード中...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("ダウンロード完了")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
)

landmarker = FaceLandmarker.create_from_options(options)

# 利用可能なカメラを検出し、選択させる関数
def select_camera():
    available_cameras = []
    print("利用可能なカメラを検出中...")
    for i in range(5):
        cap_test = cv2.VidioCapture(i, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(i)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            if ret:
                available_cameras.append(i)
            cap_test.release()

        if not available_cameras:
            print("利用可能なカメラが見つかりませんでした。デフォルト0番を使用します。")
            return 0

        print("\n---利用可能なカメラ一覧---")
        for idx in available_cameras:
            print(f"[{idx}] カメラ {idx}")
        print("-" * 28)

        while True:
            try:
                choice = int(input("使用するカメラの番号を入力してください："))
                if choice in available_cameras:
                    return choice
                print("無効な番号です。一覧にある番号を入力してください。")
            except ValueError:
                ("半角の数字で入力して下さい。")

# カメラの選択と初期化
camera_id = select_camera()
if os.name == 'nt':
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOw)
else:
    cap = cv2.VideoCapture(camera_id)

start_time = time_base = cv2.getTickCount()

Left_Upper_eyelid = [246, 161, 160, 159, 158, 157, 173]
Left_Upper_center = [159]
Left_Lower_eyelid = [33, 7, 163, 144, 145, 153, 154, 155, 133]
Left_Lower_center = [145]
Left_eyelid = Left_Upper_eyelid + Left_Lower_eyelid

Right_Upper_eyelid = [466, 388, 387, 386, 385, 384, 398]
Right_Upper_center = [386]
Right_Lower_eyelid = [263, 249, 390, 373, 374, 380, 381, 382, 362]
Right_Lower_center = [374]
Right_eyelid = Right_Upper_eyelid + Right_Lower_eyelid

Left_lris = [468, 469, 470, 471, 472]
Right_lris = [473, 474, 475, 476, 477]

Left_eyes = Left_eyelid + Left_lris
Right_eyes = Right_eyelid + Right_lris

Left_important = Left_Upper_center + Left_Lower_center + Left_lris
Right_important = Right_Upper_center + Right_Lower_center + Right_lris

eyes = Left_eyes + Right_eyes
eyes_important = Left_important + Right_important

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _= frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # タイムスタンプ
    timestamp_ms = int(
        (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
    )
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            # 478個のランドマークを取得
            # 取得したランドマークから目やに虹彩の座標を描画する
            # for landmark in face_landmarks:
            #     x = int(landmark.x * w)
            #     y = int(landmark.y * h)
            #     # 全てのランドマークを描画する場合：
            #    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            for idx in eyes:
                landmark = face_landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Eye Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()