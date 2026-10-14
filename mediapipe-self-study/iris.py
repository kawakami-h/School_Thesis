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

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
start_time = time_base = cv2.getTickCount()

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
            pass

    cv2.imshow("Eye Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()