"""
Webカメラ + MediaPipe Face Landmarker による計測。
※ 映像・画像は保存しない。フレームごとの数値を集計するだけ。
"""
import os
import time
import urllib.request

import numpy as np

import config as C

NAN = float("nan")

# Face Mesh のランドマーク番号(左右は画像上ではなく被写体基準)
LEFT_EYE = dict(c1=33, c2=133, up=159, low=145, iris=468)
RIGHT_EYE = dict(c1=362, c2=263, up=386, low=374, iris=473)
FOREHEAD, CHIN, CHEEK_L, CHEEK_R = 10, 152, 234, 454


def ensure_model(path=C.MODEL_PATH):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[camera] モデルをダウンロードします: {C.MODEL_URL}")
    urllib.request.urlretrieve(C.MODEL_URL, path)
    return path


class CameraMonitor:
    def __init__(self, index=C.CAMERA_INDEX):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.cv2, self.mp = cv2, mp
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ {index} を開けません(権限/接続/他アプリの使用を確認)")

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

        self._t_start = time.time()
        self._last_change_ts = self._t_start
        self._reset_window(self._t_start)

        # 状態(区間をまたいで持ち越すもの)
        self._calib_start = None
        self._calib_vals = []
        self.baseline = None
        self.threshold = None
        self._closed_since = None
        self._anchor = None
        self._dwell_start = None
        self._y_fast = None
        self._y_ref = None
        self._last_face_t = None
        self.status = "no face"

    # ---------- 区間の集計 ----------
    def _reset_window(self, now):
        self._win_start = now
        self._n_frames = 0
        self._face_frames = 0
        self._obs_sec = 0.0
        self._blinks = 0
        self._open_sum, self._open_n = 0.0, 0
        self._fix_sec = 0.0
        self._max_dwell = 0.0
        self._gaze_shifts = 0
        self._y_sum, self._w_sum, self._pose_n = 0.0, 0.0, 0
        self._posture_ts = []
        self._hold_base = self._last_change_ts   # 区間開始時点で最後に姿勢を直した時刻

    # ---------- 幾何 ----------
    @staticmethod
    def _pt(lm, i, w, h):
        return np.array([lm[i].x * w, lm[i].y * h])

    def _eye_features(self, lm, e, w, h):
        c1, c2 = self._pt(lm, e["c1"], w, h), self._pt(lm, e["c2"], w, h)
        up, low = self._pt(lm, e["up"], w, h), self._pt(lm, e["low"], w, h)
        iris = self._pt(lm, e["iris"], w, h)
        width = np.linalg.norm(c1 - c2) + 1e-9
        openness = np.linalg.norm(up - low) / width           # 眼の開き具合
        a, b = (c1, c2) if c1[0] <= c2[0] else (c2, c1)        # 画像上の左端→右端
        gx = float(np.dot(iris - a, b - a) / (np.linalg.norm(b - a) ** 2 + 1e-9)) - 0.5
        gy = float((iris[1] - (up[1] + low[1]) / 2) / width)   # 目の幅を1とした単位
        return openness, gx, gy

    # ---------- 1フレーム分の処理(テスト用にmediapipeから分離) ----------
    def process_landmarks(self, lm, w, h, t):
        dt = 0.0 if self._last_face_t is None else min(t - self._last_face_t, 0.5)
        if self._last_face_t is not None and t - self._last_face_t > 2.0:
            self._y_fast = self._y_ref = None               # 離席などの後は姿勢基準を取り直す
            self._anchor = None
            self._closed_since = None
        self._last_face_t = t
        self._face_frames += 1

        ol, gxl, gyl = self._eye_features(lm, LEFT_EYE, w, h)
        orr, gxr, gyr = self._eye_features(lm, RIGHT_EYE, w, h)
        openness = (ol + orr) / 2

        # --- キャリブレーション: 最初のCALIB_SEC秒の「通常の開き具合」を基準にする ---
        if self.baseline is None:
            if self._calib_start is None:
                self._calib_start = t
            self._calib_vals.append(openness)
            self.status = f"calibrating {t - self._calib_start:.0f}/{C.CALIB_SEC}s"
            if t - self._calib_start >= C.CALIB_SEC:
                self.baseline = float(np.median(self._calib_vals))
                self.threshold = self.baseline * C.BLINK_THRESHOLD_RATIO
                print(f"[camera] キャリブレーション完了: 基準の眼の開き={self.baseline:.3f}")
            return
        self.status = "tracking"
        self._obs_sec += dt

        # --- 瞬き ---
        closed = openness < self.threshold
        if closed:
            if self._closed_since is None:
                self._closed_since = t
        else:
            if self._closed_since is not None:
                if t - self._closed_since <= C.BLINK_MAX_SEC:
                    self._blinks += 1
                self._closed_since = None
            self._open_sum += openness
            self._open_n += 1

        # --- 視線(開眼時のみ) ---
        if not closed:
            g = np.array([(gxl + gxr) / 2, (gyl + gyr) / 2])
            if self._anchor is None or np.linalg.norm(g - self._anchor) > C.GAZE_RADIUS:
                if self._anchor is not None:
                    self._gaze_shifts += 1
                self._anchor, self._dwell_start = g, t
            dwell = t - self._dwell_start
            self._max_dwell = max(self._max_dwell, dwell)
            if dwell >= C.FIXATION_MIN_SEC:
                self._fix_sec += dt

        # --- 姿勢(顔の中心のy座標・顔幅) ---
        y = (lm[FOREHEAD].y + lm[CHIN].y) / 2
        face_w = np.linalg.norm(self._pt(lm, CHEEK_L, w, h) - self._pt(lm, CHEEK_R, w, h)) / w
        self._y_sum += y
        self._w_sum += face_w
        self._pose_n += 1
        self._y_fast = y if self._y_fast is None else 0.3 * y + 0.7 * self._y_fast
        if self._y_ref is None:
            self._y_ref = self._y_fast
        elif (abs(self._y_fast - self._y_ref) > C.POSTURE_MOVE_THR
              and t - self._last_change_ts >= C.POSTURE_COOLDOWN_SEC):
            self._posture_ts.append(t)
            self._last_change_ts = t
            self._y_ref = self._y_fast

    # ---------- フレーム取得 ----------
    def step(self):
        """1フレーム読んで処理する。(frame, landmarks or None)を返す"""
        t_begin = time.time()
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.05)
            return None, None
        t = time.time()
        self._n_frames += 1
        h, w = frame.shape[:2]
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((t - self._t_start) * 1000)
        res = self.landmarker.detect_for_video(img, ts_ms)
        lm = res.face_landmarks[0] if res.face_landmarks else None
        if lm is not None:
            self.process_landmarks(lm, w, h, t)
        else:
            self.status = "no face"
        wait = 1.0 / C.TARGET_FPS - (time.time() - t_begin)
        if wait > 0:
            time.sleep(wait)
        return frame, lm

    # ---------- 集計 ----------
    def snapshot(self, now):
        total = max(self._n_frames, 1)
        face_ratio = self._face_frames / total
        obs = self._obs_sec
        out = {"face_detected_ratio": face_ratio, "obs_sec": obs,
               "eye_open_baseline": self.baseline if self.baseline is not None else NAN}
        valid = face_ratio >= C.MIN_FACE_RATIO and obs > 1.0 and self._pose_n > 0
        if valid:
            mins = obs / 60.0
            # 姿勢維持時間: 姿勢変化の間隔の最長(区間をまたいだ持ち越し込み)
            pts = [self._hold_base] + self._posture_ts + [now]
            max_hold = max(b - a for a, b in zip(pts, pts[1:]))
            mean_open = self._open_sum / self._open_n if self._open_n else NAN
            out.update({
                "blink_count": self._blinks,
                "blink_per_min": self._blinks / mins,
                "eye_open_mean": mean_open,
                "eye_open_rel": mean_open / self.baseline if self.baseline else NAN,
                "fixation_total_sec": self._fix_sec,
                "fixation_ratio": self._fix_sec / obs,
                "max_fixation_sec": self._max_dwell,
                "gaze_shift_per_min": self._gaze_shifts / mins,
                "posture_change_count": len(self._posture_ts),
                "posture_max_hold_sec": max_hold,
                "face_y_mean": self._y_sum / self._pose_n,
                "face_width_mean": self._w_sum / self._pose_n,
            })
        self._reset_window(now)
        return out

    def close(self):
        self.cap.release()
        self.landmarker.close()
