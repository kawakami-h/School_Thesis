"""
Webカメラ + MediaPipe Face Landmarker による計測。
※ 映像・画像はここでは保存しない(映像保存は video_recorder.py、--save-video / 実験モードのみ)。
"""
import math
import os
import time
import urllib.request

import numpy as np

import config as C

NAN = float("nan")

# Face Mesh のランドマーク番号(左右は被写体基準)
LEFT_EYE = dict(c1=33, c2=133, up=159, low=145, iris=468)
RIGHT_EYE = dict(c1=362, c2=263, up=386, low=374, iris=473)
FOREHEAD, CHIN, CHEEK_L, CHEEK_R = 10, 152, 234, 454
EYE_OUTER_L, EYE_OUTER_R = 33, 263


def ensure_model(path=C.MODEL_PATH):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[camera] モデルをダウンロードします: {C.MODEL_URL}")
    urllib.request.urlretrieve(C.MODEL_URL, path)
    return path


def head_pose(lm, w, h):
    """顔の向き(度)をランドマークの3D座標から求める。符号は目安で、基準(正面)との差として使う。
    yaw   : 左右。右頬(画像右)が奥にあると正
    pitch : 上下。額が顎より奥(顔が上を向く)と正
    roll  : 傾き。画像右の目尻が下がると正
    ※ ランドマークのzは、xと同じ尺度(画像幅に対する比)。"""
    def p(i):
        return np.array([lm[i].x * w, lm[i].y * h, lm[i].z * w])
    cl, cr = p(CHEEK_L), p(CHEEK_R)
    top, chin = p(FOREHEAD), p(CHIN)
    e1, e2 = p(EYE_OUTER_L), p(EYE_OUTER_R)
    yaw = math.degrees(math.atan2(cr[2] - cl[2], cr[0] - cl[0]))
    pitch = math.degrees(math.atan2(top[2] - chin[2], chin[1] - top[1]))
    roll = math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))
    return yaw, pitch, roll


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
        self._last_ts = -1
        self._init_state(time.time())

    # ---------- 状態 ----------
    def _init_state(self, t0):
        self._t_start = t0
        self._last_change_ts = t0
        self._reset_window(t0)
        # 区間をまたいで持ち越す状態
        self.baseline = None            # 眼の開き具合の基準(キャリブレーションで決まる)
        self.threshold = None
        self.neutral = None             # 正面を向いたときの向き・顔の位置・顔の大きさ
        self.calibration_source = None  # "screen"(5点) / "auto"(起動直後の自動)
        self._calib_start = None
        self._calib_rows = []
        self._closed_since = None
        self._anchor = None
        self._dwell_start = None
        self._y_fast = None
        self._y_ref = None
        self._last_face_t = None
        self.status = "no face"

    def _reset_window(self, now):
        self._win_start = now
        self._n_frames = 0
        self._face_frames = 0
        self._cal_frames = 0           # キャリブレーション後に顔を検出したフレーム数
        self._face_sec = 0.0           # 顔を検出できていた秒数
        self._obs_sec = 0.0            # うち、顔の向きが有効だった秒数(瞬き・視線の観測時間)
        self._blinks = 0
        self._open_sum, self._open_n = 0.0, 0
        self._fix_sec = 0.0
        self._max_dwell = 0.0
        self._gaze_shifts = 0
        self._y_sum, self._w_sum = 0.0, 0.0
        self._pose_sum = np.zeros(3)   # 基準からの向きの差の合計(yaw, pitch, roll)
        self._posture_ts = []
        self._hold_base = self._last_change_ts

    # ---------- キャリブレーション ----------
    def set_calibration(self, cal):
        """全画面5点キャリブレーションの結果を設定する"""
        self.baseline = float(cal["baseline_open"])
        self.threshold = self.baseline * C.BLINK_THRESHOLD_RATIO
        self.neutral = dict(cal["neutral"])
        self.calibration_source = "screen"
        self.status = "tracking"

    # ---------- 幾何 ----------
    @staticmethod
    def _pt(lm, i, w, h):
        return np.array([lm[i].x * w, lm[i].y * h])

    def _eye_features(self, lm, e, w, h):
        c1, c2 = self._pt(lm, e["c1"], w, h), self._pt(lm, e["c2"], w, h)
        up, low = self._pt(lm, e["up"], w, h), self._pt(lm, e["low"], w, h)
        iris = self._pt(lm, e["iris"], w, h)
        width = np.linalg.norm(c1 - c2) + 1e-9
        openness = np.linalg.norm(up - low) / width           # 眼の開き具合(顔の大きさに依存しない比)
        a, b = (c1, c2) if c1[0] <= c2[0] else (c2, c1)        # 画像上の左端→右端
        gx = float(np.dot(iris - a, b - a) / (np.linalg.norm(b - a) ** 2 + 1e-9)) - 0.5
        gy = float((iris[1] - (up[1] + low[1]) / 2) / width)   # 目の幅を1とした単位
        return openness, gx, gy

    def frame_features(self, lm, w, h):
        """1フレームの特徴量(キャリブレーションと通常計測で共通)"""
        ol, gxl, gyl = self._eye_features(lm, LEFT_EYE, w, h)
        orr, gxr, gyr = self._eye_features(lm, RIGHT_EYE, w, h)
        yaw, pitch, roll = head_pose(lm, w, h)
        return {
            "open": (ol + orr) / 2, "open_l": ol, "open_r": orr,
            "gx": (gxl + gxr) / 2, "gy": (gyl + gyr) / 2,
            "gxl": gxl, "gyl": gyl, "gxr": gxr, "gyr": gyr,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "face_y": (lm[FOREHEAD].y + lm[CHIN].y) / 2,
            "face_w": float(np.linalg.norm(self._pt(lm, CHEEK_L, w, h) - self._pt(lm, CHEEK_R, w, h)) / w),
        }

    # ---------- 1フレーム分の処理(テスト用にmediapipeから分離) ----------
    def process_landmarks(self, lm, w, h, t):
        ft = self.frame_features(lm, w, h)
        dt = 0.0 if self._last_face_t is None else min(t - self._last_face_t, 0.5)
        if self._last_face_t is not None and t - self._last_face_t > 2.0:
            self._y_fast = self._y_ref = None               # 離席などの後は姿勢基準を取り直す
            self._anchor = None
            self._closed_since = None
        self._last_face_t = t
        self._face_frames += 1

        # --- 5点キャリブレーションを使わない場合: 起動直後のCALIB_SEC秒から自動で基準を決める ---
        if self.baseline is None:
            if self._calib_start is None:
                self._calib_start = t
            self._calib_rows.append(ft)
            self.status = f"calibrating {t - self._calib_start:.0f}/{C.CALIB_SEC}s"
            if t - self._calib_start >= C.CALIB_SEC:
                med = lambda k: float(np.median([r[k] for r in self._calib_rows]))
                self.baseline = med("open")
                self.threshold = self.baseline * C.BLINK_THRESHOLD_RATIO
                self.neutral = {k: med(k) for k in ("yaw", "pitch", "roll", "face_y", "face_w")}
                self.calibration_source = "auto"
                print(f"[camera] 自動キャリブレーション完了: 基準の眼の開き={self.baseline:.3f}")
            return
        self.status = "tracking"
        self._cal_frames += 1
        self._face_sec += dt

        # --- 顔の向き: 基準(正面)からの差 ---
        d = np.array([ft["yaw"] - self.neutral["yaw"],
                      ft["pitch"] - self.neutral["pitch"],
                      ft["roll"] - self.neutral["roll"]])
        self._pose_sum += d
        pose_ok = (abs(d[0]) <= C.YAW_MAX_DEG and abs(d[1]) <= C.PITCH_MAX_DEG
                   and abs(d[2]) <= C.ROLL_MAX_DEG)

        # --- 姿勢(顔の中心のy座標・顔幅)。向きに関係なく使う ---
        y, face_w = ft["face_y"], ft["face_w"]
        self._y_sum += y
        self._w_sum += face_w
        self._y_fast = y if self._y_fast is None else 0.3 * y + 0.7 * self._y_fast
        if self._y_ref is None:
            self._y_ref = self._y_fast
        elif (abs(self._y_fast - self._y_ref) > C.POSTURE_MOVE_THR
              and t - self._last_change_ts >= C.POSTURE_COOLDOWN_SEC):
            self._posture_ts.append(t)
            self._last_change_ts = t
            self._y_ref = self._y_fast

        # --- 向きが大きく外れたフレームは、瞬き・眼の開き・視線に使わない ---
        if not pose_ok:
            self._closed_since = None
            self._anchor = None
            return
        self._obs_sec += dt

        # --- 瞬き ---
        openness = ft["open"]
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
            g = np.array([ft["gx"], ft["gy"]])
            if self._anchor is None or np.linalg.norm(g - self._anchor) > C.GAZE_RADIUS:
                if self._anchor is not None:
                    self._gaze_shifts += 1
                self._anchor, self._dwell_start = g, t
            dwell = t - self._dwell_start
            self._max_dwell = max(self._max_dwell, dwell)
            if dwell >= C.FIXATION_MIN_SEC:
                self._fix_sec += dt

    # ---------- フレーム取得 ----------
    def _detect(self, frame, t):
        h, w = frame.shape[:2]
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int((t - self._t_start) * 1000), self._last_ts + 1)   # 単調増加が必須
        self._last_ts = ts_ms
        res = self.landmarker.detect_for_video(img, ts_ms)
        return res.face_landmarks[0] if res.face_landmarks else None

    def grab(self):
        """キャリブレーション用: 1フレーム読み、(frame, 特徴量 or None, 画質情報) を返す"""
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.02)
            return None, None, {}
        h, w = frame.shape[:2]
        lm = self._detect(frame, time.time())
        if lm is None:
            return frame, None, {}
        ft = self.frame_features(lm, w, h)
        xs = [lm[CHEEK_L].x, lm[CHEEK_R].x]
        ys = [lm[FOREHEAD].y, lm[CHIN].y]
        x0, x1 = int(max(min(xs), 0) * w), int(min(max(xs), 1) * w)
        y0, y1 = int(max(min(ys), 0) * h), int(min(max(ys), 1) * h)
        crop = frame[y0:y1, x0:x1]
        bright = float(self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY).mean()) if crop.size else NAN
        return frame, ft, {"brightness": bright, "face_w": ft["face_w"]}

    def step(self):
        """通常計測: 1フレーム読んで処理する。(frame, landmarks or None)を返す"""
        t_begin = time.time()
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.05)
            return None, None
        t = time.time()
        self._n_frames += 1
        h, w = frame.shape[:2]
        lm = self._detect(frame, t)
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
        out = {"face_detected_ratio": face_ratio, "face_sec": self._face_sec,
               "obs_sec": self._obs_sec,
               "eye_open_baseline": self.baseline if self.baseline is not None else NAN}
        face_ok = face_ratio >= C.MIN_FACE_RATIO and self._cal_frames > 0 and self._face_sec > 1.0
        if face_ok:
            n = self._cal_frames
            # 姿勢維持時間: 姿勢変化の間隔の最長(区間をまたいだ持ち越し込み)
            pts = [self._hold_base] + self._posture_ts + [now]
            max_hold = max(b - a for a, b in zip(pts, pts[1:]))
            y_mean, w_mean = self._y_sum / n, self._w_sum / n
            dy, dp, dr = self._pose_sum / n
            out.update({
                "pose_valid_ratio": self._obs_sec / self._face_sec,
                "head_yaw_mean": dy, "head_pitch_mean": dp, "head_roll_mean": dr,
                "posture_change_count": len(self._posture_ts),
                "posture_max_hold_sec": max_hold,
                "face_y_mean": y_mean, "face_width_mean": w_mean,
                "face_y_dev": y_mean - self.neutral["face_y"],
                "face_width_rel": w_mean / self.neutral["face_w"],
            })
        if face_ok and self._obs_sec >= C.MIN_EYE_OBS_SEC:
            mins = self._obs_sec / 60.0
            mean_open = self._open_sum / self._open_n if self._open_n else NAN
            out.update({
                "blink_count": self._blinks,
                "blink_per_min": self._blinks / mins,
                "eye_open_mean": mean_open,
                "eye_open_rel": mean_open / self.baseline if self.baseline else NAN,
                "fixation_total_sec": self._fix_sec,
                "fixation_ratio": self._fix_sec / self._obs_sec,
                "max_fixation_sec": self._max_dwell,
                "gaze_shift_per_min": self._gaze_shifts / mins,
            })
        self._reset_window(now)
        return out

    def close(self):
        self.cap.release()
        self.landmarker.close()
