"""
カメラ映像の保存。--save-video または実験モードのときだけ使う。
・実時間に同期するよう、処理が遅れたときはフレームを複製して時間軸を合わせる。
・正常終了(q / Ctrl+C)で release() され、ファイルが完成する。
  プログラムの強制終了・ターミナルを閉じる等の場合は、動画が再生できなくなることがある。
"""
import os
from datetime import datetime

import config as C


class VideoRecorder:
    def __init__(self, path_stem, fps=C.TARGET_FPS, scale=C.VIDEO_SCALE):
        import cv2
        self.cv2 = cv2
        self.path_stem = path_stem
        self.fps = fps
        self.scale = scale
        self.writer = None
        self.path = None
        self.t0 = None
        self.written = 0

    def _open(self, size):
        for fourcc, ext in C.VIDEO_CODECS:
            path = self.path_stem + ext
            w = self.cv2.VideoWriter(path, self.cv2.VideoWriter_fourcc(*fourcc), self.fps, size)
            if w.isOpened():
                self.writer, self.path = w, path
                return
            w.release()
            if os.path.exists(path):
                os.remove(path)
        raise RuntimeError("動画ファイルを作成できません(コーデック非対応)")

    def write(self, frame, t):
        cv2 = self.cv2
        img = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.writer is None:
            h, w = img.shape[:2]
            self._open((w, h))
            self.t0 = t
        if C.VIDEO_TIMESTAMP:
            text = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        target = int((t - self.t0) * self.fps) + 1      # 実時間に対して書くべきフレーム数
        for _ in range(max(target - self.written, 0)):
            self.writer.write(img)
            self.written += 1

    def close(self):
        """終了時に呼ぶ。保存したファイルのパスを返す(何も録画していなければNone)"""
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        return self.path
