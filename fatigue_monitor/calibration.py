"""
全画面5点キャリブレーション。
  1) 準備チェック: 顔の検出・明るさ・距離(顔の大きさ)を確認し、OKが続いたら自動で開始
  2) 5点(中央・左上・右上・右下・左下)を順に目だけで見てもらい、各点のデータを集める
得られるもの: 眼の開き具合の基準、正面を向いたときの向き・顔の位置・顔の大きさ、各点ごとの生の特徴量
(各点の特徴量はmeta.jsonに保存するので、画面位置への対応づけや向きの補正は後から解析で追加できる)
※ 画像は保存しない。数値だけを使う。
"""
import math
import sys
import time

import numpy as np

import config as C

POINTS = [  # (名前, 画面上の位置(横, 縦)の割合)
    ("center", 0.5, 0.5),
    ("top_left", 0.08, 0.10),
    ("top_right", 0.92, 0.10),
    ("bottom_right", 0.92, 0.90),
    ("bottom_left", 0.08, 0.90),
]
NEUTRAL_KEYS = ("yaw", "pitch", "roll", "face_y", "face_w")


class CalibrationSession:
    """キャリブレーションの進行と判定(画面表示から独立。テスト可能)"""

    def __init__(self):
        self.phase = "prepare"        # prepare -> points -> done / failed / aborted
        self.idx = 0
        self.checks = []              # [(メッセージ, OKか)]
        self.ok_since = None
        self.quality_ok = False
        self.t_point = None
        self.fail_reason = ""
        self._buf = {name: [] for name, _, _ in POINTS}
        self._result = None

    # ---------- 操作 ----------
    def abort(self):
        if self.phase in ("prepare", "points"):
            self.phase = "aborted"

    def force_start(self, now):
        """準備チェックがOKにならなくても、そのまま開始する(記録上はquality_ok=False)"""
        if self.phase == "prepare":
            self._start_points(now)

    def _start_points(self, now):
        self.phase, self.idx, self.t_point = "points", 0, now

    # ---------- 1フレームの更新 ----------
    def update(self, ft, stats, now):
        if self.phase == "prepare":
            self._update_prepare(ft, stats or {}, now)
        elif self.phase == "points":
            self._update_points(ft, now)

    def _update_prepare(self, ft, stats, now):
        if ft is None:
            self.checks = [("顔が検出されていません。カメラに顔が写る位置に座ってください", False)]
            self.ok_since = None
            return
        b, fw = stats.get("brightness", float("nan")), stats.get("face_w", 0.0)
        if math.isnan(b):
            c_b, msg_b = False, "顔の明るさを確認できません"
        elif b < C.CALIB_BRIGHTNESS_MIN:
            c_b, msg_b = False, "顔が暗すぎます。部屋を明るくするか、光の当たる向きを変えてください"
        elif b > C.CALIB_BRIGHTNESS_MAX:
            c_b, msg_b = False, "顔が明るすぎます(逆光・強い照明)。光の向きを変えてください"
        else:
            c_b, msg_b = True, "顔の明るさ: 適切"
        c_d = C.CALIB_FACE_WIDTH_MIN <= fw <= C.CALIB_FACE_WIDTH_MAX
        if c_d:
            msg_d = "カメラとの距離: 適切"
        elif fw < C.CALIB_FACE_WIDTH_MIN:
            msg_d = "カメラから遠すぎます。少し近づいてください"
        else:
            msg_d = "カメラに近すぎます。少し離れてください"
        self.checks = [("顔を検出しています", True), (msg_b, c_b), (msg_d, c_d)]
        if c_b and c_d:
            if self.ok_since is None:
                self.ok_since = now
            if now - self.ok_since >= C.CALIB_PREP_OK_SEC:
                self.quality_ok = True
                self._start_points(now)
        else:
            self.ok_since = None

    def _update_points(self, ft, now):
        name = POINTS[self.idx][0]
        el = now - self.t_point
        lo, hi = C.CALIB_POINT_SETTLE_SEC, C.CALIB_POINT_SETTLE_SEC + C.CALIB_POINT_COLLECT_SEC
        if lo <= el < hi and ft is not None:
            self._buf[name].append(ft)
        if el >= hi:
            self.idx += 1
            self.t_point = now
            if self.idx >= len(POINTS):
                self._finalize()

    # ---------- 表示用 ----------
    def target(self, now):
        """今の点: (名前, x割合, y割合, 状態('settle'|'collect'), 進行0-1, 何点目)"""
        if self.phase != "points" or self.idx >= len(POINTS):
            return None
        name, x, y = POINTS[self.idx]
        el = now - self.t_point
        settle = C.CALIB_POINT_SETTLE_SEC
        total = settle + C.CALIB_POINT_COLLECT_SEC
        return name, x, y, ("settle" if el < settle else "collect"), min(el / total, 1.0), self.idx + 1

    # ---------- 結果 ----------
    def _finalize(self):
        center = self._buf["center"]
        if len(center) < C.CALIB_MIN_POINT_FRAMES:
            self.phase = "failed"
            self.fail_reason = "中央の点を見ている間の顔を十分に検出できませんでした"
            return
        med = lambda rows, k: float(np.median([r[k] for r in rows]))
        pooled = [r for rows in self._buf.values() for r in rows]
        points = {}
        for name, x, y in POINTS:
            rows = self._buf[name]
            if not rows:
                continue
            keys = list(rows[0].keys())
            points[name] = {
                "screen_xy": [x, y], "n": len(rows),
                "median": {k: med(rows, k) for k in keys},
                "std": {k: float(np.std([r[k] for r in rows])) for k in keys},
            }
        self._result = {
            "baseline_open": med(pooled, "open"),         # 5点すべてのデータの中央値
            "neutral": {k: med(center, k) for k in NEUTRAL_KEYS},   # 正面(中央の点)の状態
            "points": points,
            "quality_ok": self.quality_ok,
            "n_frames_total": len(pooled),
        }
        self.phase = "done"

    def result(self):
        return self._result


# ======================================================================
# 全画面表示(tkinter)。表示だけを担当する。
# ======================================================================
def _font():
    if sys.platform == "win32":
        return "Yu Gothic UI"
    if sys.platform == "darwin":
        return "Hiragino Sans"
    return "Noto Sans CJK JP"


def run_calibration(cam):
    """全画面で5点キャリブレーションを行い、結果(dict)を返す。中止・失敗・表示不可ならNone"""
    try:
        import tkinter as tk
    except ImportError:
        print("[calibration] tkinterが使えないため、全画面キャリブレーションをスキップします")
        return None
    try:
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)   # 実際のピクセルで描画
            except Exception:
                pass
        root = tk.Tk()
        root.configure(bg="#1e1e1e")
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        canvas = tk.Canvas(root, bg="#1e1e1e", highlightthickness=0, cursor="none")
        canvas.pack(fill="both", expand=True)
        root.focus_force()
    except Exception as e:
        print(f"[calibration] 全画面表示を開始できません: {e!r}")
        return None

    sess = CalibrationSession()
    fnt = _font()
    screen = [root.winfo_screenwidth(), root.winfo_screenheight()]
    root.bind("<Escape>", lambda e: sess.abort())
    root.bind("<Return>", lambda e: sess.force_start(time.time()))

    def draw(now):
        canvas.delete("all")
        W, H = canvas.winfo_width(), canvas.winfo_height()
        if sess.phase == "prepare":
            canvas.create_text(W / 2, H * 0.22, text="カメラの準備", fill="white", font=(fnt, 32, "bold"))
            canvas.create_text(W / 2, H * 0.32, fill="#cccccc", font=(fnt, 18), justify="center",
                               text="いつも作業するときの姿勢で座り、画面の中央を見てください")
            for i, (msg, ok) in enumerate(sess.checks):
                canvas.create_text(W / 2, H * (0.45 + 0.07 * i), text=("✔ " if ok else "✘ ") + msg,
                                   fill="#5ee07a" if ok else "#ff7b72", font=(fnt, 20))
            canvas.create_text(W / 2, H * 0.85, fill="#888888", font=(fnt, 14),
                               text="準備ができると自動で始まります   Enter: すぐに始める   Esc: スキップ")
            canvas.create_oval(W / 2 - 6, H / 2 - 6, W / 2 + 6, H / 2 + 6, fill="#555555", outline="")
        elif sess.phase == "points":
            tg = sess.target(now)
            if tg:
                _, x, y, state, prog, n = tg
                cx, cy = x * W, y * H
                r = 34 - 22 * prog
                col = "#ffffff" if state == "settle" else "#5ee07a"
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=col, width=3)
                canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill="#ff4d4d", outline="")
                canvas.create_text(W / 2, H * 0.06 if y > 0.3 else H * 0.94, fill="#cccccc", font=(fnt, 18),
                                   text=f"画面の点を、頭を動かさず目だけで見てください ({n}/{len(POINTS)})")

    def tick():
        if sess.phase in ("done", "failed", "aborted"):
            root.quit()
            return
        try:
            _, ft, stats = cam.grab()
            now = time.time()
            sess.update(ft, stats, now)
            draw(now)
        except Exception as e:
            print(f"[calibration] エラー: {e!r}")
            sess.phase = "aborted"
        root.after(1, tick)

    root.after(50, tick)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    if sess.phase == "done":
        res = sess.result()
        res["screen_px"] = screen
        return res
    print("[calibration] " + {"aborted": "スキップされました", "failed": sess.fail_reason}.get(sess.phase, "中止"))
    return None
