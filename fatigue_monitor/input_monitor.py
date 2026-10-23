"""
キーボード・マウスの計測(pynput)。
※ 入力された文字の内容は一切記録しない。回数・時刻・種類のみを集計する。
"""
import math
import threading
import time

from pynput import keyboard, mouse

import config as C

CTRL_NAMES = {"ctrl", "ctrl_l", "ctrl_r", "cmd", "cmd_l", "cmd_r"}   # macではCmdもCtrl扱い
SHIFT_NAMES = {"shift", "shift_l", "shift_r"}
ALT_NAMES = {"alt", "alt_l", "alt_r", "alt_gr", "cmd", "cmd_l", "cmd_r"}  # Alt+Tab / Cmd+Tab
MODIFIER_NAMES = CTRL_NAMES | SHIFT_NAMES | ALT_NAMES | {"caps_lock"}


def _ident(key):
    """押下/解放で同じ値になる識別子(Ctrl併用時にcharが変わる問題の回避)"""
    if isinstance(key, keyboard.Key):
        return key.name
    vk = getattr(key, "vk", None)
    return vk if vk is not None else getattr(key, "char", None)


def _letter(key):
    ch = getattr(key, "char", None)
    if not ch or len(ch) != 1:
        return None
    if 1 <= ord(ch) <= 26:          # Windowsでは Ctrl+Z が '\x1a' で届く
        return chr(ord(ch) + 96)
    return ch.lower()


class InputMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._pressed = {}              # ident -> 押下時刻(オートリピート除外用)
        self.session_start = None
        self.last_event_ts = None
        self._last_mouse = None
        self._reset_window(time.time())
        self._kl = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._ml = mouse.Listener(on_move=self._on_move, on_click=self._on_click,
                                  on_scroll=self._on_scroll)

    # ---------- 開始/停止 ----------
    def start(self):
        self._kl.start()
        self._ml.start()

    def stop(self):
        self._kl.stop()
        self._ml.stop()

    # ---------- 内部 ----------
    def _reset_window(self, now):
        self._win_start = now
        self._key_ts = []
        self._corr_count = 0
        self._corr_weighted = 0.0
        self._win_switch = 0
        self._scroll = 0
        self._clicks = 0
        self._mouse_dist = 0.0
        self._mouse_secs = set()

    def _touch(self, now):
        if self.last_event_ts is None or now - self.last_event_ts > C.SESSION_RESET_IDLE_SEC:
            self.session_start = now
        self.last_event_ts = now

    def _held(self, names):
        return any(k in names for k in self._pressed)

    # ---------- キーボード ----------
    def _on_press(self, key):
        now = time.time()
        ident = _ident(key)
        with self._lock:
            t0 = self._pressed.get(ident)
            if t0 is not None and now - t0 < 5.0:
                return                      # オートリピートは数えない
            self._pressed[ident] = now
            if isinstance(key, keyboard.Key) and key.name in MODIFIER_NAMES:
                return                      # 修飾キー単体は入力として数えない

            self._touch(now)
            self._key_ts.append(now)

            ctrl = self._held(CTRL_NAMES)
            shift = self._held(SHIFT_NAMES)
            name = None
            if key == keyboard.Key.backspace:
                name = "ctrl+backspace" if ctrl else "backspace"
            elif key == keyboard.Key.delete:
                name = "delete"
            elif ctrl:
                ch = _letter(key)
                if ch == "z":
                    name = "ctrl+shift+z" if shift else "ctrl+z"
                elif ch == "y":
                    name = "ctrl+y"
                elif ch == "x":
                    name = "ctrl+x"
            if name:
                self._corr_count += 1
                self._corr_weighted += C.CORRECTION_WEIGHTS.get(name, 1.0)

            if key == keyboard.Key.tab and self._held(ALT_NAMES):
                self._win_switch += 1

    def _on_release(self, key):
        with self._lock:
            self._pressed.pop(_ident(key), None)

    # ---------- マウス ----------
    def _on_move(self, x, y):
        now = time.time()
        with self._lock:
            self._touch(now)
            if self._last_mouse is not None:
                self._mouse_dist += math.hypot(x - self._last_mouse[0], y - self._last_mouse[1])
            self._last_mouse = (x, y)
            self._mouse_secs.add(int(now))

    def _on_click(self, x, y, button, pressed):
        if not pressed:
            return
        now = time.time()
        with self._lock:
            self._touch(now)
            self._clicks += 1
            self._mouse_secs.add(int(now))

    def _on_scroll(self, x, y, dx, dy):
        now = time.time()
        with self._lock:
            self._touch(now)
            self._scroll += 1
            self._mouse_secs.add(int(now))

    # ---------- 集計 ----------
    def snapshot(self, now):
        """現在の区間を集計して返し、次の区間のために初期化する"""
        with self._lock:
            start = self._win_start
            dur = max(now - start, 1e-6)
            ts = list(self._key_ts)
            n = len(ts)

            # 連続入力(最長): STREAK_GAP_SEC以内の間隔で続いた入力の最長継続秒数
            max_streak, streak_start, prev = 0.0, None, None
            for t in ts:
                if prev is None or t - prev > C.STREAK_GAP_SEC:
                    streak_start = t
                max_streak = max(max_streak, t - streak_start)
                prev = t

            # 入力しない時間: 区間の端も含めた隙間のうち、KEY_IDLE_GAP_SEC以上のものの合計
            pts = [start] + ts + [now]
            idle_total, max_idle = 0.0, 0.0
            for a, b in zip(pts, pts[1:]):
                gap = b - a
                max_idle = max(max_idle, gap)
                if gap >= C.KEY_IDLE_GAP_SEC:
                    idle_total += gap

            if self.last_event_ts is not None and now - self.last_event_ts <= C.SESSION_RESET_IDLE_SEC:
                work_min = (now - self.session_start) / 60.0
            else:
                work_min = 0.0

            out = {
                "work_elapsed_min": work_min,
                "typing_count": n,
                "typing_per_min": n / (dur / 60.0),
                "correction_count": self._corr_count,
                "correction_weighted": self._corr_weighted,
                "correction_ratio": (self._corr_weighted / n) if n else 0.0,
                "net_input": n - self._corr_count,
                "max_streak_sec": max_streak,
                "key_idle_total_sec": idle_total,
                "key_max_idle_sec": max_idle,
                "scroll_count": self._scroll,
                "mouse_click_count": self._clicks,
                "mouse_distance_px": self._mouse_dist,
                "mouse_active_ratio": len(self._mouse_secs) / dur,
                "window_switch_count": self._win_switch,
            }
            self._reset_window(now)
            return out
