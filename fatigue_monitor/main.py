"""
作業中の疲労計測アプリ(ローカル実行)

使い方:
    python main.py --user taro              # カメラ+キーボード/マウス
    python main.py --user taro --no-camera  # キーボード/マウスのみ
    python main.py --window-sec 60          # 1分単位で集計(動作確認用)

終了: プレビューウィンドウで q / プレビューなしの場合は Ctrl+C
"""
import argparse
import csv
import math
import os
import time
from datetime import datetime

import config as C
from input_monitor import InputMonitor

COLUMNS = [
    "user", "start", "end", "duration_sec",
    # --- キーボード / マウス ---
    "work_elapsed_min", "typing_count", "typing_per_min",
    "correction_count", "correction_weighted", "correction_ratio", "net_input",
    "max_streak_sec", "key_idle_total_sec", "key_max_idle_sec",
    "scroll_count", "mouse_click_count", "mouse_distance_px", "mouse_active_ratio",
    "window_switch_count",
    # --- カメラ ---
    "face_detected_ratio", "obs_sec", "eye_open_baseline",
    "blink_count", "blink_per_min", "eye_open_mean", "eye_open_rel",
    "fixation_total_sec", "fixation_ratio", "max_fixation_sec", "gaze_shift_per_min",
    "posture_change_count", "posture_max_hold_sec", "face_y_mean", "face_width_mean",
]


def clean(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 4)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="user", help="被験者ID(ファイル名と解析のグルーピングに使う)")
    ap.add_argument("--window-sec", type=int, default=C.WINDOW_SEC)
    ap.add_argument("--camera-index", type=int, default=C.CAMERA_INDEX)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    C.WINDOW_SEC = args.window_sec

    inp = InputMonitor()
    inp.start()
    cam = None
    if not args.no_camera:
        from camera_monitor import CameraMonitor
        cam = CameraMonitor(args.camera_index)
    preview = cam is not None and not args.no_preview
    if preview:
        import cv2

    os.makedirs("data", exist_ok=True)
    path = f"data/{args.user}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
    writer.writeheader()
    print(f"[main] 計測開始 -> {path}  (集計単位 {C.WINDOW_SEC}秒)")

    win_start = time.time()
    win_end = win_start + C.WINDOW_SEC

    def write_window(now):
        row = {
            "user": args.user,
            "start": datetime.fromtimestamp(win_start).isoformat(timespec="seconds"),
            "end": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "duration_sec": now - win_start,
        }
        row.update(inp.snapshot(now))
        if cam:
            row.update(cam.snapshot(now))
        writer.writerow({k: clean(v) for k, v in row.items()})
        f.flush()
        print(f"[main] {row['end']}  typing/min={row['typing_per_min']:.0f}  "
              f"corr={row['correction_ratio']:.2f}  blink/min={clean(row.get('blink_per_min', ''))}")

    try:
        while True:
            if cam:
                frame, lm = cam.step()
                if preview and frame is not None:
                    show = frame.copy()
                    cv2.putText(show, f"{cam.status}  (q: quit)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if lm is not None:
                        hh, ww = show.shape[:2]
                        for i in (33, 133, 159, 145, 362, 263, 386, 374, 468, 473, 10, 152):
                            cv2.circle(show, (int(lm[i].x * ww), int(lm[i].y * hh)), 2, (0, 255, 255), -1)
                    cv2.imshow("fatigue monitor", cv2.resize(show, None, fx=0.5, fy=0.5))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                time.sleep(0.2)
            now = time.time()
            if now >= win_end:
                write_window(now)
                win_start, win_end = now, now + C.WINDOW_SEC
    except KeyboardInterrupt:
        pass
    finally:
        now = time.time()
        if now - win_start >= 60:           # 途中終了でも1分以上あれば最後の区間を保存
            write_window(now)
        f.close()
        inp.stop()
        if cam:
            cam.close()
        if preview:
            cv2.destroyAllWindows()
        print("[main] 終了しました")


if __name__ == "__main__":
    main()
