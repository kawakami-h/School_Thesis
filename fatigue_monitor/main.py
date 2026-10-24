"""
作業中の疲労計測アプリ(ローカル実行)

引数なしで起動すると、モードと計測対象を対話で選べる。

    python main.py                                   # 対話で選ぶ
    python main.py --mode personal --sensors both    # 個人用・全部計測
    python main.py --mode experiment --user P01      # 実験用(同意確認・疲労度の自己申告・映像保存)
    python main.py --sensors input                   # キーボード/マウスのみ
    python main.py --sensors camera --save-video     # カメラのみ + 映像保存
    python main.py --window-sec 60                   # 1分単位で集計(動作確認用)

終了: プレビュー窓で q / プレビューなしの場合は Ctrl+C
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import config as C

COLUMNS = [
    "user", "mode", "start", "end", "duration_sec",
    # --- キーボード / マウス ---
    "continuous_work_min", "active_ratio", "typing_count", "typing_per_min",
    "correction_count", "correction_weighted", "correction_ratio", "net_input",
    "max_streak_sec", "key_idle_total_sec", "key_max_idle_sec",
    "scroll_count", "mouse_click_count", "mouse_distance_px", "mouse_active_ratio",
    "window_switch_count",
    # --- カメラ ---
    "face_detected_ratio", "face_sec", "obs_sec", "pose_valid_ratio", "eye_open_baseline",
    "blink_count", "blink_per_min", "eye_open_mean", "eye_open_rel",
    "fixation_total_sec", "fixation_ratio", "max_fixation_sec", "gaze_shift_per_min",
    "head_yaw_mean", "head_pitch_mean", "head_roll_mean",
    "posture_change_count", "posture_max_hold_sec",
    "face_y_mean", "face_width_mean", "face_y_dev", "face_width_rel",
]


def clean(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 4)
    return v


# ---------- 対話入力 ----------
def ask_choice(title, options, default=1):
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        s = input(f"番号を入力 [既定 {default}]: ").strip()
        if not s:
            return default
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s)
        print("  → 選択肢の番号を入力してください")


def ask_yes_no(msg, default=False):
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{msg} [{d}]: ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes", "はい", "h"):
            return True
        if s in ("n", "no", "いいえ"):
            return False


def ask_rating(msg):
    while True:
        s = input(f"{msg} (1:全く疲れていない 〜 5:非常に疲れている): ").strip()
        if s.isdigit() and 1 <= int(s) <= 5:
            return int(s)
        print("  → 1〜5 の数字で入力してください")


def safe_name(s):
    return re.sub(r"[^\w\-]", "_", s)[:40] or "user"


def config_snapshot():
    return {k: v for k, v in vars(C).items()
            if k.isupper() and isinstance(v, (int, float, str, bool, list, dict, tuple))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["personal", "experiment"], help="個人用 / 実験用")
    ap.add_argument("--sensors", choices=["both", "camera", "input"], help="計測対象")
    ap.add_argument("--user", default=None, help="被験者ID(ファイル名と解析のグルーピングに使う)")
    ap.add_argument("--window-sec", type=int, default=C.WINDOW_SEC)
    ap.add_argument("--camera-index", type=int, default=C.CAMERA_INDEX)
    ap.add_argument("--save-video", action="store_true", help="カメラ映像を保存する")
    ap.add_argument("--no-video", action="store_true", help="(実験モードでも)映像を保存しない")
    ap.add_argument("--no-camera", action="store_true", help="--sensors input と同じ(旧オプション)")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-calibration", action="store_true",
                    help="全画面キャリブレーションを行わない(起動直後の20秒で自動的に基準を決める)")
    args = ap.parse_args()
    C.WINDOW_SEC = args.window_sec
    if args.no_camera:
        args.sensors = "input"

    # ---------- モード/計測対象の選択 ----------
    if sys.stdin.isatty():
        if args.mode is None:
            args.mode = ["personal", "experiment"][ask_choice(
                "モードを選んでください",
                ["個人用(personal): 自分の計測・動作確認", "実験用(experiment): 被験者ID・同意確認・疲労度の自己申告を記録"])-1]
        if args.sensors is None:
            args.sensors = ["both", "camera", "input"][ask_choice(
                "計測対象を選んでください",
                ["キーボード・マウス + カメラ", "カメラのみ", "キーボード・マウスのみ"])-1]
    mode = args.mode or "personal"
    sensors = args.sensors or "both"
    use_input = sensors in ("both", "input")
    use_cam = sensors in ("both", "camera")

    # ---------- 映像保存の決定 ----------
    if not use_cam or args.no_video:
        save_video = False
    elif args.save_video or mode == "experiment":
        save_video = True
    elif sys.stdin.isatty():
        save_video = ask_yes_no("\nカメラ映像(顔が写る動画)を保存しますか?", default=False)
    else:
        save_video = False

    # ---------- 実験モード: 被験者ID・同意・開始前の疲労度 ----------
    user = args.user
    meta = {}
    if mode == "experiment":
        while not user:
            user = input("\n被験者ID: ").strip()
        print("\n=== 実験の記録内容(被験者に説明してください) ===")
        if use_input:
            print(" ・キーボード/マウス: 回数・時刻・操作の種類のみ。入力した文字の内容は記録しません。")
            print(" ・前面ウィンドウの切り替え回数(Windows)。ウィンドウのタイトルは記録しません。")
        if use_cam:
            print(" ・カメラ: 顔の特徴点から計算した数値(瞬き・視線・姿勢など)を5分ごとに集計して記録します。")
        if save_video:
            print(" ・カメラ映像(顔が写る動画)をこのPCの data/ に保存します。")
        else:
            print(" ・カメラ映像は保存しません。")
        print(" ・開始前と終了後に疲労度(1〜5)をおたずねします。")
        if not ask_yes_no("上記について被験者の同意を得ていますか?", default=False):
            print("同意が得られていないため、計測を開始しません。")
            return
        meta["fatigue_before"] = ask_rating("\n【開始前】現在の疲労度")
        meta["consent_confirmed"] = True
    user = safe_name(user or "user")

    # ---------- 出力先 ----------
    os.makedirs("data", exist_ok=True)
    stem = f"{user}_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = f"data/{stem}.csv"
    meta_path = f"data/{stem}_meta.json"
    t_begin = datetime.now()

    def write_meta(extra=None):
        d = {"user": user, "mode": mode, "sensors": sensors,
             "start": t_begin.isoformat(timespec="seconds"),
             "window_sec": C.WINDOW_SEC, "csv": os.path.basename(csv_path),
             "video": os.path.basename(video_path) if video_path else None,
             "config": config_snapshot()}
        d.update(meta)
        if extra:
            d.update(extra)
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(d, mf, ensure_ascii=False, indent=2)

    # ---------- カメラの準備 → キャリブレーション → 入力監視の開始 ----------
    inp = None
    cam = rec = None
    video_path = None
    try:
        if use_cam:
            from camera_monitor import CameraMonitor
            cam = CameraMonitor(args.camera_index)
            if save_video:
                from video_recorder import VideoRecorder
                rec = VideoRecorder(f"data/{stem}")
    except Exception as e:
        print(f"[main] 開始できません: {e}")
        return

    if cam is not None:
        if args.no_calibration:
            meta["calibration"] = {"status": "skipped"}
        else:
            print("\n[main] 全画面キャリブレーションを開始します(約20〜30秒)")
            try:
                from calibration import run_calibration
                cal = run_calibration(cam)
            except Exception as e:
                print(f"[main] キャリブレーションでエラー: {e!r}")
                cal = None
            if cal:
                cam.set_calibration(cal)
                meta["calibration"] = dict(cal, status="ok")
            else:
                meta["calibration"] = {"status": "skipped_or_failed"}
                print(f"[main] 起動直後の{C.CALIB_SEC}秒で自動的に基準を決めます"
                      "(その間は目を普通に開けて画面を見てください)")

    if use_input:                      # キャリブレーション中のキー操作を数えないよう、ここで開始
        from input_monitor import InputMonitor
        inp = InputMonitor()
        inp.start()
    preview = cam is not None and not args.no_preview
    if preview:
        import cv2

    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
    writer.writeheader()
    write_meta()
    print(f"\n[main] 計測開始 mode={mode} sensors={sensors} -> {csv_path}  (集計単位 {C.WINDOW_SEC}秒)")
    if rec:
        print("[main] 映像はプログラム終了時に data/ へ保存されます(q / Ctrl+C で正常終了してください)")
    print("[main] 終了: " + ("プレビュー窓で q / " if preview else "") + "Ctrl+C")

    win_start = time.time()
    win_end = win_start + C.WINDOW_SEC

    def write_window(now):
        row = {
            "user": user, "mode": mode,
            "start": datetime.fromtimestamp(win_start).isoformat(timespec="seconds"),
            "end": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "duration_sec": now - win_start,
        }
        if inp:
            row.update(inp.snapshot(now))
        if cam:
            row.update(cam.snapshot(now))
        writer.writerow({k: clean(v) for k, v in row.items()})
        f.flush()
        print(f"[main] {row['end']}  typing/min={clean(row.get('typing_per_min', ''))}  "
              f"corr={clean(row.get('correction_ratio', ''))}  blink/min={clean(row.get('blink_per_min', ''))}")

    try:
        while True:
            if cam:
                frame, lm = cam.step()
                if rec and frame is not None:
                    rec.write(frame, time.time())
                if preview and frame is not None:
                    show = frame.copy()
                    cv2.putText(show, f"{cam.status}{'  REC' if rec else ''}  (q: quit)", (10, 30),
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
        if inp:
            inp.stop()                      # 終了後の質問への入力を計測に混ぜない
        if cam:
            cam.close()
        if rec:
            video_path = rec.close()
            if video_path:
                mb = os.path.getsize(video_path) / 1e6
                print(f"[main] 映像を保存しました: {video_path} ({mb:.1f} MB)")
        if preview:
            cv2.destroyAllWindows()

    extra = {"end": datetime.now().isoformat(timespec="seconds")}
    if mode == "experiment":
        extra["fatigue_after"] = ask_rating("\n【終了後】現在の疲労度")
        memo = input("メモ(任意。Enterで省略): ").strip()
        if memo:
            extra["memo"] = memo
    write_meta(extra)
    print(f"[main] 終了しました (記録: {csv_path} / {meta_path})")


if __name__ == "__main__":
    main()
