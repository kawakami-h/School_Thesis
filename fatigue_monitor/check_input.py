"""入力監視の診断。fatigue_monitor フォルダで `python check_input.py` を実行し、
各10秒の間に次を行ってください。
  1回目: キー入力 / マウス移動 / クリック / スクロール(マウスホイールとタッチパッドの両方)
  2回目: 上に加えて、Alt+Tab・タスクバーのクリック・別ウィンドウのクリックでウィンドウを数回切り替える"""
import platform
import sys
import time

print("Python", sys.version.split()[0], "/", platform.platform())
try:
    from pynput import keyboard, mouse
except Exception as e:
    print("[NG] pynputをimportできません:", repr(e))
    raise SystemExit(1)

cnt = {"key": 0, "move": 0, "click": 0, "scroll": 0}
scroll_samples = []


def on_scroll(x, y, dx, dy):
    cnt["scroll"] += 1
    if len(scroll_samples) < 8:
        scroll_samples.append((dx, dy))


kl = keyboard.Listener(on_press=lambda k: cnt.__setitem__("key", cnt["key"] + 1))
ml = mouse.Listener(
    on_move=lambda x, y: cnt.__setitem__("move", cnt["move"] + 1),
    on_click=lambda x, y, b, p: cnt.__setitem__("click", cnt["click"] + 1),
    on_scroll=on_scroll,
)
kl.start(); ml.start()
print("【1回目】10秒間、キー入力・マウス移動・クリック・スクロールをしてください...")
time.sleep(10)
kl.stop(); ml.stop()
print("受信数:", cnt)
print("スクロールの生データ(dx, dy):", scroll_samples)
for name in ("key", "move", "click", "scroll"):
    print(f"  {name}: {'OK' if cnt[name] else '0件(受信できていません)'}")

# 本体のクラスでも確認
sys.path.insert(0, ".")
try:
    from input_monitor import InputMonitor
    im = InputMonitor(); im.start()
    print("\n【2回目】InputMonitorで確認します。10秒間、スクロールとウィンドウ切り替えを数回してください...")
    time.sleep(10)
    s = im.snapshot(time.time()); im.stop()
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})
    print(f"  scroll_count={s['scroll_count']}  window_switch_count={s['window_switch_count']}")
except Exception as e:
    print("[NG] InputMonitorでエラー:", repr(e))
