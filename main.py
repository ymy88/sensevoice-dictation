"""按住快捷键说话，松开后用 SenseVoice-Small 在本地识别，把文字打进当前窗口。

python main.py               运行
python main.py --set-hotkey  按一下想用的键，设为快捷键
python main.py some.wav      只识别一个 wav 文件（自检）
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from pynput import keyboard
from Quartz import (
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventKeyboardGetUnicodeString,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventSourceUnixProcessID,
    kCGKeyboardEventKeycode,
)

MODEL_DIR = Path(__file__).parent / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
CONFIG = Path(__file__).parent / "config.json"
LAUNCHD_LABEL = "local.sensevoice-dictation"
DEFAULT_HOTKEY = 114  # PC 键盘的 Insert 键在 macOS 上发出的是 Help 键的键码 114
ESC = 53
# 不产生可打印字符、但打字时常用的键：回车、Tab、删除、小键盘回车、向前删除、方向键
TYPING_KEYS = {36, 48, 51, 76, 117, 123, 124, 125, 126}
SAMPLE_RATE = 16000
MIN_SECONDS = 0.3  # 短于这个长度多半是误触，识别出来通常是空串或一个语气词
# 录音里最响的 30ms 片段的 RMS 低于这个值就当没说话。不跳过的话，SenseVoice 会把静音识别成「Okay.」「.」。
# 2026-11-07 在 Ghostty + 内置麦克风上实测：环境音最大 0.0012，正常说话中位数 0.05 以上。换麦克风后可能要调。
SILENCE_RMS = 0.01

# 修饰键的键码 → 事件标志位里表示「这个键按着」的位。
# 左右两边的同名键共用一个通用标志位，所以用区分左右的设备标志位（IOKit 的 NX_DEVICE*KEYMASK）。
MODIFIER_BITS = {
    54: 0x10,  # 右 Cmd
    55: 0x08,  # 左 Cmd
    56: 0x02,  # 左 Shift
    58: 0x20,  # 左 Option
    59: 0x01,  # 左 Control
    60: 0x04,  # 右 Shift
    61: 0x40,  # 右 Option
    62: 0x2000,  # 右 Control
    63: 0x800000,  # Fn
}
KEY_NAMES = {
    114: "Insert",
    54: "右 Cmd",
    55: "左 Cmd",
    56: "左 Shift",
    58: "左 Option",
    59: "左 Control",
    60: "右 Shift",
    61: "右 Option",
    62: "右 Control",
    63: "Fn",
    105: "F13",
    107: "F14",
    113: "F15",
    106: "F16",
    64: "F17",
    79: "F18",
    80: "F19",
}


def key_name(keycode: int) -> str:
    return KEY_NAMES.get(keycode, f"键码 {keycode}")


def key_label(keycode: int) -> str:
    name = key_name(keycode)
    return name if name.startswith("键码") else f"{name}（键码 {keycode}）"


def load_hotkey() -> int:
    if not CONFIG.exists():
        return DEFAULT_HOTKEY
    return int(json.loads(CONFIG.read_text())["keycode"])


def keycode_of(event) -> int:
    return CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)


def modifier_down(keycode: int, event) -> bool:
    return bool(CGEventGetFlags(event) & MODIFIER_BITS[keycode])


def make_recognizer() -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(MODEL_DIR / "model.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=4,
        language="auto",
        use_itn=True,  # 输出标点，数字写成阿拉伯数字
    )


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def loudest_frame_rms(samples: np.ndarray) -> float:
    frame = SAMPLE_RATE * 30 // 1000
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    return float(np.sqrt((frames**2).mean(axis=1)).max())


def paste(typer: keyboard.Controller, text: str) -> None:
    # 不用逐字模拟按键：pynput 把英文字母发成真实键码，拼音输入法开着时会被当成拼音吃掉。
    # 走剪贴板 + Cmd+V，任何输入法下都能原样输入。
    # ponytail: 只恢复剪贴板里的纯文本；原来是图片或文件时会被清掉，需要时改用 NSPasteboard 整体备份。
    old = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input=text.encode())
    with typer.pressed(keyboard.Key.cmd):
        typer.tap("v")
    time.sleep(0.3)  # 等目标 App 读完剪贴板再恢复
    subprocess.run(["pbcopy"], input=old)


def set_hotkey() -> None:
    chosen: list[int] = []
    done = threading.Event()

    def on_event(event_type, event):
        keycode = keycode_of(event)
        pressed = event_type == kCGEventKeyDown or (
            event_type == kCGEventFlagsChanged and keycode in MODIFIER_BITS and modifier_down(keycode, event)
        )
        if not pressed or done.is_set():
            return event
        # 非修饰键会被拦下，当前 App 收不到；设成打字要用的键，这个键就再也打不出来了。
        chars = CGEventKeyboardGetUnicodeString(event, 4, None, None)[1] or ""
        if event_type == kCGEventKeyDown and keycode != ESC and (
            keycode in TYPING_KEYS or (chars != "" and chars.isprintable())  # "".isprintable() 也是 True
        ):
            shown = f"「{chars}」" if chars.strip() and chars.isprintable() else key_name(keycode)
            print(f"{shown} 平时打字要用，换一个键。", flush=True)
            return event
        chosen.append(keycode)
        done.set()
        return event

    print("按一下要用作快捷键的键（Esc 取消）…", flush=True)
    with keyboard.Listener(darwin_intercept=on_event):
        done.wait()
    keycode = chosen[0]
    if keycode == ESC:
        print("已取消。")
        return
    CONFIG.write_text(json.dumps({"keycode": keycode}) + "\n")
    print(f"快捷键设为 {key_label(keycode)}，已写入 {CONFIG}")
    restart = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], capture_output=True
    )
    if restart.returncode == 0:
        print("已重启后台服务，新快捷键已生效。")
    else:
        print("没有找到后台服务。如果在前台运行 main.py，重新启动它即可。")


def main() -> None:
    recognizer = make_recognizer()
    typer = keyboard.Controller()
    hotkey = load_hotkey()
    is_modifier = hotkey in MODIFIER_BITS
    my_pid = os.getpid()
    events: queue.Queue[str] = queue.Queue()
    held = False
    cancelled = False

    # 这个回调里只做入队。回调阻塞太久（比如打开麦克风时弹出权限框），
    # macOS 会停用整个按键监听，之后快捷键既不录音也拦不住。
    def on_event(event_type, event):
        nonlocal held, cancelled
        if CGEventGetIntegerValueField(event, kCGEventSourceUnixProcessID) == my_pid:
            return event  # 自己发的 Cmd+V；快捷键是 Cmd 时不能把它当成按下快捷键
        keycode = keycode_of(event)

        if not is_modifier:
            if keycode != hotkey:
                return event
            if event_type == kCGEventKeyDown and not held:  # 按住不放时系统会重复发按下事件
                held = True
                events.put("down")
            elif event_type == kCGEventKeyUp and held:
                held = False
                events.put("up")
            return None  # 拦下快捷键，不让当前 App 收到（Insert 在终端里会变成 ESC [2~）

        # 修饰键不拦：拦了的话，这个键和别的键的组合（比如右 Cmd+C）就不能用了。
        # 所以只有单独按住时才录音；按住期间按了别的键，就当是组合键，取消录音。
        if keycode == hotkey and event_type == kCGEventFlagsChanged:
            if modifier_down(hotkey, event) and not held:
                held = True
                cancelled = False
                events.put("down")
            elif not modifier_down(hotkey, event) and held:
                held = False
                if not cancelled:
                    events.put("up")
        elif held and not cancelled and event_type in (kCGEventKeyDown, kCGEventFlagsChanged):
            cancelled = True
            events.put("cancel")
        return event

    listener = keyboard.Listener(darwin_intercept=on_event)
    listener.start()
    print(f"按住 {key_name(hotkey)} 说话，松开后输出文字。Ctrl+C 退出。", flush=True)

    chunks: list[np.ndarray] = []
    mic: sd.InputStream | None = None
    while True:
        event = events.get()
        if event == "down" and mic is None:
            chunks.clear()
            # 麦克风只在按住期间打开，菜单栏的橙色麦克风标志也只在这时出现。
            mic = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=lambda data, *_: chunks.append(data[:, 0].copy()),
            )
            mic.start()
            print("录音中…", flush=True)
        elif event == "cancel" and mic is not None:
            mic.stop()
            mic.close()
            mic = None
            print("按了组合键，取消录音", flush=True)
        elif event == "up" and mic is not None:
            mic.stop()
            mic.close()
            mic = None
            seconds = sum(len(c) for c in chunks) / SAMPLE_RATE
            print(f"录到 {seconds:.1f} 秒", flush=True)
            if not chunks:
                continue
            samples = np.concatenate(chunks)
            if len(samples) < SAMPLE_RATE * MIN_SECONDS:
                continue
            if loudest_frame_rms(samples) < SILENCE_RMS:
                # 没有麦克风权限时 macOS 不报错，只给全零的音频，所以这里要打印出来，否则看不出是权限问题。
                print(f"(没检测到声音，跳过；最大 RMS {loudest_frame_rms(samples):.4f})", flush=True)
                continue
            text = transcribe(recognizer, samples)
            print(f"> {text}", flush=True)
            if text:
                paste(typer, text)


if __name__ == "__main__":
    if sys.argv[1:] == ["--set-hotkey"]:
        set_hotkey()
    elif len(sys.argv) == 2:
        import wave

        with wave.open(sys.argv[1]) as w:
            assert w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        print(transcribe(make_recognizer(), pcm.astype(np.float32) / 32768))
    else:
        main()
