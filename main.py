"""按住 Insert 说话，松开后用 SenseVoice-Small 在本地识别，把文字打进当前窗口。"""

import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from pynput import keyboard
from Quartz import CGEventGetIntegerValueField, kCGKeyboardEventKeycode

MODEL_DIR = Path(__file__).parent / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
# PC 键盘的 Insert 键在 macOS 上发出的是 Help 键的键码 114。
HOTKEY_VK = 114
SAMPLE_RATE = 16000
MIN_SECONDS = 0.3  # 短于这个长度多半是误触，识别出来通常是空串或一个语气词
# 录音里最响的 30ms 片段的 RMS 低于这个值就当没说话。不跳过的话，SenseVoice 会把静音识别成「Okay.」「.」。
# 2026-11-07 在 Ghostty + 内置麦克风上实测：环境音最大 0.0012，正常说话中位数 0.05 以上。换麦克风后可能要调。
SILENCE_RMS = 0.01


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


def main() -> None:
    recognizer = make_recognizer()
    typer = keyboard.Controller()
    events: queue.Queue[str] = queue.Queue()
    held = False

    def on_press(key) -> None:
        nonlocal held
        if getattr(key, "vk", None) == HOTKEY_VK and not held:  # 按住不放时系统会重复发按下事件
            held = True
            events.put("down")

    def on_release(key) -> None:
        nonlocal held
        if getattr(key, "vk", None) == HOTKEY_VK and held:
            held = False
            events.put("up")

    def swallow_hotkey(event_type, event):
        # 拦下 Insert，不让当前 App 收到（终端会把它转成 ESC [2~）。
        if CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode) == HOTKEY_VK:
            return None
        return event

    # 键盘回调里只做入队。回调阻塞太久（比如打开麦克风时弹出权限框），
    # macOS 会停用整个按键监听，之后 Insert 既不录音也拦不住。
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, darwin_intercept=swallow_hotkey)
    listener.start()
    print("按住 Insert 说话，松开后输出文字。Ctrl+C 退出。", flush=True)

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
        elif event == "up" and mic is not None:
            mic.stop()
            mic.close()
            mic = None
            if not chunks:
                continue
            samples = np.concatenate(chunks)
            if len(samples) < SAMPLE_RATE * MIN_SECONDS or loudest_frame_rms(samples) < SILENCE_RMS:
                continue
            text = transcribe(recognizer, samples)
            print(f"> {text}", flush=True)
            if text:
                paste(typer, text)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # 自检：python main.py some.wav，只识别文件，不监听键盘
        import wave

        with wave.open(sys.argv[1]) as w:
            assert w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        print(transcribe(make_recognizer(), pcm.astype(np.float32) / 32768))
    else:
        main()
