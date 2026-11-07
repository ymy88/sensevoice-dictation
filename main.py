"""按住右 Option 说话，松开后用 SenseVoice-Small 在本地识别，把文字打进当前窗口。"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from pynput import keyboard

MODEL_DIR = Path(__file__).parent / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
HOTKEY = keyboard.Key.alt_r
SAMPLE_RATE = 16000
MIN_SECONDS = 0.3  # 短于这个长度多半是误触，识别出来通常是空串或一个语气词


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
    chunks: list[np.ndarray] = []
    mic: sd.InputStream | None = None

    def on_audio(data: np.ndarray, frames: int, _time, status) -> None:
        chunks.append(data[:, 0].copy())

    def finish(samples: np.ndarray) -> None:
        text = transcribe(recognizer, samples)
        print(f"> {text}", flush=True)
        if text:
            paste(typer, text)

    def on_press(key) -> None:
        nonlocal mic
        if key != HOTKEY or mic is not None:
            return
        chunks.clear()
        # 麦克风只在按住期间打开，菜单栏的橙色麦克风标志也只在这时出现。
        mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio)
        mic.start()

    def on_release(key) -> None:
        nonlocal mic
        if key != HOTKEY or mic is None:
            return
        mic.stop()
        mic.close()
        mic = None
        if not chunks:
            return
        samples = np.concatenate(chunks)
        if len(samples) < SAMPLE_RATE * MIN_SECONDS:
            return
        # 识别放到别的线程：键盘监听的回调阻塞太久，macOS 会停掉事件监听。
        threading.Thread(target=finish, args=(samples,), daemon=True).start()

    print("按住右 Option 说话，松开后输出文字。Ctrl+C 退出。", flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


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
