"""按住快捷键说话，松开后用 X-ASR 在本地识别，把文字打进当前窗口。

python main.py               运行
python main.py --set-hotkey  按一下想用的键，设为快捷键
python main.py some.wav      只识别一个 wav 文件（自检）
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from pynput import keyboard
from Quartz import (
    CGEventGetFlags,
    CGEventCreateKeyboardEvent,
    CGEventGetIntegerValueField,
    CGEventKeyboardGetUnicodeString,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventSourceUnixProcessID,
    kCGKeyboardEventKeycode,
)

MODELS = Path(__file__).parent / "models"
MODEL_DIR = MODELS / "sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"
HOTWORDS = Path(__file__).parent / "hotwords.txt"  # 一行一个词；文件存在时识别会偏向这些词
CONFIG = Path(__file__).parent / "config.json"
RECORDINGS_DIR = Path(__file__).parent / "recordings"  # 这个目录存在时，每段录音都存成 wav，用来比较模型
LAUNCHD_LABEL = "local.sensevoice-dictation"
DEFAULT_HOTKEY = 114  # PC 键盘的 Insert 键在 macOS 上发出的是 Help 键的键码 114
ESC = 53
FN = 63
# 打字时常用、但不产生可见字符的键：回车、Tab、空格、删除、小键盘回车、向前删除、方向键
TYPING_KEYS = {36, 48, 49, 51, 76, 117, 123, 124, 125, 126}
CMD_OR_CONTROL = {54, 55, 59, 62}
SAMPLE_RATE = 16000
# 两个以上、只隔一个空格的单个字母。ponytail: 英文句子里的「a」「I」挨着另一个单字母时也会被合并，需要时改成只合并大写字母。
SPELLED_LETTERS = re.compile(r"(?<![A-Za-z])[A-Za-z](?: [A-Za-z])+(?![A-Za-z])")
MIN_SECONDS = 0.3  # 短于这个长度多半是误触，识别出来通常是空串或一个语气词
# 松开快捷键后再录这么久才关麦克风。蓝牙耳机的声音传到电脑有延迟，松开就关的话最后一个字只录到一半，
# 2026-11-07 的录音里「推送」被识别成「推销」「退缩」。
TAIL_SECONDS = 0.3
# 识别前在录音末尾补这么长的静音。X-ASR 认一个字时要用到这个字后面的声音，结尾没有静音时最后一个字常被丢掉。
# 2026-11-07 用 42 段录音测：不补时「GPU」常变成「gp」；补 0.3 秒能认出来，补 0.5 秒有时会在结尾多出一个词。
PAD_SECONDS = 0.3
# 录音里最响的 30ms 片段的 RMS 低于这个值就当没说话。不跳过的话，模型会把静音识别成「Okay.」「.」。
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


def key_char(keycode: int) -> str:
    """这个键在当前键盘布局下不按修饰键时打出的字符，例如 40 → "K"。不出字符的键返回空串。"""
    chars = CGEventKeyboardGetUnicodeString(CGEventCreateKeyboardEvent(None, keycode, True), 4, None, None)[1] or ""
    return chars.upper() if chars.strip() and chars.isprintable() else ""


def key_name(keycode: int) -> str:
    return KEY_NAMES.get(keycode) or key_char(keycode) or f"键码 {keycode}"


def hotkey_name(keycode: int, modifiers: frozenset[int]) -> str:
    return "+".join([key_name(m) for m in sorted(modifiers)] + [key_name(keycode)])


def is_typing_key(keycode: int) -> bool:
    return keycode in TYPING_KEYS or key_char(keycode) != ""


def load_hotkey() -> tuple[int, frozenset[int]]:
    if not CONFIG.exists():
        return DEFAULT_HOTKEY, frozenset()
    config = json.loads(CONFIG.read_text())
    return int(config["keycode"]), frozenset(config.get("modifiers", []))


def keycode_of(event) -> int:
    return CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)


def held_modifiers(event) -> frozenset[int]:
    # 不算 Fn：方向键、F 键的按键事件里，系统会自动带上 Fn 标志位。
    flags = CGEventGetFlags(event)
    return frozenset(m for m, bit in MODIFIER_BITS.items() if m != FN and flags & bit)


def modifier_down(keycode: int, event) -> bool:
    return bool(CGEventGetFlags(event) & MODIFIER_BITS[keycode])


def make_recognizer() -> sherpa_onnx.OfflineRecognizer:
    # 2026-11-07 用同一段 28 秒中英混说的录音比较：X-ASR 加数字转换的错误率 8.3%，SenseVoice-Small 13.5%，
    # 识别耗时都在 0.3 秒左右。Qwen3-ASR、FireRedASR2 慢 10 倍以上，错误率也不更低。
    hotwords = {}
    if HOTWORDS.exists():
        hotwords = dict(
            decoding_method="modified_beam_search",  # 热词只在这种解码方式下生效
            hotwords_file=str(HOTWORDS),
            hotwords_score=2.0,
            modeling_unit="cjkchar+bpe",  # 热词里的汉字按字切分，英文按 bpe.vocab 切分
            bpe_vocab=str(MODEL_DIR / "bpe.vocab"),
        )
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=4,
        # 把「三八八七」转成「3887」。副作用：「一个」也会变成「1个」。
        rule_fsts=str(MODELS / "itn_zh_number.fst"),
        **hotwords,
    )


def join_spelled_letters(text: str) -> str:
    """把逐个字母读出的词合并：X-ASR 把「GPU」输出成「G P U」。"""
    return SPELLED_LETTERS.sub(lambda m: m.group().replace(" ", ""), text)


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    padding = np.zeros(int(SAMPLE_RATE * PAD_SECONDS), dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, np.concatenate([samples, padding]))
    recognizer.decode_stream(stream)
    return join_spelled_letters(stream.result.text.strip())


def loudest_frame_rms(samples: np.ndarray) -> float:
    frame = SAMPLE_RATE * 30 // 1000
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    return float(np.sqrt((frames**2).mean(axis=1)).max())


def save_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"已保存 {path}", flush=True)


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
    result: list[tuple[int, frozenset[int]] | None] = []
    done = threading.Event()
    held: set[int] = set()
    pressed_together: set[int] = set()  # 这一轮按过的修饰键
    other_key_pressed = False

    def finish(hotkey: tuple[int, frozenset[int]] | None) -> None:
        result.append(hotkey)
        done.set()

    # 修饰键要等全部松开才知道是单独按，还是要和别的键组合，所以松开时再记录。
    def on_event(event_type, event):
        nonlocal other_key_pressed
        if done.is_set():
            return event
        keycode = keycode_of(event)
        if event_type == kCGEventFlagsChanged and keycode in MODIFIER_BITS:
            if modifier_down(keycode, event):
                held.add(keycode)
                pressed_together.add(keycode)
            else:
                held.discard(keycode)
                if not held:
                    if not other_key_pressed and len(pressed_together) == 1:
                        finish((keycode, frozenset()))
                    elif not other_key_pressed:
                        print("同时按了几个修饰键。请单独按一个修饰键，或者按住修饰键再按一个键。", flush=True)
                    pressed_together.clear()
                    other_key_pressed = False
            return event
        if event_type != kCGEventKeyDown:
            return event
        modifiers = frozenset(held - {FN})
        other_key_pressed = bool(held)
        if keycode == ESC and not modifiers:
            finish(None)
        # 快捷键会被拦下，当前 App 收不到。打字要用的键至少要配 Cmd 或 Control，否则这个键就打不出来了。
        elif is_typing_key(keycode) and not modifiers & CMD_OR_CONTROL:
            print(f"{hotkey_name(keycode, modifiers)} 平时打字要用，换一个键，或者加上 Cmd 或 Control。", flush=True)
        else:
            finish((keycode, modifiers))
        return event

    print("按一下要用作快捷键的键或组合键（Esc 取消）…", flush=True)
    with keyboard.Listener(darwin_intercept=on_event):
        done.wait()
    if result[0] is None:
        print("已取消。")
        return
    keycode, modifiers = result[0]
    CONFIG.write_text(json.dumps({"keycode": keycode, "modifiers": sorted(modifiers)}) + "\n")
    print(f"快捷键设为 {hotkey_name(keycode, modifiers)}，已写入 {CONFIG}")
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
    hotkey, hotkey_modifiers = load_hotkey()
    modifier_alone = hotkey in MODIFIER_BITS and not hotkey_modifiers
    my_pid = os.getpid()
    events: queue.Queue[str] = queue.Queue()
    held = False
    cancelled = False
    swallowing = False  # 快捷键的按下事件被拦了，它的松开事件也要拦

    # 这个回调里只做入队。回调阻塞太久（比如打开麦克风时弹出权限框），
    # macOS 会停用整个按键监听，之后快捷键既不录音也拦不住。
    def on_event(event_type, event):
        nonlocal held, cancelled
        if CGEventGetIntegerValueField(event, kCGEventSourceUnixProcessID) == my_pid:
            return event  # 自己发的 Cmd+V；快捷键是 Cmd 时不能把它当成按下快捷键
        keycode = keycode_of(event)

        if not modifier_alone:
            return on_key_or_combo(event_type, event, keycode)

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

    # 普通键或组合键：按下开始录音，松开其中任意一个键就停止。快捷键会被拦下，当前 App 收不到
    # （Insert 在终端里会变成 ESC [2~）。
    def on_key_or_combo(event_type, event, keycode):
        nonlocal held, swallowing
        if event_type == kCGEventKeyDown and keycode == hotkey:
            if held or swallowing:  # 按住不放时系统会重复发按下事件
                return None
            if held_modifiers(event) != hotkey_modifiers:
                return event
            held = swallowing = True
            events.put("down")
            return None
        if event_type == kCGEventKeyUp and keycode == hotkey and swallowing:
            swallowing = False
            if held:
                held = False
                events.put("up")
            return None
        if event_type == kCGEventFlagsChanged and held and held_modifiers(event) != hotkey_modifiers:
            held = False  # 先松开了修饰键
            events.put("up")
        return event

    listener = keyboard.Listener(darwin_intercept=on_event)
    listener.start()
    print(f"按住 {hotkey_name(hotkey, hotkey_modifiers)} 说话，松开后输出文字。Ctrl+C 退出。", flush=True)

    chunks: list[np.ndarray] = []
    mic: sd.InputStream | None = None
    while True:
        event = events.get()
        if event == "down" and mic is None:
            chunks.clear()
            # PortAudio 只在初始化时读一次设备列表。服务启动后才连上的蓝牙耳机、切换过的默认麦克风，
            # 不重新初始化就用不上，录到的全是 0。2026-11-07 实测重新初始化耗时 2~3ms。
            sd._terminate()
            sd._initialize()
            # 麦克风只在按住期间打开，菜单栏的橙色麦克风标志也只在这时出现。
            mic = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=lambda data, *_: chunks.append(data[:, 0].copy()),
            )
            mic.start()
            print(f"录音中…（{sd.query_devices(kind='input')['name']}）", flush=True)
        elif event == "cancel" and mic is not None:
            mic.stop()
            mic.close()
            mic = None
            print("按了组合键，取消录音", flush=True)
        elif event == "up" and mic is not None:
            time.sleep(TAIL_SECONDS)
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
            if RECORDINGS_DIR.is_dir():
                save_wav(RECORDINGS_DIR / time.strftime("%Y%m%d-%H%M%S.wav"), samples)
            text = transcribe(recognizer, samples)
            print(f"> {text}", flush=True)
            if text:
                paste(typer, text)


if __name__ == "__main__":
    if sys.argv[1:] == ["--set-hotkey"]:
        set_hotkey()
    elif len(sys.argv) == 2:
        with wave.open(sys.argv[1]) as w:
            assert w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        print(transcribe(make_recognizer(), pcm.astype(np.float32) / 32768))
    else:
        main()
