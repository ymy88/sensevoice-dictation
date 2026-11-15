"""按住快捷键说话，松开后用 X-ASR 在本地识别，把文字打进当前窗口。

python main.py               运行
python main.py --set-hotkey  按一下想用的键，设为快捷键
python main.py some.wav      只识别一个 wav 文件（自检）
"""

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import AppKit
import numpy as np
import sherpa_onnx
import sounddevice as sd
from pynput import keyboard
from PyObjCTools import AppHelper
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
# 识别后把这些词换掉，不区分大小写。热词纠正不了的才放这里：Claude 和 cloud 读音几乎一样，
# 2026-11-07 试过热词，分数调到能纠正时别的句子会整段吞词。
REPLACEMENTS = {"cloud code": "Claude Code"}
MIN_SECONDS = 0.3 # 短于这个长度多半是误触，识别出来通常是空串或一个语气词
# 松开快捷键后再录这么久才关麦克风。蓝牙耳机的声音传到电脑有延迟，松开就关的话最后一个字只录到一半，
# 2026-11-07 的录音里「推送」被识别成「推销」「退缩」。
TAIL_SECONDS = 0.3
# 识别前在录音末尾补这么长的静音。X-ASR 认一个字时要用到这个字后面的声音，结尾没有静音时最后一个字常被丢掉。
# 2026-11-07 用 42 段录音测：不补时「GPU」常变成「gp」；补 0.3 秒能认出来，补 0.5 秒有时会在结尾多出一个词。
PAD_SECONDS = 0.3
# X-ASR 一次最多处理 4000 帧（40 秒，含补的静音），再长就报错。更长的录音切成不超过这个长度的几段分别识别。
CHUNK_SECONDS = 30
FRAME = SAMPLE_RATE * 30 // 1000  # 30ms，算音量用的片段长度
# 录音里最响的 30ms 片段的 RMS 低于这个值就当没说话。不跳过的话，模型会把静音识别成「Okay.」「.」。
# 2026-11-07 在 Ghostty + 内置麦克风上实测：环境音最大 0.0012，正常说话中位数 0.05 以上。换麦克风后可能要调。
SILENCE_RMS = 0.01
# 菜单栏图标下方提示框的大小（点）
HINT_WIDTH = 120
HINT_HEIGHT = 30

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


def replace_words(text: str) -> str:
    for wrong, right in REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def split_long(samples: np.ndarray) -> list[np.ndarray]:
    """超过 CHUNK_SECONDS 的录音，在 20 秒到 CHUNK_SECONDS 之间最安静的 30ms 处切开，多半是两句话之间的停顿。"""
    chunks = []
    while len(samples) > SAMPLE_RATE * CHUNK_SECONDS:
        start = SAMPLE_RATE * 20
        cut = start + int(frame_rms(samples[start : SAMPLE_RATE * CHUNK_SECONDS]).argmin()) * FRAME + FRAME // 2
        chunks.append(samples[:cut])
        samples = samples[cut:]
    return chunks + [samples]


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    padding = np.zeros(int(SAMPLE_RATE * PAD_SECONDS), dtype=np.float32)
    texts = []
    for chunk in split_long(samples):
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.concatenate([chunk, padding]))
        recognizer.decode_stream(stream)
        texts.append(stream.result.text.strip())
    # X-ASR 自己在标点后面也加空格，所以段与段之间用空格连接
    return replace_words(join_spelled_letters(" ".join(t for t in texts if t)))


def frame_rms(samples: np.ndarray) -> np.ndarray:
    """每个 30ms 片段的 RMS。"""
    frames = samples[: len(samples) // FRAME * FRAME].reshape(-1, FRAME)
    return np.sqrt((frames**2).mean(axis=1))


def loudest_frame_rms(samples: np.ndarray) -> float:
    return float(frame_rms(samples).max())


def lid_closed() -> bool:
    out = subprocess.run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"], capture_output=True, text=True).stdout
    return '"AppleClamshellState" = Yes' in out


def fallback_input(devices) -> int | None:
    """MacBook 合盖后自带麦克风只给全零，换一个能用的麦克风。跳过 iPhone 的麦克风（会去开手机的麦克风），
    也跳过同时有输出的设备（Zoom 之类的虚拟设备是这样；蓝牙耳机、USB 麦克风的输入是单独一个设备）。
    ponytail: 按名字和声道数猜，有多个外接麦克风时取第一个；需要时改成在 config.json 里指定设备名。"""
    for d in devices:
        if d["max_input_channels"] and not d["max_output_channels"] and not re.search("MacBook|iPhone", d["name"]):
            return d["index"]
    return None


def pick_input() -> int | None:
    """要用的麦克风；None 表示用系统默认输入设备。2026-11-08 耳机重连后 macOS 没把输入切回耳机，
    默认输入还是合着盖的自带麦克风，录到的全是 0。"""
    if "MacBook" not in sd.query_devices(kind="input")["name"] or not lid_closed():
        return None
    return fallback_input(sd.query_devices())


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


class LevelBars(AppKit.NSView):
    """跟着说话音量跳的三根竖条，用来看出麦克风真的收到了声音。没声音时只剩最短的一截。"""

    def drawRect_(self, rect):
        AppKit.NSColor.labelColor().set()  # 跟着明暗模式变，在浅色背景上是深色
        for i, factor in enumerate((0.6, 1.0, 0.75)):
            height = 3 + 13 * min(1.0, self.level * factor)
            bar = AppKit.NSMakeRect(i * 8.0, (16 - height) / 2, 4.0, height)
            AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bar, 2.0, 2.0).fill()


class ClickHandler(AppKit.NSObject):
    """菜单栏按钮的 target。按钮的动作只能由 Objective-C 对象接收，这里转给 Python 函数。"""

    def click_(self, _sender) -> None:
        self.on_click()


class MenuBar:
    """菜单栏图标。程序在跑就一直显示；按住快捷键时图标换成实心的，并在图标下方弹出提示。
    画界面必须在主线程，所以公开方法都用 AppHelper.callAfter 转过去，调用方不用管线程。
    点一下图标等于按下快捷键，再点一下等于松开——键盘监听被系统拦住时用这个。"""

    def __init__(self, events: queue.Queue[str]) -> None:
        self.events = events
        self.app = AppKit.NSApplication.sharedApplication()
        self.app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # 只占菜单栏，不进 Dock 和 Cmd+Tab
        self.item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)

        self.hint = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, HINT_WIDTH, HINT_HEIGHT),
            AppKit.NSWindowStyleMaskBorderless,  # 无边框、不能成为主窗口，所以不会抢走键盘焦点
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.hint.setLevel_(AppKit.NSStatusWindowLevel)  # 盖在全屏 App 上面
        self.hint.setIgnoresMouseEvents_(True)
        self.hint.setOpaque_(False)
        self.hint.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.hint.setReleasedWhenClosed_(False)
        self.hint.setHidesOnDeactivate_(False)
        bubble = AppKit.NSVisualEffectView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, HINT_WIDTH, HINT_HEIGHT))
        bubble.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
        bubble.setWantsLayer_(True)
        bubble.layer().setCornerRadius_(HINT_HEIGHT / 2)
        bubble.layer().setMasksToBounds_(True)
        label = AppKit.NSTextField.labelWithString_("正在听写")
        label.setFrame_(AppKit.NSMakeRect(0, (HINT_HEIGHT - 16) / 2, 69, 16))
        label.setAlignment_(AppKit.NSTextAlignmentRight)
        label.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        label.setTextColor_(AppKit.NSColor.labelColor())
        bubble.addSubview_(label)
        self.bars = LevelBars.alloc().initWithFrame_(AppKit.NSMakeRect(75, (HINT_HEIGHT - 16) / 2, 24, 16))
        self.bars.level = 0.0  # 0~1，由录音回调更新
        bubble.addSubview_(self.bars)
        self.hint.setContentView_(bubble)
        self.level = 0.0
        AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.033, True, self._tick)

        # 不给菜单，直接挂动作：点一下就切换录音状态。按钮只弱引用 target，所以要留个强引用。
        self.clicker = ClickHandler.alloc().init()
        self.clicker.on_click = lambda: events.put("toggle")
        self.item.button().setTarget_(self.clicker)
        self.item.button().setAction_("click:")

        self.set_recording(False)

    def set_level(self, level: float) -> None:
        """录音回调里调用，传这一小块音频的 RMS。说话时 RMS 中位数 0.05 左右，乘 12 后接近满格。"""
        self.level = min(1.0, level * 12)

    def _tick(self, _timer) -> None:
        if not self.hint.isVisible():
            return
        # 取「这一帧的音量」和「上一帧的 80%」里大的那个，字与字之间的间隙里竖条不会一帧掉到底
        level = max(self.level, self.bars.level * 0.8)
        if level != self.bars.level:
            self.bars.level = level
            self.bars.setNeedsDisplay_(True)

    def set_recording(self, on: bool) -> None:
        AppHelper.callAfter(self._set_recording, on)

    def _set_recording(self, on: bool) -> None:
        icon = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("mic.fill" if on else "mic", None)
        icon.setTemplate_(True)  # 跟随菜单栏深浅色
        self.item.button().setImage_(icon)
        if on:
            self._place_under_icon()
            self.hint.orderFront_(None)
        else:
            self.hint.orderOut_(None)

    def _place_under_icon(self) -> None:
        button = self.item.button()
        frame = button.window().convertRectToScreen_(button.convertRect_toView_(button.bounds(), None))
        # 屏幕坐标原点在左下角，图标下方就是 y 更小的一侧。状态栏窗口比图标本身高一点，留 6 点空隙。
        x = frame.origin.x + (frame.size.width - HINT_WIDTH) / 2
        y = frame.origin.y - HINT_HEIGHT - 6
        self.hint.setFrameOrigin_((x, y))


def dictate(
    recognizer: sherpa_onnx.OfflineRecognizer, typer: keyboard.Controller, events: queue.Queue[str], menubar: MenuBar
) -> None:
    """处理按键线程发来的事件。跑在主线程之外：主线程留给菜单栏的 run loop。"""
    chunks: list[np.ndarray] = []
    mic: sd.InputStream | None = None
    while True:
        event = events.get()
        if event == "toggle":  # 点击图标：麦克风开着就是结束，否则就是开始
            event = "down" if mic is None else "up"
        if event == "down" and mic is None:
            menubar.set_recording(True)
            chunks.clear()
            # PortAudio 只在初始化时读一次设备列表。服务启动后才连上的蓝牙耳机、切换过的默认麦克风，
            # 不重新初始化就用不上，录到的全是 0。2026-11-07 实测重新初始化耗时 2~3ms。
            sd._terminate()
            sd._initialize()
            device = pick_input()
            # 麦克风只在按住期间打开，菜单栏的橙色麦克风标志也只在这时出现。
            def on_audio(data, *_) -> None:
                block = data[:, 0].copy()
                chunks.append(block)
                menubar.set_level(float(np.sqrt((block**2).mean())))

            mic = sd.InputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=on_audio,
            )
            mic.start()
            print(f"录音中…（{sd.query_devices(device, kind='input')['name']}）", flush=True)
        elif event == "cancel" and mic is not None:
            menubar.set_recording(False)
            mic.stop()
            mic.close()
            mic = None
            print("按了组合键，取消录音", flush=True)
        elif event == "up" and mic is not None:
            time.sleep(TAIL_SECONDS)
            mic.stop()
            mic.close()
            mic = None
            menubar.set_recording(False)
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

    menubar = MenuBar(events)
    # 菜单栏的 run loop 在 C 里跑，Python 的信号处理函数平时轮不到执行，定时器让它每 0.2 秒回到解释器一次。
    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())

    def check_interrupt(_timer) -> None:
        if interrupted.is_set():
            menubar.app.terminate_(None)  # stop_ 停不掉这个 run loop

    AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.2, True, check_interrupt)
    threading.Thread(target=dictate, args=(recognizer, typer, events, menubar), daemon=True).start()
    menubar.app.run()  # 菜单栏图标和提示框要主线程的 run loop 才画得出来


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
