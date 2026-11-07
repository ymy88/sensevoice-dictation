"""快捷键逻辑的自检：uv run python test_hotkey.py

不向系统发送任何按键：用假的键盘监听拿到回调函数，直接把构造的按键事件交给回调。
麦克风、识别、粘贴、config.json 和 launchctl 也都换成假的。
"""

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import Quartz as Q

import main

GENERIC = {
    54: Q.kCGEventFlagMaskCommand,
    55: Q.kCGEventFlagMaskCommand,
    56: Q.kCGEventFlagMaskShift,
    60: Q.kCGEventFlagMaskShift,
    58: Q.kCGEventFlagMaskAlternate,
    61: Q.kCGEventFlagMaskAlternate,
    59: Q.kCGEventFlagMaskControl,
    62: Q.kCGEventFlagMaskControl,
}
R_CTRL, R_SHIFT, L_CMD, L_SHIFT, R_OPT, L_OPT = 62, 60, 55, 56, 61, 58
K, A, C, INSERT, F13, ESC = 40, 0, 8, 114, 105, 53

callback = None


class FakeListener:
    def __init__(self, darwin_intercept=None, **_):
        global callback
        callback = darwin_intercept

    def start(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class Keyboard:
    """按顺序把按键事件交给回调，记下每个事件有没有被拦下。"""

    def __init__(self):
        self.held: set[int] = set()
        self.swallowed: list[str] = []

    def _send(self, event_type, keycode, label):
        e = Q.CGEventCreateKeyboardEvent(None, keycode, event_type == Q.kCGEventKeyDown)
        Q.CGEventSetType(e, event_type)
        # 在本进程里构造的事件带着本进程的进程号，main.py 会把它当成自己发的 Cmd+V 而忽略。真键盘的事件是 0。
        Q.CGEventSetIntegerValueField(e, Q.kCGEventSourceUnixProcessID, 0)
        flags = 0
        for m in self.held:
            flags |= GENERIC[m] | main.MODIFIER_BITS[m]
        Q.CGEventSetFlags(e, flags)
        if callback(event_type, e) is None:
            self.swallowed.append(label)

    def mod(self, keycode, down):
        (self.held.add if down else self.held.discard)(keycode)
        self._send(Q.kCGEventFlagsChanged, keycode, f"m{keycode}{'+' if down else '-'}")

    def key(self, keycode, down=True):
        self._send(Q.kCGEventKeyDown if down else Q.kCGEventKeyUp, keycode, f"k{keycode}{'+' if down else '-'}")

    def tap(self, keycode):
        self.key(keycode, True)
        self.key(keycode, False)


main.keyboard.Listener = FakeListener
main.subprocess.run = lambda *a, **k: type("R", (), {"returncode": 1, "stdout": b""})()
main.CONFIG = Path(tempfile.mkdtemp()) / "config.json"
main.print = lambda *a, **k: None
main.TAIL_SECONDS = 0  # 测试不等真麦克风


# ---------- --set-hotkey ----------


def record(steps) -> dict | None:
    main.CONFIG.unlink(missing_ok=True)
    t = threading.Thread(target=main.set_hotkey, daemon=True)
    t.start()
    while callback is None or not t.is_alive():
        time.sleep(0.01)
    kb = Keyboard()
    steps(kb)
    t.join(1)
    assert not t.is_alive(), "set_hotkey 没有结束"
    return main.load_hotkey() if main.CONFIG.exists() else None


def set_cases():
    global callback
    cases = {
        "单独右 Ctrl": (lambda kb: (kb.mod(R_CTRL, True), kb.mod(R_CTRL, False)), (R_CTRL, frozenset())),
        "右 Ctrl+K": (
            lambda kb: (kb.mod(R_CTRL, True), kb.tap(K), kb.mod(R_CTRL, False)),
            (K, frozenset({R_CTRL})),
        ),
        "右 Ctrl+右 Shift 一起松开不算，之后按 F13": (
            lambda kb: (kb.mod(R_CTRL, True), kb.mod(R_SHIFT, True), kb.mod(R_SHIFT, False), kb.mod(R_CTRL, False), kb.tap(F13)),
            (F13, frozenset()),
        ),
        "Shift+A、单独 K 都拒绝，之后按 Insert": (
            lambda kb: (kb.mod(L_SHIFT, True), kb.tap(A), kb.mod(L_SHIFT, False), kb.tap(K), kb.tap(INSERT)),
            (INSERT, frozenset()),
        ),
        "左 Cmd+左 Shift+K": (
            lambda kb: (kb.mod(L_CMD, True), kb.mod(L_SHIFT, True), kb.tap(K), kb.mod(L_SHIFT, False), kb.mod(L_CMD, False)),
            (K, frozenset({L_CMD, L_SHIFT})),
        ),
        "Esc 取消": (lambda kb: kb.tap(ESC), None),
    }
    for name, (steps, expected) in cases.items():
        callback = None
        got = record(steps)
        assert got == expected, f"--set-hotkey {name}: 期望 {expected}，实际 {got}"
        print(f"ok  --set-hotkey {name}")


# ---------- 运行时 ----------


class FakeStream:
    opened = 0

    def __init__(self, callback=None, **_):
        FakeStream.opened += 1
        self.callback = callback

    def start(self):
        self.callback(np.full((8000, 1), 0.1, dtype=np.float32))  # 0.5 秒有声音的音频

    def stop(self):
        pass

    def close(self):
        pass


pastes: list[str] = []
main.sd.InputStream = FakeStream
main.make_recognizer = lambda: None
main.transcribe = lambda r, s: "x"
main.paste = lambda typer, text: pastes.append(text)


def run(config: dict | None, steps) -> tuple[int, int, list[str]]:
    """返回 (打开麦克风次数, 粘贴次数, 被拦下的事件)。"""
    global callback
    if config is None:
        main.CONFIG.unlink(missing_ok=True)
    else:
        main.CONFIG.write_text(main.json.dumps(config))
    callback = None
    FakeStream.opened = 0
    pastes.clear()
    threading.Thread(target=main.main, daemon=True).start()
    while callback is None:
        time.sleep(0.01)
    kb = Keyboard()
    steps(kb)
    time.sleep(0.2)  # 等主循环处理完队列
    return FakeStream.opened, len(pastes), kb.swallowed


def runtime_cases():
    def insert_with_repeat(kb):
        kb.key(INSERT)
        kb.key(INSERT)  # 按住不放时的重复按下
        kb.key(INSERT, False)

    rctrl_k = {"keycode": K, "modifiers": [R_CTRL]}
    cases = [
        ("默认 Insert，带重复按下", None, insert_with_repeat, (1, 1, ["k114+", "k114+", "k114-"])),
        ("旧格式 config 只有 keycode", {"keycode": F13}, lambda kb: kb.tap(F13), (1, 1, ["k105+", "k105-"])),
        ("单独右 Option", {"keycode": R_OPT}, lambda kb: (kb.mod(R_OPT, True), kb.mod(R_OPT, False)), (1, 1, [])),
        (
            "右 Option+C 当组合键，取消录音",
            {"keycode": R_OPT},
            lambda kb: (kb.mod(R_OPT, True), kb.tap(C), kb.mod(R_OPT, False)),
            (1, 0, []),
        ),
        ("快捷键右 Option，按左 Option", {"keycode": R_OPT}, lambda kb: (kb.mod(L_OPT, True), kb.mod(L_OPT, False)), (0, 0, [])),
        (
            "右 Ctrl+K，带重复按下",
            rctrl_k,
            lambda kb: (kb.mod(R_CTRL, True), kb.key(K), kb.key(K), kb.key(K, False), kb.mod(R_CTRL, False)),
            (1, 1, ["k40+", "k40+", "k40-"]),
        ),
        (
            "右 Ctrl+K，先松开右 Ctrl",
            rctrl_k,
            lambda kb: (kb.mod(R_CTRL, True), kb.key(K), kb.mod(R_CTRL, False), kb.key(K, False)),
            (1, 1, ["k40+", "k40-"]),
        ),
        ("快捷键右 Ctrl+K，单独按 K", rctrl_k, lambda kb: kb.tap(K), (0, 0, [])),
        (
            "快捷键右 Ctrl+K，按左 Ctrl+K",
            rctrl_k,
            lambda kb: (kb.mod(59, True), kb.tap(K), kb.mod(59, False)),
            (0, 0, []),
        ),
        (
            "快捷键右 Ctrl+K，按右 Ctrl+右 Shift+K",
            rctrl_k,
            lambda kb: (kb.mod(R_CTRL, True), kb.mod(R_SHIFT, True), kb.tap(K), kb.mod(R_SHIFT, False), kb.mod(R_CTRL, False)),
            (0, 0, []),
        ),
    ]
    for name, config, steps, expected in cases:
        got = run(config, steps)
        assert got == expected, f"{name}: 期望 (开麦, 粘贴, 拦下) = {expected}，实际 {got}"
        print(f"ok  {name}")


if __name__ == "__main__":
    set_cases()
    runtime_cases()
    print("全部通过")
