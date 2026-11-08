"""合盖时换麦克风的自检：uv run python test_mic.py"""

from main import fallback_input


def dev(index, name, ins, outs):
    return {"index": index, "name": name, "max_input_channels": ins, "max_output_channels": outs}


# 2026-11-08 这台机器上的设备列表
DEVICES = [
    dev(0, "“Ymy_iPhone”的麦克风", 1, 0),
    dev(1, "LG UltraFine显示屏音频", 0, 2),
    dev(2, "MacBook Pro麦克风", 1, 0),
    dev(3, "MacBook Pro扬声器", 0, 2),
    dev(4, "漫步者•花再 Zero Air", 1, 0),
    dev(5, "漫步者•花再 Zero Air", 0, 2),
    dev(6, "ZoomAudioDevice", 2, 2),
]

assert fallback_input(DEVICES) == 4
# 耳机没连：没有能换的麦克风，用默认的
assert fallback_input([d for d in DEVICES if d["index"] not in (4, 5)]) is None
print("ok")
