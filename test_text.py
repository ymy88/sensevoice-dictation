"""识别结果后处理的自检：uv run python test_text.py"""

import numpy as np

import main
from main import join_spelled_letters

CASES = {
    "怎么用 G P U 跑模型": "怎么用 GPU 跑模型",
    "调 a p i 接口": "调 api 接口",
    "用 GPU 跑": "用 GPU 跑",  # 本来就连着的不动
    "M 2484已经合并": "M 2484已经合并",  # 单个字母后面跟数字，不合并
    "G P Unit": "GP Unit",  # 后面是完整单词，只合并前面的单字母
    "new UI 已经同步": "new UI 已经同步",
    "A，B 两个方案": "A，B 两个方案",  # 隔着标点，不合并
}

for raw, expected in CASES.items():
    got = join_spelled_letters(raw)
    assert got == expected, f"{raw!r}: 期望 {expected!r}，实际 {got!r}"

REPLACE_CASES = {
    "控制我本机的 cloud code 或者": "控制我本机的 Claude Code 或者",
    "用 Cloud Code 跑": "用 Claude Code 跑",
    "在 Google Cloud 上": "在 Google Cloud 上",  # 单独的 cloud 不动
    "icloud code": "icloud code",  # 只换完整的词
}
for raw, expected in REPLACE_CASES.items():
    got = main.replace_words(raw)
    assert got == expected, f"{raw!r}: 期望 {expected!r}，实际 {got!r}"

# 长录音切段：每段不超过 CHUNK_SECONDS，切点落在停顿里，拼回去和原录音一样
sec = main.SAMPLE_RATE
speech = lambda s: np.full(int(s * sec), 0.1, dtype=np.float32)
silence = lambda s: np.zeros(int(s * sec), dtype=np.float32)
audio = np.concatenate([speech(24), silence(0.5), speech(20), silence(0.5), speech(15)])  # 停顿在 24 秒和 44.5 秒
chunks = main.split_long(audio)
cuts = np.cumsum([len(c) for c in chunks])[:-1] / sec
assert len(cuts) == 2 and 24 <= cuts[0] <= 24.5 and 44.5 <= cuts[1] <= 45, cuts
assert np.array_equal(np.concatenate(chunks), audio)
assert len(main.split_long(speech(29))) == 1
assert all(len(c) <= main.CHUNK_SECONDS * sec for c in main.split_long(speech(100)))  # 没有停顿也照样切
print("全部通过")
