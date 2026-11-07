"""识别结果后处理的自检：uv run python test_text.py"""

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
print("全部通过")
