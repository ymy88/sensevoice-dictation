# sensevoice-dictation

在 macOS 上按住右 Option 说话，松开后用 SenseVoice-Small 在本地把语音转成文字，再粘贴到当前光标位置。支持中英混说，不联网。

## 安装

```bash
uv sync
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xjf sherpa-onnx-sense-voice-*.tar.bz2 && rm sherpa-onnx-sense-voice-*.tar.bz2
rm sherpa-onnx-sense-voice-*/model.onnx   # 只用 int8 版本，fp32 版本 900MB 用不到
```

## 运行

```bash
uv run python main.py
```

第一次运行时，要在「系统设置 → 隐私与安全性」里给运行这个程序的终端 App 打开三项权限：

- 麦克风：录音
- 输入监控：监听右 Option
- 辅助功能：发送 Cmd+V

授权后要完全退出终端（Cmd+Q）再重新打开。

## 自检

```bash
uv run python main.py models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
```

只识别这个 16kHz 的 wav 文件并打印结果，不监听键盘。

## 已知限制

- 文字通过剪贴板粘贴。粘贴后程序只恢复原来的纯文本剪贴板内容；原来是图片或文件时会被清掉。
