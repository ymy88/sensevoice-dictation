# sensevoice-dictation

在 macOS 上按住一个键说话，松开后文字出现在光标位置。语音识别用 [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice) 在本机完成，不联网。

Hold a key to dictate on macOS. Speech is transcribed locally by SenseVoice-Small. Mixed Chinese and English work well.

## 特点

- **中英混说**：「把这个 component 的 props 改成 optional」这类句子，英文单词保持英文。
- **本地识别**：音频不离开本机。
- **快**：在 Apple Silicon 上，8 秒的语音识别用时约 0.13 秒。
- **任何输入法下都能用**：结果通过剪贴板粘贴，拼音输入法开着时英文也不会被当成拼音。
- **在任何 App 里都能用**：终端、编辑器、浏览器、聊天软件都可以。
- **没说话就不输出**：按住快捷键但没说话时，程序跳过识别。

默认快捷键是 **Insert**（PC 键盘）。Mac 键盘没有 Insert，见下文「换快捷键」。

## 要求

- macOS，Apple Silicon（在 macOS 26 上测试过）
- [uv](https://docs.astral.sh/uv/)
- Xcode 命令行工具（提供 `clang`，只有后台运行需要）：`xcode-select --install`

## 安装

```bash
git clone https://github.com/ymy88/sensevoice-dictation.git
cd sensevoice-dictation
uv sync

# 下载模型。压缩包里有 fp32 和 int8 两个版本，程序只用 240MB 的 int8 版本。
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xjf sherpa-onnx-sense-voice-*.tar.bz2 && rm sherpa-onnx-sense-voice-*.tar.bz2
rm sherpa-onnx-sense-voice-*/model.onnx
cd ..
```

确认模型能用：

```bash
uv run python main.py models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
# 开饭时间早上9点至下午5点。
```

带一个 wav 文件参数时，程序只识别这个文件并打印结果，不监听键盘。wav 文件必须是 16kHz、16 位。

## 先在终端里试一下

```bash
uv run python main.py
```

按住 Insert 说话，松开后文字会粘贴到当前光标位置，终端里也会打印 `> 识别结果`。

macOS 会要求给终端 App 授权：

- **辅助功能**：用来监听快捷键、拦下快捷键、发送 Cmd+V。在「系统设置 → 隐私与安全性 → 辅助功能」里打开你的终端 App。
- **麦克风**：第一次录音时弹框，点允许。

授权后要完全退出终端（Cmd+Q）再重新打开。

## 后台运行，开机自动启动

```bash
./app/build.sh
cp build/local.sensevoice-dictation.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.sensevoice-dictation.plist
```

`app/build.sh` 生成两样东西：

- `build/SenseVoice Dictation.app`：一个没有界面的 App，负责启动 `main.py`。
- `build/local.sensevoice-dictation.plist`：launchd 的配置文件，路径按项目当前位置填好。launchd 在登录时启动 App，App 退出后会自动再启动。

然后在「系统设置 → 隐私与安全性 → 辅助功能」里打开 `SenseVoice Dictation`。列表里没有的话，点「+」，选 `build/SenseVoice Dictation.app`。授权后重启服务：

```bash
launchctl kickstart -k gui/$(id -u)/local.sensevoice-dictation
```

第一次按住快捷键时，macOS 可能会询问 `SenseVoice Dictation` 能否使用麦克风。

为什么要包一个 App：权限框和设置里显示的是 `SenseVoice Dictation`，而不是 `python3.13`。权限也只给这个 App。如果直接给 `python3.13` 授权，同一个 Python 跑的任何脚本都能监听和模拟按键。App 用子进程运行 Python，macOS 把子进程的权限算在 App 上。

常用命令：

| 做什么 | 命令 |
|---|---|
| 看日志 | `tail -f ~/Library/Logs/sensevoice-dictation.log` |
| 重启（改了 `main.py` 之后） | `launchctl kickstart -k gui/$(id -u)/local.sensevoice-dictation` |
| 停用 | `launchctl bootout gui/$(id -u)/local.sensevoice-dictation` |
| 卸载 | 先停用，再删除 `~/Library/LaunchAgents/local.sensevoice-dictation.plist` |

改了 `app/` 下的文件后，要重跑 `./app/build.sh`。App 用的是本机临时签名，重新生成后 macOS 会把它当成新 App，辅助功能权限要重新打开一次。项目换了位置也要重跑 `./app/build.sh`，再按上面的步骤重新安装 plist。

在前台调试前，先停掉后台服务。两个进程同时运行时，同一段话会被粘贴两次。

## 换快捷键

```bash
uv run python main.py --set-hotkey
```

按一下想用的键，程序把它存进项目里的 `config.json`，后台服务在运行的话会自动重启。按 Esc 取消。没有 `config.json` 时用 Insert。

可以用两类键：

- **功能键**：Insert、F13 到 F19 等。程序会拦下这个键，当前 App 收不到它。
- **修饰键**：右 Option、右 Cmd、右 Control、右 Shift、Fn，左边的也可以。Mac 键盘没有 Insert，推荐用右 Option 或右 Cmd。

修饰键不会被拦下，因为拦下后，这个键的组合键（比如右 Cmd+C）就不能用了。所以修饰键的规则是：

- 单独按住时录音，松开后识别。
- 按住期间又按了别的键，程序当作组合键，取消这次录音。
- 快速按一下（短于 0.3 秒）不识别。

字母、数字、符号、空格、回车、Tab、删除和方向键不能设为快捷键，程序会提示换一个键。

用 Fn 时，先在「系统设置 → 键盘」里把「按下 🌐 键时」设为「无操作」。否则每次按住 Fn，macOS 还会弹出表情面板或切换输入法。

## 怎么工作

1. 程序通过 macOS 的按键事件监听快捷键。按下快捷键时打开麦克风，松开时关闭。麦克风只在按住期间打开。
2. 录音里最响的 30 毫秒片段，RMS 低于 `SILENCE_RMS` 时，程序认为没人说话，跳过识别。不跳过的话，SenseVoice 会把静音识别成「Okay.」之类的词。
3. sherpa-onnx 运行 SenseVoice-Small int8 模型，自动判断语言，输出带标点的文字。
4. 程序把文字写进剪贴板，发送 Cmd+V，等 0.3 秒，再把剪贴板恢复成原来的文字。

## 排查

| 现象 | 原因和办法 |
|---|---|
| 按 Insert 时，终端里出现 `^[[2~` | 按键监听没起作用。检查辅助功能权限，然后重启程序或服务。 |
| 日志里有 `This process is not trusted!` | 辅助功能权限没给，或者给了但程序还没重启。 |
| 日志里有「没检测到声音」，RMS 是 0.0000 | 没有麦克风权限。macOS 拒绝麦克风时不报错，只给全零的音频。在「隐私与安全性 → 麦克风」里打开。 |
| 日志里有「没检测到声音」，RMS 不是 0 | 声音太小。调低 `main.py` 里的 `SILENCE_RMS`。 |
| 刚启动时按快捷键没反应 | 模型加载要几秒。日志出现「按住 … 说话」后再试。 |
| 用修饰键当快捷键时，有时不录音 | 按住期间碰到了别的键，程序当作组合键取消了录音。日志里会有「按了组合键，取消录音」。 |

## 已知限制

- 粘贴后，程序只恢复剪贴板里原来的纯文本。剪贴板里原来是图片或文件时，这些内容会被清掉。
- 只支持 macOS。

## 许可

代码使用 MIT 许可，见 [LICENSE](LICENSE)。

SenseVoice 模型需要单独下载。使用模型时要遵守 [SenseVoice 的模型许可](https://github.com/FunAudioLLM/SenseVoice)。
