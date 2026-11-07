#!/bin/sh
# 生成 build/SenseVoice Dictation.app 和 build/local.sensevoice-dictation.plist（路径按项目当前位置填）。
# 改了 launcher.c 或 Info.plist 后重跑；重新签名后 macOS 可能要求重新授权。
set -e
cd "$(dirname "$0")/.."
APP="build/SenseVoice Dictation.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp app/Info.plist "$APP/Contents/"
clang -O2 -DPROJECT_DIR="\"$PWD\"" -o "$APP/Contents/MacOS/sensevoice-dictation" app/launcher.c
codesign --force --sign - --identifier local.sensevoice-dictation "$APP"
sed -e "s|@APP_EXEC@|$PWD/$APP/Contents/MacOS/sensevoice-dictation|" \
    -e "s|@PROJECT_DIR@|$PWD|" \
    -e "s|@LOG@|$HOME/Library/Logs/sensevoice-dictation.log|" \
    app/launchd.plist.in > build/local.sensevoice-dictation.plist
plutil -lint build/local.sensevoice-dictation.plist
echo "built $APP"
