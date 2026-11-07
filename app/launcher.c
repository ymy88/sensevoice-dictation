// 「SenseVoice Dictation.app」的可执行文件。
// 用子进程跑 main.py，而不是 exec：macOS 把子进程的麦克风、输入监控、辅助功能权限算在这个 App 头上，
// 权限框显示 App 名字，权限也不会给到共用的 python3.13。exec 会把进程身份换成 python3.13。
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;
static pid_t child;

static void forward(int sig) { kill(child, sig); }

int main(void) {
    char *argv[] = {PROJECT_DIR "/.venv/bin/python", "-u", PROJECT_DIR "/main.py", NULL};
    if (posix_spawn(&child, argv[0], NULL, NULL, argv, environ) != 0) return 127;
    signal(SIGTERM, forward);  // launchd 停服务时只给这个进程发 SIGTERM，要转给 Python，否则 Python 会留在后台
    signal(SIGINT, forward);
    int status;
    while (waitpid(child, &status, 0) < 0) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
