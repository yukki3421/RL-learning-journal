# 调试手写 RL 代码

项目已安装 `debugpy` 开发依赖，VS Code 配置使用 `${workspaceFolder}/.venv/bin/python`。请以本仓库为工作区根目录；远程连接时，在服务器端安装编辑器推荐的 Python 和 Python Debugger 扩展。

## 第一次打断点

1. 打开 `01_cartpole.py`，在 `observation, info = env.reset(seed=7)` 左侧点击行号旁的空白处，出现红色断点。
2. 打开“运行和调试”，选择 `RL：调试手写 CartPole（入口暂停）`，按 F5。
3. 在入口停住后按 F5，继续到断点。黄色行表示即将执行的语句；停在 `reset` 这一行时，赋值还没有发生。
4. 按 F10 执行这一行，再在变量面板或 Debug Console 查看 `observation`。这时应有四个初始数值。
5. 在 `trajectory.append(...)` 处再打一个断点，观察动作前的 `observation` 与动作后的 `next_observation`。
6. 再执行 `observation = next_observation`，查看当前观测如何更新。

## 常用操作

| 操作 | 默认快捷键 | 用途 |
| --- | --- | --- |
| 继续运行 | F5 | 到下一个断点 |
| 单步跳过 | F10 | 执行当前行，不展开函数内部 |
| 单步进入 | F11 | 进入函数内部 |
| 单步跳出 | Shift+F11 | 跑完当前函数，回到调用处 |
| 停止调试 | Shift+F5 | 结束当前调试 |

快捷键可能受操作系统或个人绑定影响，调试工具栏中也有对应按钮。

可在“监视”面板添加：`observation`、`next_observation`、`len(trajectory)`、`step_count`、`reward`、`terminated`、`truncated`。变量需要在对应语句执行之后才存在。

要只看第 9 步，可在循环中 `step_count += 1` 之后的语句设置条件断点：`step_count == 9`。

## 进入第三方环境源码

选择 `RL：进入 Gymnasium 源码`，在 `env.step(action)` 处按 F11。你会先进入环境包装器，再逐层到达 CartPole 的物理状态更新和阈值判断。该配置设置了 `justMyCode=false`。

## 终端启动，再连接调试器

```bash
uv run python -m debugpy --listen 127.0.0.1:5678 --wait-for-client handson/01_cartpole.py
```

这个命令会等待调试器连接，然后选择 `RL：连接本机 debugpy :5678` 并按 F5。在 Remote SSH / 容器工作区中，终端与调试扩展应运行在同一远程环境；如果编辑器只在本地运行，需要转发服务器的 5678 端口。

仅在普通终端中调试，也可以使用 Python 自带的 pdb：

```bash
uv run python -m pdb handson/01_cartpole.py
```

常用命令：`n` 执行下一行，`s` 进入函数，`p observation` 打印变量，`c` 继续，`q` 退出。

配置参考：[VS Code Python 调试](https://code.visualstudio.com/docs/python/debugging)、[debugpy 官方命令行说明](https://github.com/microsoft/debugpy/wiki/Command-Line-Reference)。
