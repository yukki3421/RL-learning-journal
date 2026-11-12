# 从零手写 RL

**当前总体目标是尽快跑通 RL4VLA 单任务 PPO 闭环。** 基础阶段已完成一次 REINFORCE 更新核对；接下来主要沿 RL4VLA 的环境、基线、训练、保存与独立评估推进。工程顺序与资源记录见 [项目当前主目标](../README.md#当前主目标与执行优先级2026-11-12-更新)。Critic、GAE 和 PPO 的补课围绕实际训练源码进行。

**当前课程：**已通过“参数更新与重新计算网络输出”的概念核对，进入 [10 · GAE](../notes/10-generalized-advantage-estimation.md)。先理解讲义，再由你自己创建 `handson/05_gae.py`，手写一次优势和价值目标的反向递推。完成后直接接 PPO，并读 RL4VLA 的相应源码。

这里是你亲手实现的练习区。先说明代码与知识点的关系，你写代码并运行，再根据实际输出继续。从第 8 课起加快节奏，每轮完成一个完整环节，通过断点检查关键数据，而不再逐行暂停。

## 环境

在仓库根目录使用项目的 uv 环境：

```bash
uv --cache-dir .uv-cache sync --locked
uv run python handson/01_cartpole.py
```

Gymnasium 提供 CartPole 环境，PyTorch 提供张量、网络、自动求导与优化器。当前使用 CPU 版 PyTorch，足够完成这个小环境的训练。已安装 classic-control 渲染依赖和 Pillow，可离屏生成动画；远程终端不需要图形桌面或 GPU。

uv 缓存放在项目下的 `.uv-cache/`，避免占用系统盘；它和 `.venv/` 一样不提交到 Git。

环境 API 的依据见 [Gymnasium 基础用法](https://gymnasium.farama.org/introduction/basic_usage/)；安装来源见 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。具体依赖版本以本仓库的 `uv.lock` 为准。

## 我们会亲手完成哪些部分？

| 顺序 | 要实现的内容 | 对应知识点 |
| --- | --- | --- |
| 1，已完成 | 创建环境、reset、查看观测和动作空间 | 状态与观测、动作、episode 起点 |
| 2，当前 | 先手动指定一个动作并执行 step，再扩展成随机轨迹采集 | 奖励、rollout、终止与截断 |
| 3，待开始 | 从奖励反向计算回报 | 折扣回报与递推 |
| 4，开始学习分布 | 用 PyTorch 写策略网络、抽取动作；先在独立文件认识 logits 与采样 | 动作概率、策略参数 |
| 5，待开始 | 保存 log-prob、构造损失并反向传播 | 策略梯度与 log 概率 |
| 6，待开始 | 训练循环、保存模型、独立评估 | 采样与更新、训练前后比较 |
| 7，后续 | 增加价值预测与优势估计 | 价值函数与 Actor-Critic |

先实现基本策略梯度，再逐步加入价值基准、GAE 与 PPO。遇到尚未学过的内容，先通过小例子补齐。

## 第一步：环境初始化，已完成

先阅读 [认识 Gymnasium 与 CartPole](../notes/06-gymnasium-and-cartpole.md)，理解任务与接口后，再补代码。

文件：[01_cartpole.py](01_cartpole.py)。初始化代码已由你写好并运行，代码注释补充了观测、空间与动作编号的含义。

先理解任务：CartPole 中，小车左右移动，目标是让杆保持直立。你将看到的是四个数的观测，而不是图像。

已经完成的操作：

1. 用 `gym.make("CartPole-v1")` 创建环境，保存到 `env`。
2. 用 `env.reset(seed=7)` 开始一次 episode，分别接收 `observation` 和 `info`。
3. 打印观测值 `observation`、观测空间 `env.observation_space`、动作空间 `env.action_space`。
4. 调用 `env.close()` 释放环境资源。

`reset()` 对应一次任务的开始。`seed=7` 用于让初始化可重复，不代表任务状态编号是 7。

已核对输出：观测包含四个数，动作空间为 `Discrete(2)`。

## 第二步：亲手执行一次动作，已完成

在 `env.close()` 之前补全新 TODO：

1. 指定 `action = 0`，这一模拟步向左推小车。
2. 调用 `env.step(action)`，依次接收 `next_observation`、`reward`、`terminated`、`truncated`、`info`。
3. 打印所选动作、新观测、本步奖励和两个结束标记。

对应关系是：已有 `observation` → 执行 `action` → 获得 `reward` 和 `next_observation`。默认奖励为 1，也包含导致终止的那一步；两个结束标记分别保留，不提前合并丢弃。

仍使用 `uv run python handson/01_cartpole.py` 运行。先解释一次转移，下一步再加入循环收集完整 episode。

## 已完成：固定动作运行到 episode 结束

你已把单步交互扩展成循环。`seed=7`、一直向左推时，第 8 步杆角度约 11.13 度，第 9 步约 13.96 度，超过 12 度失败阈值，故在第 9 步真正终止。触发终止的最后一步也计数、也获得默认奖励 1。

循环结束后使用已有的 `observation`、`terminated`、`truncated` 查看最终结果。不要保留旧的单步示例再次调用 `step()`；需要重新开始时先 `reset()`。

## 看见发生了什么：离屏渲染

```bash
uv run python handson/render_cartpole.py
```

辅助工具按同样的 seed 和固定动作重新运行真实环境，生成：

- `runs/cartpole-first/episode.gif`：包括初始帧和全部 9 步，按 10 倍慢放循环播放。
- `runs/cartpole-first/keyframes.png`：关键帧静态对照。
- `runs/cartpole-first/trajectory.csv`：每一步的位置、速度、角度、奖励和结束标记。

在编辑器的远程文件列表中打开 GIF / PNG，或者下载到自己的电脑查看。这个工具只负责观察环境，不包含策略训练；学习主线仍在你手写的 `01_cartpole.py` 中。

需要实时弹出窗口时，可使用 `render_mode="human"`，但远程服务器必须有可用的图形显示或显示转发；当前优先使用无需桌面的 `rgb_array` 输出。

后续代码会在这个文件中逐步扩展；这里不预先给出完整训练实现。

## 新课：策略网络与动作分布

你已经完成固定策略交互和九步轨迹记录。阅读 [第 7 课](../notes/07-policy-network-and-distribution.md)，再补全 [02_policy.py](02_policy.py) 中的四个 TODO，先理解 logits、概率和采样，再接入网络。

```bash
uv run python handson/02_policy.py
```

回报函数仍需要在后续训练前亲手补齐。这里先用独立分布实验认识策略的输出形式，不会提前替你完成训练实现。

## Python 调试

已配置 VS Code / Remote SSH 的 `.venv` 调试入口，也可使用终端 pdb。具体断点位置和变量观察方法见 [调试指南](DEBUG.md)。

## 已完成的基础里程碑：第一次参数更新

入口：[03_update_once.py](03_update_once.py)，讲义：[第 8 课：第一次策略更新](../notes/08-first-policy-update.md)。

你已在 `02_policy.py` 完成前向计算、采样与 log 概率观察。本轮直接将策略接入完整交互循环，然后补齐回报递推和一次参数更新。文件保留 TODO，由你手写；讲义分成采样、回报、损失与更新三个代码块。

验收输出：总步数与 log 概率数量相同，回报首尾值合理，loss 有限，参数最大变化大于零。这里先只验收计算闭环，下一轮才扩展训练、保存和独立评估。

```bash
uv run python handson/03_update_once.py
```
