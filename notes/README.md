# 具身强化学习知识点

配合 [学习路线](../README.md) 使用，记录已经学习的内容。后续按学习顺序增加笔记。

**当前主目标：尽快跑通 RL4VLA 单任务 PPO 的采样、更新、保存与独立评估。** CartPole 一次更新已完成核对；后续讲义围绕 RL4VLA 实际训练入口补齐所需概念，按 [README 中的当前执行优先级](../README.md) 推进。

在 [学习工作台](../index.html) 的“学习笔记”中可以直接阅读完整笔记，并分别记录阅读和代码练习进度。修改本目录的 Markdown 后，运行 `python scripts/build_notes.py` 同步网页正文；构建依赖见项目 README。

## 阅读顺序

1. [奖励与回报](01-reward-and-return.md)：VLA 中的 RL 概念、折扣回报、时间下标和计算例子。
2. [训练时如何计算回报](02-computing-returns.md)：完整轨迹、反向递推、自举、终止与截断。
3. [状态价值函数](03-state-value-function.md)：函数的直觉、实际回报与期望回报、价值网络的训练，以及如何用于自举。
4. [动作价值与优势](04-action-value-and-advantage.md)：V 与 Q 的区别、优势的正负含义，以及真实优势与样本估计。
5. [策略如何学习](05-how-policy-learns.md)：动作概率、优势指导的更新方向、基础策略损失，以及可运行的最小参数更新例子。
6. [认识 Gymnasium 与 CartPole](06-gymnasium-and-cartpole.md)：手写代码前的任务介绍、观测和动作、奖励、结束条件，以及环境接口。
7. [策略网络与动作分布](07-policy-network-and-distribution.md)：logits、softmax、Categorical 采样，以及从四个观测生成两个动作分数。
8. [第一次策略更新](08-first-policy-update.md)：完整采样、保存 log 概率、手写回报递推，以及 REINFORCE 的一次参数更新。
9. [Actor-Critic 与 TD 误差](09-actor-critic-and-td.md)：价值网络、一步自举目标、TD 残差，以及 RL4VLA 源码中的对应位置。
10. [GAE：广义优势估计](10-generalized-advantage-estimation.md)：多步 TD 误差加权、lambda、反向递推与 RL4VLA 的优势及价值目标。

当前学习 **10 · GAE**，下一课为 **11 · PPO**。每课先理解，再由你手写一个最小核对实验，并对应到 RL4VLA 实际源码。第 11 课尚待编写。

## 配套代码练习

先按 [代码练习指南](../practice/README.md) 运行、修改并解释结果，再继续增加理论知识。每次先写下预测，再运行命令核对。

| 笔记 | 对应练习 | 重点阅读的函数 |
| --- | --- | --- |
| 01–02 · 奖励、回报、自举 | 练习 1–2：查看 rollout、计算回报 | `collect_episode`、`discounted_returns` |
| 03 · 状态价值函数 | 练习 3：训练一个价值预测 | `run_value` |
| 04 · 动作价值与优势 | 练习 1、4：找出 G、V、A 并解释正负号 | `run_rollout`、`run_update` |
| 05 · 策略如何学习 | 练习 4–6：单次更新、完整训练、冻结策略对照 | `policy_gradient`、`update`、`run_train` |

## 统一符号约定

本目录采用“执行第 $t$ 步动作后获得的奖励记作 $r_t$”的约定：

$$
s_t \xrightarrow{a_t,\;r_t} s_{t+1}
$$

| 符号 | 含义 |
| --- | --- |
| $s_t$ | 第 $t$ 步的环境状态 |
| $o_t$ | 第 $t$ 步策略获得的观测，不一定包含完整状态 |
| $a_t$ | 第 $t$ 步执行的动作 |
| $r_t$ | 执行 $a_t$ 后，环境在这次转移中给出的奖励 |
| $\pi_\theta$ | 参数为 $\theta$ 的策略 |
| $\pi_\theta(a\mid s)$ | 本目录的离散动作例子中，策略在状态 $s$ 下选择动作 $a$ 的概率 |
| $\gamma$ | 折扣因子 |
| $G_t$ | 从第 $t$ 步开始的回报 |
| $T$ | 一条完整轨迹的动作总数，最后一个动作下标为 $T-1$ |
| $s_T$ | 真正终止后的状态 |
| $V^\pi(s)$ | 从状态 $s$ 开始，按策略 $\pi$ 行动时的期望回报 |
| $V_\phi(s)$ | 参数为 $\phi$ 的价值函数近似 |
| $Q^\pi(s,a)$ | 在状态 $s$ 下先执行动作 $a$，再按策略 $\pi$ 行动时的期望回报 |
| $A^\pi(s,a)$ | 动作优势，定义为 $Q^\pi(s,a)-V^\pi(s)$ |
| $\hat A_t$ | 第 $t$ 步所采动作的优势估计 |

有些教材把执行 $a_t$ 后的奖励记作 $r_{t+1}$，于是回报从 $r_{t+1}$ 开始写。两种约定都成立，但同一推导中不能混用。

## 公式格式

笔记使用 Markdown 的 LaTeX 数学语法：行内公式使用 `$...$`，独立公式使用 `$$...$$`。预览器需要支持数学渲染；纯文本编辑器会显示公式源码。
