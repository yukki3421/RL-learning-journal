# 10 · GAE：把多步 TD 误差组合成优势估计

前置知识：[Actor-Critic 与 TD 误差](09-actor-critic-and-td.md)。

**目标：读懂 RL4VLA 的 `compute_returns_ppo()`。** 本课先解释 GAE，再由你手写一个短函数核对数值；下一课进入 PPO。学习代码仍由你自己编写和修改。

## 1. GAE 是什么？

**GAE 是 Generalized Advantage Estimation，中文叫“广义优势估计”。它把当前及后续多步的 TD 误差按权重组合，得到当前动作的优势估计。**

第 9 课使用一个 TD 误差作为一步优势估计：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

这个单步信号包含环境随机性和价值预测误差。GAE 增加一个参数 $\lambda$，让我们调节后续多步 TD 信号参与当前估计的程度。

GAE 负责构造优势估计，不负责选择动作，也不会自己更新网络参数。PPO 会使用它生成的优势构造策略损失。

## 2. 先读懂这一个公式

为避免边界混淆，先讨论一条内部没有 episode 重置、包含 $H$ 次转移的片段。价值预测在采样时记录下来，计算 GAE 时保持固定。

$$
\hat A_t^{\mathrm{GAE}}
=\delta_t+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots
+(\gamma\lambda)^{H-t-1}\delta_{H-1}
$$

逐项读：

- 当前这步的 TD 误差，权重为 1。
- 后一步的 TD 误差，权重为 $\gamma\lambda$。
- 再后一步的 TD 误差，权重为 $(\gamma\lambda)^2$。

权重以**当前第 $t$ 步**为参照。这里的 $(\gamma\lambda)^k$ 用来组合相隔 $k$ 步的 TD 误差，不是第 8 课损失外面那个从 episode 起点计数的 $\gamma^t$。

最终得到的仍然是“这个动作相对当前策略平均水平好多少”的估计，不是 ground truth，也不是任务成功概率。

## 3. lambda 控制什么？

通常取 $0\leq\lambda\leq1$。

| 取值 | 得到什么 | 含义 |
| --- | --- | --- |
| $\lambda=0$ | $\hat A_t=\delta_t$ | 只用一步 TD 信号 |
| $0<\lambda<1$ | 多步 TD 残差的加权组合 | 调节较远信号的参与程度 |
| $\lambda=1$，完整且真正终止的轨迹 | $\hat A_t=G_t-V(s_t)$ | 化为完整蒙特卡洛回报减去价值预测 |

通常，较小的 $\lambda$ 更依赖较近的自举预测，价值预测误差带来的偏差可能更明显；较大的 $\lambda$ 纳入更长的实际后续信号，但采样波动也可能更大。这是常说的偏差与方差权衡，并不保证某个取值在所有任务上最好。

$\gamma$ 描述奖励的时间折扣，$\lambda$ 调节优势估计方式。改变 $\lambda$ 不会改变环境的奖励规则。

## 4. 手算一个三步例子

假设任务在第三次动作之后真正终止：

```text
状态：s0 → s1 → s2 → 终止
奖励：  0    0    1
```

固定的价值预测为：

$$
V(s_0)=0.5,\quad V(s_1)=0.6,\quad
V(s_2)=0.7,\quad V(s_3)=0
$$

这里末尾真正终止，所以剩余价值为零。为了便于手算，取 $\gamma=0.9$、$\lambda=0.8$，于是 $\gamma\lambda=0.72$。

先算每一步的 TD 误差：

$$
\begin{aligned}
\delta_0&=0+0.9\times0.6-0.5=0.04,\\
\delta_1&=0+0.9\times0.7-0.6=0.03,\\
\delta_2&=1+0.9\times0-0.7=0.3.
\end{aligned}
$$

再从后向前组合：

$$
\begin{aligned}
\hat A_2&=0.3,\\
\hat A_1&=0.03+0.72\times0.3=0.246,\\
\hat A_0&=0.04+0.72\times0.246=0.21712.
\end{aligned}
$$

因此：

```text
TD 误差：[0.04, 0.03, 0.3]
GAE 优势：[0.21712, 0.246, 0.3]
```

本例中，后面的正 TD 信号也参与了前面动作的优势估计。每个样本的估计仍会受到价值预测误差影响。

## 5. 最适合写代码的反向递推

上一节其实已经用了递推形式：

$$
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}
$$

因为 $\hat A_{t+1}$ 已经包含更后面的所有已采到的 TD 信号，所以无需每一步再写一遍长求和。

片段末尾设 $\hat A_H=0$，表示后面没有更多**已采到的 TD 误差项**参与这次有限片段递推。**这不等于要求 $V(s_H)=0$。**

- 真正终止：末尾价值为零。
- 只是采样片段截止，任务仍可继续：末尾价值使用 Critic 预测，已经进入最后一步的 $\delta_{H-1}$。

例如只有一个未终止转移，$r=0$、$V(s_0)=0.6$、$V(s_1)=0.8$、$\gamma=0.9$，则：

$$
\hat A_0=\delta_0=0+0.9\times0.8-0.6=0.12.
$$

即使递推边界优势为零，末尾预测 0.8 仍然参与了计算。

## 6. 优势与 Critic 训练目标的关系

常见做法是先用未标准化的 GAE 优势构造价值训练目标：

$$
\hat R_t=\hat A_t^{\mathrm{GAE}}+V(s_t)
$$

上一节的三步例子得到：

```text
原价值预测：  [0.5,     0.6,   0.7]
GAE 优势：   [0.21712, 0.246, 0.3]
价值训练目标：[0.71712, 0.846, 1.0]
```

这些是估计目标，不是完整实际回报。这个例子的蒙特卡洛回报为 `[0.81, 0.9, 1.0]`；只有在这里令 $\lambda=1$，GAE 加原价值才会化为这组完整回报。

| 量 | 给谁用？ |
| --- | --- |
| 优势估计 $\hat A_t$ | 给 Actor 构造策略损失 |
| 价值目标 $\hat R_t$ | 给 Critic 构造价值损失 |

如果还要标准化优势，先保留上述价值目标，再另外处理策略使用的优势。不要把标准化后的优势加回价值，当成同一个价值目标。

## 7. 对应 RL4VLA 的哪些代码？

固定版本：`de46e0174e7ac310255a2836718fcefab28b8963`。

查看 `SimplerEnv/simpler_env/utils/replay_buffer.py` 的 `compute_returns_ppo()`：

1. 取当前和下一步的 `value_preds`。
2. 根据奖励和结束掩码计算 `delta`。
3. 用 `delta` 与上一轮反向递推得到的 `gae` 继续递推。
4. 用原始 `gae + 当前价值` 写入 `returns`，作为价值训练目标。
5. 得到原始优势后，另外进行批内标准化。

源码中掩码参与下一价值项和 GAE 的跨步递推，用于切断相应边界。具体边界含义还要追到 `Runner.insert()` 和 `SimlerWrapper.step()`：本版本后者向训练侧返回的结束标记来自 `truncated`，不能把该实现泛化成所有环境的终止处理规则。

上游参数默认采用 $\gamma=0.99$、$\lambda=0.95$。本课用 $0.9$、$0.8$ 只是为了手算清楚，不是在更改 RL4VLA 默认配置。

## 8. 你来写的代码练习

自己新建 `handson/05_gae.py`，写一个 `compute_gae(rewards, values, gamma, gae_lambda)` 函数。

输入约定：

- `rewards` 长度为 $H$。
- `values` 长度为 $H+1$，多出来的一项是片段末尾价值。
- 单次调用只处理一条内部没有 episode 重置的连续片段。
- 调用者根据任务目标设置末尾价值：真正终止为 0；尚未结束、需要自举时传入末尾预测。
- 外部时间截断仍按第 9 课所讲的目标定义处理，不能仅凭“截断”一词决定置零。

实现顺序：

```text
创建 advantages 列表，初值全为 0
设置 gae = 0
从最后一步往前遍历
    算当前 delta
    gae = delta + gamma × gae_lambda × gae
    把 gae 写入当前 advantages
算 value_targets = advantages + 对应的原价值预测
返回 advantages 和 value_targets
```

先用以下输入核对：

```python
rewards = [0.0, 0.0, 1.0]
values = [0.5, 0.6, 0.7, 0.0]
```

验收三种情况：

| gamma | lambda | 优势预期值 | 价值目标预期值 |
| --- | --- | --- | --- |
| 0.9 | 0.8 | `[0.21712, 0.246, 0.3]` | `[0.71712, 0.846, 1.0]` |
| 0.9 | 0 | `[0.04, 0.03, 0.3]` | `[0.54, 0.63, 1.0]` |
| 0.9 | 1 | `[0.31, 0.3, 0.3]` | `[0.81, 0.9, 1.0]` |

额外用上一节的一步未终止样本，核对末尾价值没有被错误清零：优势应为 `[0.12]`，价值目标应为 `[0.72]`。

```bash
uv run python handson/05_gae.py
```

本轮只需要 Python 列表和循环，不需要启动 GPU、训练新网络。验证通过后，直接对照上游 GAE 函数并进入 PPO。

## 9. 复习题

1. GAE 是在组合多步奖励，还是多步 TD 误差？
2. $\lambda=0$ 时，GAE 会变成什么？
3. 为什么递推末尾优势设为零，不等于末尾状态价值也必须为零？
4. 三次动作对应几项奖励、几项价值预测？
5. 标准化后的优势能直接加原价值，作为上面同一个 Critic 目标吗？

<details>
<summary>查看答案</summary>

1. 直接组合的是 TD 误差；每个 TD 误差自身已经含有奖励和价值预测。
2. 只剩当前一步的 TD 误差 $\delta_t$。
3. 优势递推在已采样片段处停止；尚未结束时，末尾价值预测仍通过最后一个 TD 误差进入计算。
4. 三项奖励、四项价值预测，其中最后一项对应末尾状态。
5. 不能。应先由未标准化的优势构造并保留价值目标，再单独标准化给策略使用的优势。

</details>

下一课：PPO——用旧策略概率、优势与裁剪目标更新 VLA，连接 RL4VLA 的实际训练入口。

## 参考资料

- [GAE 原论文](https://arxiv.org/abs/1506.02438)：优势估计及偏差与方差的权衡。
- [RL4VLA：回报与优势计算](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/utils/replay_buffer.py)。
- [RL4VLA：训练参数与缓冲区入口](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/train_ms3_ppo.py)。
- [RL4VLA：环境结束标记的来源](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/env/simpler_wrapper.py)。
