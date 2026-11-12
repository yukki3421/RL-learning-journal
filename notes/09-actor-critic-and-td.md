# 09 · Actor-Critic 与 TD 误差：为 RL4VLA 的 PPO 做准备

前置知识：[状态价值函数](03-state-value-function.md)、[动作价值与优势](04-action-value-and-advantage.md)、[第一次策略更新](08-first-policy-update.md)。

**总体目标仍然是尽快跑通 RL4VLA 单任务 PPO 的采样、更新、保存和独立评估。** 本课补齐实际训练中价值预测与 TD 误差的含义，下一课接 GAE，再学习 PPO。

学习方式保持不变：先读清楚知识点，再由你手写代码；助手只检查和解释。当前只做一次 TD 数值和梯度核对，不增加一套长期 CartPole 训练实验。

## 1. 为什么 REINFORCE 之后要学这一课？

第 8 课用实际回报 $G_t$ 给动作的 log 概率加权，需要知道这段任务的完整后续奖励。

RL4VLA 的 PPO 实现还会预测状态价值，并用价值预测和奖励构造优势估计。因此我们先要回答：

1. 谁负责选择动作，谁负责预测回报？
2. 后面还没跑完时，怎样用价值预测构造学习目标？
3. 这个目标比原来的预测高多少，怎样形成策略更新信号？

## 2. Actor 与 Critic 分别是什么？

**Actor-Critic 是同时学习策略和价值评估的一类方法。Actor 负责选择动作，Critic 负责估计价值，为策略更新提供评价信息。**

| 角色 | 回答的问题 | 输出 |
| --- | --- | --- |
| Actor，策略 | 当前应该怎样选择动作？ | 构造动作分布所需的参数，例如 logits |
| Critic，价值评估 | 从当前状态按策略继续执行，预计有多少回报？ | 一个价值预测 $V_\phi(s)$ |

Actor 和 Critic 可以是两个独立网络，也可以共享部分网络并使用不同输出头。不要仅凭“有两个网络”判断算法；还要看价值目标和策略目标如何构造。

用已学过的 CartPole 输入理解形状：

```python
actor = torch.nn.Linear(4, 2)   # 四个观测 → 两个动作分数
critic = torch.nn.Linear(4, 1)  # 四个观测 → 一个价值预测
```

这只是帮助理解输入输出的最小结构。RL4VLA 使用图像、语言指令和 VLA 模型，不是直接使用这两个线性网络。

## 3. 新概念：一步 TD 目标

**TD 是 Temporal Difference 的缩写，中文通常译为“时序差分”，也叫“时间差分”。** 它利用相邻时刻的信息来学习价值：执行一个动作，拿到当前奖励和下一状态，再结合下一状态的价值预测，形成新的学习目标。

```text
当前状态 s_t
    ↓ 执行动作 a_t，得到奖励 r_t
下一状态 s_(t+1)
```

“一步”指的是：**实际向前采集一次转移，使用这一步真实奖励，再用价值预测补上剩余部分。** 它不表示只考虑一步奖励。

先用状态 $s_t$ 写公式；当前 CartPole 的数值观测可用于描述其物理状态。实际具身任务中，价值网络可能使用观测、历史或额外状态，不能一概假定单帧图像就是完整状态。

这就是第 2 课中“自举”的一步版本。只向前采一步，得到一步 TD 目标：

$$
y_t=\underbrace{r_t}_{\text{实际拿到的奖励}}
+\gamma\underbrace{V_\phi(s_{t+1})}_{\text{预测剩余回报}}
$$

这个式子先假设下一状态没有真正终止，且任务目标还有后续部分。

- $r_t$：这一步实际拿到的奖励。
- $V_\phi(s_{t+1})$：到达下一状态后，预计还能拿到的回报。
- $y_t$：从当前开始的回报估计，用来训练当前状态的价值预测。

**它不是完整轨迹的实际回报 $G_t$，因为未来部分使用了预测。**

“一步”描述目标使用了几步实际奖励，不要求每执行一步就立即更新参数；实际训练可以先收集一批转移，再进行更新。这里学习的是一步 TD 形式，Actor-Critic 方法族不只包含这一种目标。

## 4. TD 误差：新目标减去原预测

$$
\delta_t=y_t-V_\phi(s_t)
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

这叫 TD 误差或 TD 残差。它比较的是：根据本步奖励和下一状态估计得到的新目标，与当前状态原来的预测。

假设：

$$
r_t=0,\quad\gamma=0.9,\quad
V_\phi(s_t)=0.6,\quad V_\phi(s_{t+1})=0.8
$$

那么：

$$
y_t=0+0.9\times0.8=0.72,\qquad
\delta_t=0.72-0.6=+0.12
$$

含义是：这一次得到的新目标比原预测高 0.12。

把三个数分开记：

| 量 | 本例数值 | 含义 |
| --- | --- | --- |
| 当前价值预测 $V_\phi(s_t)$ | 0.6 | 网络原来对当前状态的预测 |
| 一步 TD 目标 $y_t$ | 0.72 | 本步实际奖励与下一状态价值预测构成的新目标 |
| TD 误差 $\delta_t$ | 0.12 | 新目标减去原预测 |

**TD 目标是要靠近的目标值，TD 误差是目标与原预测之间的差。** 后面的价值损失和优势估计，会分别用到它们。

不要把这里的 0.12 当成动作必然带来的真实改善；它受本次转移随机性和价值预测误差影响。

## 5. Critic 怎么学，Actor 又怎么用？

### Critic：让预测靠近 TD 目标

一种基本价值损失为：

$$
L_V(\phi)=\frac12\left[V_\phi(s_t)-\operatorname{stopgrad}(y_t)\right]^2
$$

`stopgrad` 表示计算这个损失的梯度时，把目标当作固定量。PyTorch 可以用 `.detach()` 或在 `torch.no_grad()` 中计算目标。

例子中，当前预测 0.6、目标 0.72，所以这条样本推动当前价值预测上升。

### Actor：把 TD 误差作为一种优势估计

一种一步优势估计是：

$$
\hat A_t^{\mathrm{TD}}=\delta_t
$$

对应的单步局部策略损失为：

$$
L_{\mathrm{actor},t}
=-\operatorname{stopgrad}(\delta_t)\log\pi_\theta(a_t\mid s_t)
$$

当价值估计准确、状态建模合适时，TD 误差对给定状态和动作的条件期望对应优势；实际训练使用的是带噪声的估计。

- 正 TD 误差：这条样本推动增加所采动作的倾向。
- 负 TD 误差：这条样本推动减少所采动作的倾向。

这里给出的是局部项。汇总成第 8 课的起点折扣目标时，仍需遵循该目标的时间权重；进入 PPO 后，按其具体损失与采样实现分析，不直接照搬一个局部公式作为整批目标。

**Critic 的目标值、Actor 使用的优势权重，在各自损失中要按上述方式截断相应梯度。** 将 Actor 的权重 detach，不妨碍 Actor 通过 log 概率获得梯度；Critic 则通过自己的价值损失学习。

## 6. 真正终止与截断仍然要分清

如果下一状态真正终止，没有后续奖励，则：

$$
y_t=r_t,\qquad\delta_t=r_t-V_\phi(s_t)
$$

沿用当前预测 0.6，比较三个假设样本：

| 情况 | 本步奖励 | 后续价值的处理 | TD 目标 | TD 误差 |
| --- | --- | --- | --- | --- |
| 未终止，下一状态预测 0.8 | 0 | 自举 | 0.72 | +0.12 |
| 真正失败终止 | 0 | 剩余价值为零 | 0 | -0.6 |
| 真正成功终止 | 1 | 剩余价值为零 | 1 | +0.4 |

这些奖励是假设的教学样本；不是在更改 CartPole 默认每步奖励 1 的规则。

外部时间截断后是否自举，要看任务目标。第 8 课明确优化最多 500 步的目标，因此该目标边界后为零；若目标理论上继续，则通常保留末尾状态价值。有限时域的价值还可能依赖剩余时间，建模时应考虑时间信息。不要把所有结束情况都机械处理成同一种。

## 7. 直接对应 RL4VLA 源码

以下对应已核对的 commit：`de46e0174e7ac310255a2836718fcefab28b8963`。本机尚未安装运行 RL4VLA，这里先阅读上游源码。

| 源码位置 | 本课要识别什么 |
| --- | --- |
| `SimplerEnv/simpler_env/train_ms3_ppo.py` 的 `collect()` | 同一次推理取得 value、action、log probability，并收集环境反馈 |
| `SimplerEnv/simpler_env/policies/openvla/openvla_train.py` | VLA 的 `value_head` 参与价值预测，其参数与策略参数分组配置优化器 |
| `SimplerEnv/simpler_env/utils/replay_buffer.py` 的 `compute_returns_ppo()` | TD 残差来自奖励、下一状态价值、当前价值及结束掩码；后面接 GAE 递推 |

看到 value head，就把它与 Critic 联系起来；看到 TD 残差，就把它与本课的 $\delta_t$ 联系起来。

源码的状态输入、批量形状、结束掩码和损失细节，都需要逐项对照实现。下一课专门解释它如何把多个 TD 残差组合成 GAE。

## 8. 你来写的最小实验

先区分两个实验：8.1 用可求导标量观察梯度；8.2 用真实线性网络观察参数更新。**如果对“value 是不是参数”有疑问，直接做 8.2。** 两份练习代码都由你自己编写。

### 8.1 可求导标量：只观察梯度流向

先读懂前六节，再自己新建 `handson/04_td.py`。本轮只核对一个目标和一次反向传播，不写完整新训练循环。

请手写以下逻辑：

1. 用 PyTorch 创建当前预测 `value=0.6` 和下一状态预测 `next_value=0.8`，都设置 `requires_grad=True`，便于检查梯度。
2. 设置 `reward=0.0`、`gamma=0.9`，假设本次未终止。
3. 计算 `target = reward + gamma * next_value.detach()`。
4. 计算 `delta = target - value`。
5. 构造 `value_loss = 0.5 * (value - target) ** 2`，执行 `value_loss.backward()`。
6. 打印 target、delta、value_loss、value.grad 和 next_value.grad。

预期结果为：

```text
target          ≈ 0.72
delta           ≈ 0.12
value_loss      ≈ 0.0072
value.grad      ≈ -0.12
next_value.grad = None
```

`next_value.grad=None` 是这次实验想验证的结果：目标端 detach 以后，这份价值损失没有通过下一状态预测反向传播。此时 `value.grad` 是损失对这个标量的导数，还不是某个真实网络权重的梯度。

实验中的两个可求导标量是在模拟网络的预测值，不等于已经创建了完整价值网络。通过后，在实际 RL4VLA 的价值预测、目标计算与优化器路径中对应这些角色。

### 8.2 真实 Critic：更新权重，再重新计算 value

自己新建 `handson/04_td_network.py`。为便于手算，用一维特征表示当前、下一状态，分别取 3 和 4；这是假设的数值例子，不是 CartPole 的四维观测。

网络只包含一个权重，没有偏置：

$$
V_w(x)=wx
$$

初始权重设为 $w=0.2$，因此当前预测为 $0.2\times3=0.6$，下一状态预测为 $0.2\times4=0.8$。假设本次未终止、奖励为零，折扣为 0.9，TD 目标仍是 0.72。

本次使用 SGD，以便核对最简单的“参数减去学习率乘梯度”公式。Adam 的具体更新规则不同，但两者更新的都是交给优化器的网络参数。

```python
import torch

critic = torch.nn.Linear(1, 1, bias=False)
with torch.no_grad():
    critic.weight.fill_(0.2)

optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
observation = torch.tensor([3.0])
next_observation = torch.tensor([4.0])
reward = 0.0
gamma = 0.9

value = critic(observation)
with torch.no_grad():
    next_value = critic(next_observation)
    target = reward + gamma * next_value

loss = 0.5 * (value - target).square().mean()

print(f"更新前权重：{critic.weight.item():.4f}")
print(f"更新前 value：{value.item():.4f}")
print(f"TD 目标：{target.item():.4f}")

optimizer.zero_grad()
loss.backward()
print(f"权重的梯度：{critic.weight.grad.item():.4f}")
optimizer.step()

with torch.no_grad():
    new_value = critic(observation)

print(f"更新后权重：{critic.weight.item():.4f}")
print(f"重新计算的 value：{new_value.item():.4f}")
```

关键是梯度通过网络输出传到权重。由 $value=wx$：

$$
\frac{\partial L}{\partial w}
=(value-target)x
=(0.6-0.72)\times3=-0.36
$$

更新之后：

$$
w_{\mathrm{new}}=0.2-0.1\times(-0.36)=0.236,\qquad
V_{w_{\mathrm{new}}}(3)=0.236\times3=0.708
$$

预期输出：

```text
更新前权重：0.2000
更新前 value：0.6000
TD 目标：0.7200
权重的梯度：-0.3600
更新后权重：0.2360
重新计算的 value：0.7080
```

**参数是 `critic.weight`；`value` 是使用参数计算出来的输出；`target` 是本次固定的学习目标。** 优化器更新权重后，旧的 `value` 张量不会自动重新计算，需要再次调用 `critic(observation)` 才得到新输出。

这个例子也说明：不能直接用“输出对损失的梯度”代替“网络参数的梯度”。本例对输出的导数是 -0.12，传到权重时还乘上输入 3，得到 -0.36。

## 9. 复习题

1. Actor 与 Critic 各自输出什么？
2. TD 目标为什么不等于本次完整轨迹的实际回报？
3. 当前预测 0.6、奖励 0、下一预测 0.8、折扣 0.9，TD 误差是多少？
4. 同样当前预测 0.6，但本步奖励 0 且真正终止，TD 误差是多少？
5. Actor 使用 `delta.detach()`，是否就没有梯度可以更新策略了？

<details>
<summary>查看答案</summary>

1. Actor 产生动作分布参数，Critic 产生价值预测；网络结构可以共享部分主干。
2. 它用下一状态的价值预测替代未采到的未来部分，含有自举估计。
3. $0+0.9\times0.8-0.6=+0.12$。
4. $0-0.6=-0.6$，真正终止后不保留剩余价值。
5. 不是。权重固定下来，策略仍然通过动作 log 概率计算自己的梯度。

</details>

下一篇：[GAE：广义优势估计](10-generalized-advantage-estimation.md)——把多步 TD 残差组合成优势估计，读懂 RL4VLA 的缓冲区计算。

## 源码参考

- [RL4VLA：采样与训练入口](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/train_ms3_ppo.py)。
- [RL4VLA：策略与价值头](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/policies/openvla/openvla_train.py)。
- [RL4VLA：回报和优势缓冲区](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/utils/replay_buffer.py)。
