# 08 · 第一次策略更新：从轨迹到 REINFORCE

前置知识：[训练时如何计算回报](02-computing-returns.md)、[策略如何学习](05-how-policy-learns.md)、[策略网络与动作分布](07-policy-network-and-distribution.md)。

本课把已学知识串成一个完整环节：**网络与环境交互 → 计算回报 → 构造损失 → 更新一次网络参数。**

**2026-11-12 主线调整：本课的一次更新已完成运行核对，接下来转入 RL4VLA 单任务 PPO 项目，围绕实际训练代码补齐 Critic、GAE 和 PPO。** 后续主要交付是机器人任务的基线、训练、检查点与独立评估，推进顺序见 [项目当前主目标](../README.md)。

代码入口：[handson/03_update_once.py](../handson/03_update_once.py)。文件里只留了函数骨架和 TODO，由你亲手补全。保留已经完成的 `01_cartpole.py`、`02_policy.py` 作为前两阶段记录。

这次按三个代码块推进，不再每打印一行就暂停。先完成一个完整更新，再检查你对关键量的解释。

## 1. 新知识点：REINFORCE 是什么？

### 1.1 先记住完整定义

**REINFORCE 是一种蒙特卡洛策略梯度算法：使用当前随机策略采集完整 episode，根据实际获得的奖励计算回报，再用回报加权所采动作的 log 概率，估计提高期望回报的梯度并更新策略参数。**

这里有三个关键词：

| 关键词 | 在本课中是什么意思？ |
| --- | --- |
| 策略梯度 | 直接计算如何调整策略网络的参数，让期望回报提高 |
| 蒙特卡洛 | 用实际采到的完整后续奖励计算回报，作为随机样本；不要求事先知道真实的 Q 值 |
| 当前策略采样，也叫 on-policy | 使用采样时的策略收集数据，用这些数据估计该策略的更新方向；本课在采样结束后更新一次，再重新采样 |

“完整 episode”在本课指运行到真正终止，或运行到我们定义的 500 步目标边界。具体目标在第 2 节说明。

REINFORCE 学习的是**策略网络的权重和偏置**。环境负责产生反馈，算法通过动作 log 概率的梯度来更新网络，不需要对环境的物理运动过程求导。

### 1.2 它是一套怎样的训练流程？

```text
当前策略与环境交互，采集一条或多条完整 episode
                      ↓
根据实际奖励，计算每一步的后续回报 G_t
                      ↓
将 G_t 与同一步所采动作的 log 概率配对，构造损失
                      ↓
反向传播，优化器更新策略参数
                      ↓
用更新后的策略重新采样，开始下一轮
```

一条轨迹的结果具有随机性。它提供的是梯度的样本估计，不保证某次更新后任务表现立即提高。

### 1.3 它就是前面学过的损失吗？

**REINFORCE 使用了前面学过的策略梯度损失形式；算法还包括数据采集、回报计算与参数更新的完整流程。**

先看单步样本的局部损失，暂不展开整条轨迹的时间权重：

| 版本 | 用什么给 log 概率加权？ | 单步局部损失 |
| --- | --- | --- |
| 基础 REINFORCE，本课先实现 | 实际回报 $G_t$ | $-G_t\log\pi_\theta(a_t\mid o_t)$ |
| 带状态价值基准的 REINFORCE，后续学习 | 实际回报减去价值预测 $G_t-V_\phi(o_t)$ | $-[G_t-V_\phi(o_t)]\log\pi_\theta(a_t\mid o_t)$ |

如果把第二行中的差值记作优势估计：

$$
\hat A_t=G_t-V_\phi(o_t)
$$

就接回前面熟悉的形式：

$$
L_t=-\hat A_t\log\pi_\theta(a_t\mid o_t)
$$

其中 $\pi_\theta(a_t\mid o_t)$，就是你问的 $p$：策略给“在该观测下实际采到的那个动作”的概率。

需要分清两点：

- 本课基础版本用 $G_t$ 作权重，不是说实际回报就等于真实优势函数。
- $G_t-V_\phi(o_t)$ 是根据一次采样构造的优势估计；它会受到采样随机性和价值预测误差的影响。

整条轨迹如何求和，以及针对本课的起点折扣目标为什么还要乘 $\gamma^t$，在第 5 节展开。

### 1.4 把策略、算法和损失分开理解

| 名称 | 回答什么问题？ | 本项目中的例子 |
| --- | --- | --- |
| 策略网络 | 根据观测，怎样选择动作？ | `Linear(4, 2)` 生成分数，再构造动作分布 |
| REINFORCE 算法 | 怎样利用环境反馈训练这个策略？ | 采完整轨迹，计算回报，构造损失，更新参数 |
| 策略损失 | 这次更新用哪个可求导的量来产生梯度？ | 回报或优势估计加权的负 log 概率 |

因此，看到 $-\hat A\log p$，首先能识别出它是策略梯度的一种样本损失形式；判断具体算法，还要看数据从哪里来、权重怎样估计、怎样执行更新。

这课先实现**无价值基准的 REINFORCE**：采样期间保持策略参数不变，采样完成后计算回报并更新一次。掌握后，再加入价值预测，理解它怎样帮助降低梯度估计的波动。

## 2. 先明确这版练习的目标

我们优化从 episode 起点开始、最多 500 步的折扣奖励和：

$$
J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\qquad T\leq500
$$

本课取 $\gamma=0.99$。

CartPole 在时间上限处仍返回 `truncated=True`。但这版学习目标明确只累计到 500 步，所以达到这个目标边界时，边界之后的回报为零。

这与“只因采样收够了，就把仍在继续的任务剩余回报一律设零”不同。以后若学习理论上继续的任务并使用价值网络，需要按第 2 课处理时间截断与自举。现在先保留原始 `terminated`、`truncated`，停止交互时检查任一个为真。

## 3. 第一块：初始化与完整 episode 采样

先在 `main()` 中初始化。这些大部分是你已经写过的代码，只增加优化器：

```python
torch.manual_seed(7)
env = gym.make("CartPole-v1")
policy = torch.nn.Linear(4, 2)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

observation, info = env.reset(seed=7)
rewards = []
log_probs = []
```

`policy.parameters()` 是网络的可训练权重和偏置。优化器接收这些参数，之后负责更新它们。**创建优化器本身不会更新网络。**

接着，把单次前向计算放进交互循环：

```python
while True:
    inputs = torch.as_tensor(observation, dtype=torch.float32)
    distribution = Categorical(logits=policy(inputs))
    action = distribution.sample()

    log_probs.append(distribution.log_prob(action))

    observation, reward, terminated, truncated, info = env.step(action.item())
    rewards.append(reward)

    if terminated or truncated:
        break
```

这里直接把新观测赋给 `observation`，因为本次只需要保存每步的回报权重和 log 概率，没有把前后观测写入轨迹字典。如果你仍想保存转移字典，就应像 `01_cartpole.py` 一样先保留旧观测再更新。

| 变量 | 保存什么 | 为什么需要 |
| --- | --- | --- |
| `rewards[t]` | 第 $t$ 次动作后的实际奖励 | 计算这一步的后续回报 |
| `log_probs[t]` | 本步所采动作的 log 概率张量 | 构造损失，并保留它与网络参数的梯度关系 |

两者长度必须相等，且第 $t$ 项对应同一次动作。

### 为什么不能保存 `log_prob.item()`？

`.item()` 会取出普通 Python 数值，失去用于反向传播的张量关系。训练需要保留 log 概率的 Tensor。

但 `action.item()` 是合适的：环境需要动作整数。我们并不对这个离散抽样结果本身求导，而是利用所采动作的 log 概率求梯度，这正好接上第 5 课的推导。

## 4. 第二块：亲手计算每一步的回报

补全 `discounted_returns()`，它放在 `main()` 外面、文件顶部的导入语句之后：

```python
def discounted_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    G = 0.0

    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns
```

这就是第 2 课的递推：

$$
G_t=r_t+\gamma G_{t+1}
$$

例如，完整奖励序列为 `[1, 1, 1]`，取 $\gamma=0.99$：

```text
最后一步：G2 = 1
前一步：  G1 = 1 + 0.99 × 1 = 1.99
第一步：  G0 = 1 + 0.99 × 1.99 = 2.9701
```

得到 `[2.9701, 1.99, 1.0]`。这与 episode 原始奖励和 3 是不同的量。

在 `main()` 的采样循环之后调用：

```python
returns = discounted_returns(rewards, gamma)
```

## 5. 第三块：用回报构造策略损失

对于第 2 节定义的起点折扣目标，我们用下面的样本损失产生一次策略梯度估计：

$$
L(\theta)=-\sum_{t=0}^{T-1}\gamma^tG_t
\log\pi_\theta(a_t\mid o_t)
$$

三部分的意义是：

- $\log\pi_\theta(a_t\mid o_t)$：来自对应时刻的 `log_probs[t]`。
- $G_t$：从该时刻开始，实际采到的后续折扣回报。
- $\gamma^t$：把这个时刻的贡献折扣到 episode 起点，与我们定义的起点目标对齐。

### 5.1 为什么算过回报，损失里还要乘 $\gamma^t$？

**$G_t$ 把后续奖励折扣到当前第 $t$ 步，外面的 $\gamma^t$ 再把这些奖励统一折扣到 episode 起点。**

从定义展开，区别最清楚：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

这里 $k-t$ 是奖励距离当前动作的步数。再乘上 $\gamma^t$：

$$
\gamma^tG_t
=\sum_{k=t}^{T-1}\gamma^t\gamma^{k-t}r_k
=\sum_{k=t}^{T-1}\gamma^k r_k
$$

指数变成 $k$，也就是奖励距离开局的步数。因此，每份奖励最终都只按它距离开局的总步数折扣；两个因子是在拆分这段距离，而不是把同一段距离重复折扣。

这来自本课选择的“从 episode 起点计算折扣回报”的目标。它不是一条“只要计算策略损失就必须再折扣”的通用规则；目标和采样约定变化时，要重新核对更新形式。若 $\gamma=1$，这些系数全部为 1。

### 5.2 先看最后一步：它的 G 里面可能完全没有折扣

假设只有两次动作，奖励为 `[0, 1]`，取 $\gamma=0.5$。

对于第二次动作，也就是 $t=1$：

$$
G_1=r_1=1
$$

因为奖励就在当前这一步，$G_1$ 内部没有折扣。但整局目标是：

$$
G_0=r_0+0.5r_1
$$

在整局目标中，这份第二步奖励按 $0.5$ 计入，因此对应的更新权重为：

$$
\gamma^1G_1=0.5\times1=0.5
$$

外面的 $0.5$ 补上的是“从开局到第二次动作”的距离。这能帮助区分：**不是每个 $G_t$ 都已经从开局折扣过；只有 $G_0$ 本身以开局为参照。**

### 5.3 统一到起点，不表示每步都用完整的 G0

准确地说：**保留从当前第 $t$ 步开始的奖励，把这些当前及未来奖励按 episode 起点的尺度计权。**

对于完整奖励序列 `[1, 1, 1]`，取 $\gamma=0.9$：

| 当前动作编号 $t$ | 当前时刻视角的回报 $G_t$ | 外部系数 $\gamma^t$ | 用于该步更新的权重 $\gamma^tG_t$ |
| --- | --- | --- | --- |
| 0 | $1+0.9+0.81=2.71$ | 1 | 2.71 |
| 1 | $1+0.9=1.9$ | 0.9 | 1.71 |
| 2 | 1 | 0.81 | 0.81 |

特别看第 $1$ 步：

$$
\gamma G_1=0.9r_1+0.9^2r_2=1.71
$$

而完整的开局回报是：

$$
G_0=r_0+0.9r_1+0.9^2r_2=2.71
$$

本课的 reward-to-go 写法没有把此前已获得的 $r_0$ 纳入第 $1$ 步的权重，因为当前动作无法改变过去的奖励。省去这些过去奖励对应的项，不改变策略梯度的期望。完整回报加权是另一种可用的估计形式，本课采用这里的后续回报形式。

这里区分的是两个问题：折扣相对于哪个起点，以及累计哪些奖励。**统一折扣起点，并不意味着扩大累计范围，把过去奖励重新加回来。**

### 5.4 代码需要记录这是第几步吗？

本课需要知道动作在当前 episode 内的编号 $t$，但不需要额外随机采样一个编号。按顺序保存的数据已经隐含了它：

```text
rewards[0]、log_probs[0] → 第一次动作，t=0，系数为 1
rewards[1]、log_probs[1] → 第二次动作，t=1，系数为 gamma
rewards[2]、log_probs[2] → 第三次动作，t=2，系数为 gamma²
```

每次 `reset()` 开始新 episode，都从 $t=0$ 重新计数。不要使用整个训练过程的总步数，也不要把打乱后 minibatch 的下标当成原始时间编号。如果以后把多个 episode 合并到一个缓冲区，需要保留各条轨迹的时间信息和边界。

你之前打印的 `step_count` 从 1 开始，列表下标从 0 开始；第一次动作对应 $t=0$，不是 $t=1$。

第 5 课先解释了单条样本的局部形式。本课按整个 episode 求和，并明确目标与时间权重；后续实现别的算法时要核对其目标和采样约定。

### 5.5 与公式逐项对应的代码

代码中的 `returns_tensor` 保存 $G_t$；`discounts` 则生成 $[1,\gamma,\gamma^2,\ldots]$。这里恢复使用本课的 $\gamma=0.99$，前面 $0.5$ 和 $0.9$ 只是便于手算的例子。

```python
returns_tensor = torch.tensor(returns, dtype=torch.float32)
log_probs_tensor = torch.stack(log_probs)
discounts = gamma ** torch.arange(len(rewards), dtype=torch.float32)

loss = -(discounts * returns_tensor * log_probs_tensor).sum()
```

`torch.stack(log_probs)` 把一组标量张量叠成一个一维张量，同时保留计算图。不要改成 `torch.tensor(log_probs)`，后者会重新构造数据，无法保留这条梯度关系。

这里按步求和。之后增加多条轨迹时，可以按轨迹数量平均梯度；本课先不引入回报归一化、优势归一化等额外处理。

## 6. 真正更新参数的三步

为了稍后核对参数变化，先存一份快照，再执行更新三步：

```python
before = [parameter.detach().clone() for parameter in policy.parameters()]

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

| 代码 | 做了什么 | 此时参数是否已经更新？ |
| --- | --- | --- |
| `zero_grad()` | 清掉先前的梯度 | 否 |
| `backward()` | 从损失沿计算图计算各参数的梯度 | 否 |
| `step()` | 优化器利用梯度更新参数 | 是 |

PyTorch 默认会累积梯度，所以每次独立更新前需要清理。不要对同一条轨迹保留下来的这份计算图反复执行更新；本课一条轨迹更新一次，下一轮重新采样。

一次更新只说明计算闭环成立，不保证下一次 episode 更长。采样有随机性，单条轨迹的梯度估计也可能波动；后面要用多轮训练与独立评估判断表现。

## 7. 如何确认网络真的变了？

上一节的 `before` 已经在更新之前保存了参数快照，它是专门用于核对的副本，不参与损失计算。不要在更新之后重新给 `before` 赋值，否则就无法比较更新前后的差异。

更新之后计算差值：

```python
max_change = max(
    (parameter.detach() - old).abs().max().item()
    for parameter, old in zip(policy.parameters(), before)
)

print("总步数：", len(rewards))
print("log_prob 数量：", len(log_probs))
print("前3个回报：", returns[:3])
print("最后3个回报：", returns[-3:])
print("策略损失：", loss.item())
print("参数最大变化：", max_change)
env.close()
```

调试时可以在 `loss.backward()` 以及 `optimizer.step()` 处打断点：前者执行之后看 `policy.weight.grad`，后者执行之后对比 `policy.weight` 与参数快照。启动入口选“RL：调试当前 Python 文件”。

## 8. 本轮验收与运行顺序

本轮只要求这三个代码块跑通：

1. 完整采样，`len(rewards) == len(log_probs)`。
2. 回报递推正确，CartPole 默认最后一步的回报为 1。
3. 损失有限、反向传播成功、参数最大变化大于 0。

```bash
uv run python handson/03_update_once.py
```

第一次可以先把更新三行注释掉，核对轨迹与回报；核对后再执行更新。想加快推进，可以一次写完，再按断点逐块检查。

运行后发出上述打印结果。本轮已核对通过；下一阶段把这些基本概念对应到 RL4VLA 的实际 PPO 训练路径，完成该项目的环境启动、基线评估、训练与检查点独立评估。

## 9. 复习题

1. 网络前向计算完毕，参数就已经改变了吗？
2. 为什么 `action.item()` 可以交给环境，而 `log_prob.item()` 不适合保存来训练？
3. `backward()` 和 `optimizer.step()` 有什么区别？
4. 参数变化了，能否因此判断任务表现变好了？
5. 本课在 500 步边界不自举，是否说明所有时间截断都不需要自举？
6. 对 `[1, 1, 1]`、$\gamma=0.9$，第 $1$ 步用于更新的权重是 $G_1=1.9$、$G_0=2.71$，还是 $\gamma G_1=1.71$？
7. 第一次动作打印为“第 1 步”，对应的外部折扣系数是 $1$ 还是 $\gamma$？

<details>
<summary>查看答案</summary>

1. 没有。前向计算使用当前参数生成结果；这版代码在 `optimizer.step()` 才更新参数。
2. 环境需要整数动作；用于训练的 log 概率必须保留与网络参数之间的计算图。
3. 前者计算梯度，后者利用梯度更新参数。
4. 不能。需要后续采样和独立评估判断表现，不能只看某次损失或参数变化。
5. 不是。本课明确优化最多 500 步的有限时域目标；对于理论上继续的任务，外部截断后的剩余价值仍需按目标处理。
6. 本课使用 $\gamma G_1=1.71$，只累计当前及未来奖励，并以开局为折扣参照，不包含过去的 $r_0$。
7. 是 $1$。第一次动作对应从零开始的编号 $t=0$，因此 $\gamma^0=1$。

</details>

下一篇：[Actor-Critic 与 TD 误差](09-actor-critic-and-td.md)——为 RL4VLA 的价值预测、GAE 与 PPO 做准备。

## 参考资料

- [PyTorch：用 sample 与 log_prob 构造 REINFORCE](https://docs.pytorch.org/docs/2.15/distributions.html#score-function)。
- [PyTorch：优化器与参数更新](https://docs.pytorch.org/docs/2.15/optim.html)。
