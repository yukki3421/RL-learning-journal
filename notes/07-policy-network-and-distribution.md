# 07 · 策略网络与动作分布：从四个观测到两个动作

前置知识：[认识 Gymnasium 与 CartPole](06-gymnasium-and-cartpole.md)、[策略如何学习](05-how-policy-learns.md)。

你已经亲手完成环境初始化、固定动作交互、终止判断，并保存了九步轨迹。这一课增加新的内容：**策略网络输出什么？这些输出怎样变成可执行的动作？**

当前练习分成两小步：先手写一个动作分布实验，理解分数、概率和采样；再把分数的来源换成接收观测的神经网络。本课先不写完整训练循环。

## 1. 把固定动作扩展成根据观测做选择

现在代码每步都执行：

```python
action = 0
```

这是一种固定策略，总是向左推。我们接下来希望实现的路径是：

```text
四个观测数值
    ↓
策略网络
    ↓
两个动作的分数 logits
    ↓
两个动作的概率
    ↓
按概率抽取动作 0 或 1
    ↓
env.step(action)
```

这里只规定策略如何产生动作；它是否表现良好，需要后续通过训练和评估检验。

## 2. 新概念一：logits 是分数

假设网络对两个动作输出：

```text
动作 0，向左推：分数 2.0
动作 1，向右推：分数 1.0
```

我们把这组分数叫做 **logits**。可以先理解为策略对各动作给出的相对偏好。

`[2.0, 1.0]` 的总和是 3，也有数字大于 1，因此它还不是概率。分数也可以是负数，例如 `[-3.0, -2.0]`。负分数不表示负概率。

在本课的概率构造中，相对较大的分数会对应相对较大的选择概率。

注意：这里的动作分数不是 Q 值，不是在预测回报；它是用于构造动作分布的策略输出。

## 3. 新概念二：softmax 把分数转成概率

对离散动作，可以用 softmax 把每个分数转换成非负、总和为 1 的概率：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

其中 $z_i$ 是动作 $i$ 的分数，$p_i$ 是它的选择概率。

对 `[2.0, 1.0]`：

$$
p_0=\frac{e^2}{e^2+e^1}\approx0.7311,\qquad
p_1=\frac{e^1}{e^2+e^1}\approx0.2689
$$

| 动作 | 原始分数 | 转换后的概率 |
| --- | --- | --- |
| 0，向左推 | 2.0 | 约 73.11% |
| 1，向右推 | 1.0 | 约 26.89% |

它们表示选择动作的概率，不是任务成功率。

若两个动作分数相同，它们的概率都是 0.5；把所有分数同时加上同一个数，不改变 softmax 的结果。计算可以交给 PyTorch，不要求你现在手写指数归一化。

## 4. 新概念三：概率分布与动作采样

PyTorch 用 `Categorical` 表示有限个离散选项的概率分布。当前只有两个选项，对应动作编号 0 和 1。

```python
import torch
from torch.distributions import Categorical

logits = torch.tensor([2.0, 1.0])
distribution = Categorical(logits=logits)

print(distribution.probs)
action = distribution.sample()
print(action.item())
```

逐行理解：

- `torch.tensor([2.0, 1.0])`：创建存放两个动作分数的张量，这里先手动指定分数。
- `Categorical(logits=logits)`：根据分数构造概率分布，库会完成相应的概率归一化。
- `distribution.probs`：查看两个动作的概率。
- `distribution.sample()`：根据分布抽取一个动作，结果是一个标量张量。
- `action.item()`：取出普通 Python 整数，之后可以交给 `env.step()`。

虽然动作 0 的概率更大，某次采样仍可能得到动作 1。约 73% 是长期重复抽取的比例，不要求十次恰好出现七次，也不代表每次都选分数最大的动作。

注意输入形式：`Categorical(logits=...)` 接收分数；`Categorical(probs=...)` 接收概率权重。把原始 `[2, 1]` 当 `probs` 输入，会得到不同的分布，不能混用。

## 5. 分数以后由谁来产生？策略网络

前面的两个分数是手动给的。接下来由网络读取 CartPole 的四个观测，计算两个分数。

最简单的例子是：

```python
policy = torch.nn.Linear(4, 2)
observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
logits = policy(observation_tensor)
distribution = Categorical(logits=logits)
```

`Linear(4, 2)` 表示输入四个数、输出两个数。对于一个动作 $i$，计算形式为：

$$
z_i=\sum_{j=0}^{3}W_{ij}o_j+b_i
$$

- $o_j$：观测中的数值。
- $W$、$b$：可训练的权重和偏置。
- $z_i$：这个动作对应的分数。

这时同一组参数遇到不同观测，可以产生不同的动作概率。网络的初始参数尚未经过 RL 训练，不能因为“用了网络”就认为它已经比固定动作更好。

本课先理解最小线性策略；后面可以增加隐藏层，但层数不是当前重点。

## 6. 与第 5 课的 log 概率接起来

采到一个动作后，分布还可以告诉你这个动作的 log 概率：

```python
action = distribution.sample()
log_prob = distribution.log_prob(action)
```

如果采到动作 0，对前面手动给定的 `[2, 1]`：

$$
\log\pi(a=0\mid o)=\log(0.7311)\approx-0.3133
$$

第 5 课中的 $-\hat A\log\pi_\theta(a\mid o)$，就可以用这个 `log_prob` 构造。

真正训练时，需要让 logits 来自可训练网络，并保留 `log_prob` 的张量计算图。交给环境的动作可以用 `action.item()` 转成整数；用于构造损失的 `log_prob` 要保留为张量。手动创建常量 logits 的第一个实验只认识分布，还没有网络参数可以更新。

## 7. 现在亲手写哪一小段？

打开 [handson/02_policy.py](../handson/02_policy.py)，只补四个 TODO：

1. 手动创建 logits `[2.0, 1.0]`。
2. 创建 `Categorical` 分布。
3. 打印概率。
4. 抽取并打印一个动作。

```bash
uv run python handson/02_policy.py
```

第一次先核对概率是否接近 `[0.7311, 0.2689]`，动作是否为 0 或 1。然后把两个分数改为 `[1.0, 2.0]`，先预测概率怎样变化，再运行。

这个文件是独立的小实验，不需要立即改动已经写好的环境交互循环。理解分布后，再写网络并接入环境。

## 8. 复习题

1. `[2, 1]` 是概率还是分数？为什么？
2. 两个 logits 都为 0，两个动作概率分别是多少？
3. 动作 0 的概率为 0.73，是否意味着每次都执行动作 0？是否表示成功率为 73%？
4. `Linear(4, 2)` 的四个输入和两个输出，分别对应什么？
5. 动作分数较高，是否说明这个动作的 Q 值必然较高？

<details>
<summary>查看答案</summary>

1. 是原始分数。它们没有按概率约束表示，需要转换成动作分布。
2. 均为 0.5，softmax 对相同分数给予相同概率。
3. 都不是。它描述抽取该动作的概率，单次仍可能抽到另一个动作，且与任务成功率含义不同。
4. 输入是位置、速度、角度和角速度；输出是向左、向右两个动作的分数。
5. 不一定。logits 是构造策略概率的分数，Q 是先做某动作后继续执行策略的期望回报，二者定义不同。

</details>

后续实践：接入策略网络、补全回报递推，用采到的回报和 log 概率实现第一次参数更新，再引入价值基准。

## 参考资料

- [PyTorch：Categorical 分布](https://docs.pytorch.org/docs/2.15/distributions.html#torch.distributions.categorical.Categorical)。
- [PyTorch：Linear](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html)。
