# 从笔记走到代码：第一组 RL 练习

在项目根目录运行下面的命令。只需要 Python 3，无第三方依赖、GPU 或模型下载。

代码入口：[mini_rl.py](mini_rl.py)。所有实验都在一个文件中，从 `main()` 找到命令，再跳到对应函数即可。

这是教学用的三步环境，不是机器人仿真。算法是完整轨迹的 Monte Carlo 策略梯度，使用表格价值预测作为基准；尚未实现 PPO、GAE 或神经网络。

## 任务与代码约定

```text
s0 尚未靠近 → 靠近 → s1 已经靠近 → 抓取 → s2 已经抓住 → 放入盘子 → 成功
```

每个状态只有两个动作：动作 0 推进任务，动作 1 立即失败并真正终止。只有成功完成最后一步的奖励为 1，其余为 0。状态最多访问一次，任务最多三步；没有外部时间截断。

策略只接收状态编号，并不知道哪个动作有奖励。初始策略在每个状态各以 50% 的概率选两个动作，整个任务成功概率为 `0.5 × 0.5 × 0.5 = 0.125`。

`logits[state]` 是策略参数，经过 sigmoid 得到选择动作 0 的概率；`values[state]` 是该状态的可训练价值预测。这些小数组用于让你看清参数更新，后面再替换成网络。

## 练习 1：先看一条真实采样轨迹

```bash
python practice/mini_rl.py rollout --episodes 5 --seed 7
```

阅读 `CarrotEnv.step()` 和 `collect_episode()`。找到环境如何产生奖励、策略如何抽取动作、数据在哪里保存。

运行前预测：失败的 episode 一定有三步吗？运行后找一条成功轨迹和一条失败轨迹，记录它们的奖励序列。

**完成标准：**能指出打印的一行对应一次状态转移；知道 `terminated=True` 表示真正终止，不是“这一批采够了”。失败可以在第一步或第二步提前结束。

## 练习 2：把第 1、2 课的手算变成代码

```bash
python practice/mini_rl.py returns --rewards 0 0 1 --gamma 0.9
python practice/mini_rl.py returns --rewards 0 0 --gamma 0.9 --bootstrap 0.8
```

运行前先写出结果，再打开 `discounted_returns()`，逐行追踪 `future_return` 的值。注意第二条命令中的 `0.8` 是假设的末尾价值预测，不是环境奖励。

自己把 gamma 改成 1 和 0，先预测再运行。这个函数只处理单条连续轨迹片段，不会自动识别重置边界；训练代码对每条 episode 单独调用它。

**完成标准：**解释 `[0.81, 0.9, 1.0]` 与 `[0.648, 0.72]` 的来源，指出自举值在哪里进入递推。

## 练习 3：亲眼看价值预测从 0.8 向 0.5 调整

```bash
python practice/mini_rl.py value --returns 1 0 1 0 --prediction 0.8
```

阅读 `run_value()`。这里的 `[1, 0, 1, 0]` 是四次独立运行各自的完整回报，假设它们来自同一状态、同一策略。

运行前预测：第一次更新后价值应该上升还是下降？然后比较 `--lr 0` 与 `--lr 0.5`。

这里采用价值损失 $\tfrac12(V-G)^2$，多了 $\tfrac12$ 只影响梯度的常数倍，不改变目标位置。与笔记中的平方误差目标相同，学习率需要结合损失定义理解。

**完成标准：**能解释 `prediction -= lr * gradient` 每个量的含义，知道预测接近样本平均值不等于已精确知道真实价值。

## 练习 4：改变优势，看动作概率如何变化

```bash
python practice/mini_rl.py update --advantage 0.3
python practice/mini_rl.py update --advantage -0.3
python practice/mini_rl.py update --advantage 0.8
python practice/mini_rl.py update --action wrong --advantage -0.3
```

阅读 `policy_gradient()` 和 `run_update()`，对照第 5 课的单参数更新例子。四条命令彼此独立，默认都从 logit=0、两个动作各 50% 开始。

运行前分别预测“所采动作”的概率会上升还是下降。最后一条采到了动作乙，别把“甲的概率变化”与“所采动作的概率变化”混在一起。

**完成标准：**解释正负优势与更新方向的关系。保持其他量相同，比较优势 0.3 与 0.8 的梯度贡献；概率变化不要求严格按同一倍数放大。

## 练习 5：第一次完整训练

```bash
python practice/mini_rl.py train --out runs/first
```

沿着 `run_train()` → `collect_episode()` → `discounted_returns()` → `update()` 阅读。每轮先用固定策略收集 64 条完整 episode，再更新一次参数。

在 `update()` 找出下面的对应关系：

| 知识点 | 代码中的量或操作 |
| --- | --- |
| 实际回报 | `actual_return` |
| 状态价值预测 | `values[state]` |
| 优势估计 | `actual_return - values[state]` |
| 策略损失的梯度 | `policy_gradient(...)` |
| 价值损失的梯度 | `values[state] - actual_return` |
| 更新参数 | `logits[state] -= ...` 与 `values[state] -= ...` |

代码直接写出了损失的解析梯度，没有自动微分框架。Actor 按 episode 数归一化，并使用 $\gamma^t$ 权重，对应优化起点的折扣回报；Critic 按该状态被访问次数平均误差。整批优势都使用更新前的价值预测。

运行会保存 `initial.json`、`final.json`、`metrics.csv` 和 `config.json`。输出目录已存在时会拒绝覆盖；重跑请选择新名字。

日志中的“精确成功概率”是本环境足够小、可枚举后得到的核对答案，只用于评估，不参与训练。采样评估成功次数与它存在随机误差。

**完成标准：**自己跑一次，查看 CSV，并解释是哪三个策略参数改变了，而不是只看“训练完成”。

## 练习 6：独立加载评估，再做一个对照

```bash
python practice/mini_rl.py evaluate --checkpoint runs/first/initial.json --seed 123
python practice/mini_rl.py evaluate --checkpoint runs/first/final.json --seed 123
python practice/mini_rl.py rollout --checkpoint runs/first/final.json --episodes 5
python practice/mini_rl.py train --actor-lr 0 --out runs/frozen-actor
```

前两条在新进程加载模型，使用同一批评估随机种子比较训练前后。最后一条冻结策略，只允许价值预测学习。

运行前预测：如果只更新价值预测，动作概率和任务成功概率会改善吗？

**完成标准：**看到冻结策略的对照中，价值预测可以改变，但策略概率始终为 0.5，精确成功概率始终为 0.125。理解“价值学得更准”和“策略表现更好”是两件事。

## 实验记录

每次先预测、运行、解释，再在网页的“学习笔记”中勾选对应代码练习。

| 项目 | 你的记录 |
| --- | --- |
| 修改了哪个参数或哪一行代码？ | |
| 运行前预测什么会变化？ | |
| 实际命令与输出目录 | |
| 哪个结果与预期不同？ | |
| 你如何从代码解释这个结果？ | |

实现核对命令：`python -m unittest discover -s practice -v`。它检查手写梯度与数值求导是否一致、episode 边界、环境奖励，以及多个种子下的学习结果。

完成本组练习后，再继续 Actor-Critic、TD 与 GAE，最后迁移到小环境 PPO；不要把本组玩具结果当作 VLA 训练结果。
