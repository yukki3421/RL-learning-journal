"""手写 RL：按第 8 课完成一次采样、回报计算和策略参数更新。"""

import gymnasium as gym
import torch
from torch.distributions import Categorical


def discounted_returns(rewards, gamma):
    # TODO：从后往前计算每一步的回报，最后一项之后的剩余回报从 0 开始。
    # 提示：G = rewards[t] + gamma * G。
    returns = [0] * len(rewards)
    G = 0.0

    for i in reversed(range(len(rewards))):
        returns[i] = rewards[i] + gamma * G
        G = returns[i]

    return returns


def main():
    # 第一块：复用第 7 课的初始化，创建环境、策略和 Adam 优化器。
    torch.manual_seed(7)
    
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=7)

    policy = torch.nn.Linear(4, 2)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.99

    # 第二块：循环采样完整 episode，保存 rewards 与张量形式的 log_probs。
    rewards = []
    log_probs = []

    while True:
        inputs = torch.as_tensor(observation, dtype=torch.float32)
        distribution = Categorical(logits=policy(inputs))
        action = distribution.sample()

        log_probs.append(distribution.log_prob(action))

        observation, reward, terminated, truncated, info = env.step(action.item())
        rewards.append(reward)

        if terminated or truncated:
            break


    # 第三块：调用 discounted_returns，构造损失，执行一次参数更新。
    returns = discounted_returns(rewards=rewards, gamma=gamma)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    log_probs_tensor = torch.stack(log_probs)
    discounts = gamma ** torch.arange(len(rewards), dtype=torch.float32)

    loss = -(discounts * returns_tensor * log_probs_tensor).sum()

    # 验收：打印总步数、回报首尾值、loss、更新前后参数的最大差值。
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    max_change = max(
        (parameter.detach() - old).abs().max().item() for parameter, old in zip(policy.parameters(), before)
    )

    print("总步数：", len(rewards))
    print("log_prob 数量：", len(log_probs))
    print("前3个回报：", returns[:3])
    print("最后3个回报：", returns[-3:])
    print("策略损失：", loss.item())
    print("参数最大变化：", max_change)
    env.close()

    # 提示：本轮最多 500 步作为学习目标边界，尚未引入价值网络。


if __name__ == "__main__":
    main()
