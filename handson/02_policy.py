"""手写 RL · 第二个实验：认识动作分数、概率分布与动作采样。"""

import torch
from torch.distributions import Categorical
import gymnasium as gym

def main():
    torch.manual_seed(7)

    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=7)

    # 创建一个输入4个数, 输出2个分数的网络
    policy = torch.nn.Linear(4, 2)

    # 将环境的Numpy 转换为Pytorch张量
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

    # 把观测送入网络 得到动作分数
    logits = policy(observation_tensor)

    distribution = Categorical(logits=logits)
    # 第一小步：先用手动指定的分数理解分布，还不用创建神经网络。
    # TODO 1：用 torch.tensor 创建两个动作分数 [2.0, 1.0]，命名为 logits。

    # TODO 2：用 Categorical(logits=logits) 创建动作分布，命名为 distribution。
    distribution = Categorical(logits=logits)
    action = distribution.sample()

    print("输入观测：", observation_tensor)
    print("网络输出的分数：", logits)
    print("动作概率：", distribution.probs)
    print("抽到的动作：", action.item())

    # TODO 3：打印 distribution.probs，观察归一化后的动作选择概率。
    print(distribution.probs)
    # TODO 4：用 distribution.sample() 抽取动作，打印 action.item()。
    # action = distribution.sample()
    # print(action.item())
    print(distribution.log_prob(action))

if __name__ == "__main__":
    main()
