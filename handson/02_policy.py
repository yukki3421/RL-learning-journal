"""手写 RL · 第二个实验：认识动作分数、概率分布与动作采样。"""

import torch
from torch.distributions import Categorical


def main():
    # 第一小步：先用手动指定的分数理解分布，还不用创建神经网络。
    # TODO 1：用 torch.tensor 创建两个动作分数 [2.0, 1.0]，命名为 logits。
    # TODO 2：用 Categorical(logits=logits) 创建动作分布，命名为 distribution。
    # TODO 3：打印 distribution.probs，观察归一化后的动作选择概率。
    # TODO 4：用 distribution.sample() 抽取动作，打印 action.item()。
    pass


if __name__ == "__main__":
    main()
