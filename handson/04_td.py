import torch

def main():
    value = torch.tensor(0.6, requires_grad=True)
    next_value = torch.tensor(0.8, requires_grad=True)

    reward = 0.0
    gamma = 0.9

    target = reward + gamma * next_value.detach()
    delta = target - value

    value_loss = 0.5 * (value - target) ** 2
    value_loss.backward()

    print("TD 目标：", target.item())
    print("TD 误差：", delta.item())
    print("价值损失：", value_loss.item())
    print("当前预测的梯度：", value.grad)
    print("下一状态预测的梯度：", next_value.grad)


if __name__ == "__main__":
    # main()
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(0.2)

    optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)

    observation = torch.tensor([3.0])
    next_observation = torch.tensor([4.0])

    value = critic(observation) # 对当前状态的价值预测

    with torch.no_grad():
        next_value = critic(next_observation)
        target = 0.0 + 0.9 * next_value

    loss = 0.5 * (value - target).square().mean()

    print("更新前权重：", critic.weight.item())
    print("更新前 value：", value.item())
    print("TD 目标：", target.item())

    optimizer.zero_grad()
    loss.backward()

    print("权重的梯度：", critic.weight.grad.item())

    optimizer.step()

    with torch.no_grad():
        new_value = critic(observation)

    print("更新后权重：", critic.weight.item())
    print("重新计算的 value：", new_value.item())

