"""手写 RL：初始化环境，使用固定动作运行一个完整 episode。"""

import gymnasium as gym


def main():
    # 创建环境：库负责模拟小车和杆的运动，后续由我们编写策略。
    env = gym.make("CartPole-v1")

    # reset() 开始一次 episode，返回初始观测和额外信息；此时尚未执行动作。
    # seed=7 固定环境随机种子，便于重复得到相同的初始情况。
    # 初始状态带有小幅随机变化，所以初始位置、速度不一定为零。
    observation, info = env.reset(seed=7)

    # observation 是当前四个具体数值，后续会作为策略网络的输入：
    # [0] 小车位置：正数表示在中心右侧，负数表示在左侧。
    # [1] 小车速度：正数表示向右移动，负数表示向左移动。
    # [2] 杆相对竖直方向的角度，单位为弧度；接近 0 表示接近直立。
    #     本次约 0.0276 rad，即约 1.58 度。
    # [3] 杆的角速度：正数表示角度正在增大，负数表示角度正在减小。
    #     角速度描述变化趋势，不能只凭一个正负号判断动作应该选哪边。
    print("初始观测：", observation)

    # observation_space 是格式与范围说明，不是当前的观测值：
    # Box(下界数组, 上界数组, (4,), float32)
    # 下界 [-4.8, -inf, -0.41887903, -inf]，上界为对应的正值。
    # (4,) 表示一维数组有 4 个元素；float32 表示 32 位浮点数。
    # +/-inf 表示该维度没有设置有限上下界，不表示当前速度无限大。
    # 注意：空间边界不是失败阈值！位置超过约 +/-2.4，或角度超过约
    # +/-12 度就会真正终止；默认 500 步上限则由 truncated 表示。
    print("观测空间：", env.observation_space)

    # Discrete(2) 表示两个离散动作：0 向左推，1 向右推。
    # 它说明允许选择哪些动作，不表示动作概率，也不是让小车直接到达某位置。
    print("动作空间：", env.action_space)
    print("*"*60)
    
    # 连续执行固定动作 0（向左推），观察完整 episode 怎样结束。
    # 这还是固定动作的策略，尚未根据观测选择动作，也没有进行训练。
    step_count = 0
    episode_reward = 0.0
    trajectory = []

    while True:
        action = 0
        next_observation, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
                    "observation": observation.copy(),
                    "action": action,
                    "reward": reward,
                    "next_observation": next_observation.copy(),
                    "terminated": terminated,
                    "truncated": truncated,
                })

        # 每调用一次 step() 就计一步，包括触发终止的最后一步。
        step_count += 1
        episode_reward += reward

        print(
            f"第 {step_count} 步：奖励={reward}，"
            f"真正终止={terminated}，时间截断={truncated}"
        )

        # 新观测成为下一轮的当前观测。
        observation = next_observation

        # 真正终止或被截断后立即退出，不能继续 step()，除非先 reset()。
        if terminated or truncated:
            break

    print("本次总步数：", step_count)
    print("本次奖励总和：", episode_reward)
    
    # 直接查看最后一次有效 step() 返回的结果，不再额外执行动作。
    # seed=7、一直向左推时，第 9 步杆角度约 13.96 度，超过 12 度阈值。
    # 因此 terminated=True；默认每步奖励 1，所以奖励总和也是 9。
    print("最终观测：", observation)
    print("真正终止：", terminated)
    print("时间截断：", truncated)

    print("轨迹长度：", len(trajectory))
    print("第一步记录：", trajectory[0])
    print("最后一步记录：", trajectory[-1])

    rewards = [step["reward"] for step in trajectory]
    print("奖励序列：", rewards)

    # 使用结束后释放环境资源。
    env.close()


if __name__ == "__main__":
    main()
