"""纯 Python RL 练习：完整轨迹的策略梯度 + 表格价值基准，无第三方依赖。"""

import argparse
import csv
import json
import math
import random
from pathlib import Path


STATES = ("尚未靠近胡萝卜", "已经靠近胡萝卜", "已经抓住胡萝卜")
ACTIONS = (("靠近", "原地结束"), ("抓取", "提前离开"), ("放入盘子", "掉到地上"))
ENV_ID = "carrot-three-stages-v1"


def sigmoid(logit):
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def discounted_returns(rewards, gamma, bootstrap_value=0.0):
    """单条连续片段；真正终止传 0，未终止片段可传末尾价值。"""
    returns = [0.0] * len(rewards)
    future_return = bootstrap_value
    for t in reversed(range(len(rewards))):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return
    return returns


class CarrotEnv:
    """每个状态有两个动作；选错立即失败，连续三次选对才成功。"""

    def reset(self):
        self.state = 0
        self.terminated = False
        return self.state

    def step(self, action):
        if self.terminated:
            raise RuntimeError("任务已终止，请先 reset()")
        if action not in (0, 1):
            raise ValueError("动作必须是 0 或 1")
        success = self.state == 2 and action == 0
        self.terminated = action == 1 or success
        next_state = None if self.terminated else self.state + 1
        self.state = next_state
        return next_state, float(success), self.terminated


def collect_episode(logits, rng):
    """Rollout：策略与环境交互，收集一条真正终止的完整轨迹。"""
    env = CarrotEnv()
    state = env.reset()
    trajectory = []
    while True:
        p_correct = sigmoid(logits[state])
        action = 0 if rng.random() < p_correct else 1
        next_state, reward, terminated = env.step(action)
        trajectory.append({
            "state": state, "action": action, "reward": reward,
            "next_state": next_state, "terminated": terminated,
        })
        if terminated:
            return trajectory
        state = next_state


def policy_gradient(logit, action, advantage):
    """损失 -A log π(a|s) 对本状态 logit 的导数。"""
    # 甲/正确动作的标签为 1，乙/错误动作的标签为 0。
    target = 1.0 if action == 0 else 0.0
    return -advantage * (target - sigmoid(logit))


def exact_success(logits):
    return math.prod(sigmoid(logit) for logit in logits)


def exact_values(logits, gamma):
    """本玩具环境可枚举真值，仅用于核对，训练不能读取这些答案。"""
    values = [0.0] * 3
    values[2] = sigmoid(logits[2])
    for state in (1, 0):
        values[state] = sigmoid(logits[state]) * gamma * values[state + 1]
    return values


def evaluate(logits, episodes, seed):
    rng = random.Random(seed)
    successes = sum(collect_episode(logits, rng)[-1]["reward"] for _ in range(episodes))
    return int(successes)


def update(logits, values, trajectories, gamma, actor_lr, critic_lr):
    """整批采样结束后更新一次；全批优势使用更新前的价值预测。"""
    actor_gradients = [0.0] * 3
    critic_errors = [0.0] * 3
    visits = [0] * 3
    for trajectory in trajectories:
        returns = discounted_returns([step["reward"] for step in trajectory], gamma)
        for t, (step, actual_return) in enumerate(zip(trajectory, returns)):
            state, action = step["state"], step["action"]
            advantage = actual_return - values[state]
            # gamma**t 对应最大化从 episode 起点算的折扣回报。
            actor_gradients[state] += gamma**t * policy_gradient(
                logits[state], action, advantage
            )
            critic_errors[state] += values[state] - actual_return
            visits[state] += 1

    for state in range(3):
        logits[state] -= actor_lr * actor_gradients[state] / len(trajectories)
        if visits[state]:
            # 价值损失取 0.5 * (V-G)**2，导数为 V-G。
            values[state] -= critic_lr * critic_errors[state] / visits[state]


def save_model(path, logits, values, gamma):
    path.write_text(json.dumps({
        "environment": ENV_ID, "logits": logits, "values": values, "gamma": gamma,
    }, ensure_ascii=False, indent=2) + "\n")


def load_model(path):
    data = json.loads(Path(path).read_text())
    if data.get("environment") != ENV_ID:
        raise ValueError("检查点不属于当前玩具环境")
    for key in ("logits", "values"):
        if len(data[key]) != 3 or not all(math.isfinite(x) for x in data[key]):
            raise ValueError(f"检查点中的 {key} 无效")
    if not 0 <= data["gamma"] <= 1:
        raise ValueError("检查点中的 gamma 无效")
    return data


def run_returns(args):
    result = discounted_returns(args.rewards, args.gamma, args.bootstrap)
    print(f"奖励序列：{args.rewards}")
    print(f"gamma={args.gamma}，末尾价值={args.bootstrap}")
    print("回报/自举目标：", [round(x, 6) for x in result])


def run_value(args):
    prediction = args.prediction
    print(f"以下数字是多次运行各自的完整回报：{args.returns}")
    print(f"样本平均值：{sum(args.returns) / len(args.returns):.4f}")
    for iteration in range(args.steps + 1):
        loss = sum(0.5 * (prediction - g)**2 for g in args.returns) / len(args.returns)
        print(f"更新 {iteration:2d}：价值预测={prediction:.4f}，价值损失={loss:.6f}")
        gradient = sum(prediction - g for g in args.returns) / len(args.returns)
        prediction -= args.lr * gradient


def run_update(args):
    action = 0 if args.action == "correct" else 1
    before = sigmoid(args.logit)
    gradient = policy_gradient(args.logit, action, args.advantage)
    after_logit = args.logit - args.lr * gradient
    print(f"采到动作：{'甲/正确' if action == 0 else '乙/错误'}，优势={args.advantage:+.4f}")
    print(f"策略损失对 logit 的梯度：{gradient:+.6f}")
    print(f"logit：{args.logit:.6f} → {after_logit:.6f}")
    print(f"甲的选择概率：{before:.6f} → {sigmoid(after_logit):.6f}")
    sampled_before = before if action == 0 else 1 - before
    sampled_after = sigmoid(after_logit) if action == 0 else 1 - sigmoid(after_logit)
    print(f"所采动作的选择概率：{sampled_before:.6f} → {sampled_after:.6f}")


def run_rollout(args):
    rng = random.Random(args.seed)
    model = load_model(args.checkpoint) if args.checkpoint else {
        "logits": [0.0] * 3, "values": [0.0] * 3, "gamma": args.gamma,
    }
    print("state action reward G V A terminated（末尾为真正终止，无时间截断）")
    for index in range(args.episodes):
        trajectory = collect_episode(model["logits"], rng)
        returns = discounted_returns([step["reward"] for step in trajectory], model["gamma"])
        print(f"\nepisode {index + 1}")
        for step, actual_return in zip(trajectory, returns):
            state, action = step["state"], step["action"]
            value = model["values"][state]
            print(f"s{state}({STATES[state]}) {ACTIONS[state][action]} "
                  f"r={step['reward']:.1f} G={actual_return:.4f} V={value:.4f} "
                  f"A={actual_return-value:+.4f} terminated={step['terminated']}")


def run_train(args):
    output = Path(args.out)
    if output.exists():
        raise ValueError(f"输出路径已存在，请为 --out 选择新路径：{output}")
    output.mkdir(parents=True)
    logits, values = [0.0] * 3, [0.0] * 3
    rng = random.Random(args.seed)
    eval_seed = args.seed + 1000000  # 训练与评估不共享随机数生成器。
    config = {key: value for key, value in vars(args).items() if key != "func"}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_model(output / "initial.json", logits, values, args.gamma)
    env_steps = 0
    with (output / "metrics.csv").open("w", newline="") as handle:
        fields = ["iteration", "env_steps", "eval_successes", "eval_episodes",
                  "exact_success", "expected_start_return", "p0", "p1", "p2",
                  "v0", "v1", "v2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for iteration in range(args.iterations + 1):
            if iteration:
                trajectories = [collect_episode(logits, rng) for _ in range(args.batch_size)]
                env_steps += sum(len(trajectory) for trajectory in trajectories)
                update(logits, values, trajectories, args.gamma, args.actor_lr, args.critic_lr)
            if iteration % args.log_every == 0 or iteration == args.iterations:
                successes = evaluate(logits, args.eval_episodes, eval_seed)
                row = {
                    "iteration": iteration, "env_steps": env_steps,
                    "eval_successes": successes, "eval_episodes": args.eval_episodes,
                    "exact_success": exact_success(logits),
                    "expected_start_return": exact_values(logits, args.gamma)[0],
                    **{f"p{s}": sigmoid(logits[s]) for s in range(3)},
                    **{f"v{s}": values[s] for s in range(3)},
                }
                writer.writerow(row)
                handle.flush()
                print(f"迭代 {iteration:3d} | 环境步数 {env_steps:6d} | "
                      f"评估成功 {successes}/{args.eval_episodes} | "
                      f"精确成功概率 {row['exact_success']:.3f} | "
                      f"甲动作概率 {[round(sigmoid(x), 3) for x in logits]}")
    save_model(output / "final.json", logits, values, args.gamma)
    print(f"已保存：{output}（initial.json / final.json / metrics.csv / config.json）")


def run_evaluate(args):
    model = load_model(args.checkpoint)
    successes = evaluate(model["logits"], args.episodes, args.seed)
    print(f"评估成功：{successes}/{args.episodes}")
    print(f"精确成功概率：{exact_success(model['logits']):.6f}")
    print("价值预测：", [round(x, 4) for x in model["values"]])
    print("当前策略精确 V：", [round(x, 4) for x in exact_values(model["logits"], model["gamma"])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    returns = commands.add_parser("returns", help="手算与代码核对回报、自举")
    returns.add_argument("--rewards", nargs="+", type=float, default=[0, 0, 1])
    returns.add_argument("--bootstrap", type=float, default=0.0)
    returns.set_defaults(func=run_returns)
    value = commands.add_parser("value", help="多次回报怎样训练价值预测")
    value.add_argument("--returns", nargs="+", type=float, default=[1, 0, 1, 0])
    value.add_argument("--prediction", type=float, default=0.8)
    value.add_argument("--steps", type=int, default=5)
    value.add_argument("--lr", type=float, default=0.5)
    value.set_defaults(func=run_value)
    one_update = commands.add_parser("update", help="查看一条样本如何更新动作概率")
    one_update.add_argument("--action", choices=["correct", "wrong"], default="correct")
    one_update.add_argument("--advantage", type=float, default=0.3)
    one_update.add_argument("--logit", type=float, default=0.0)
    one_update.add_argument("--lr", type=float, default=0.1)
    one_update.set_defaults(func=run_update)
    rollout = commands.add_parser("rollout", help="逐步查看状态、动作、奖励、回报与优势")
    rollout.add_argument("--checkpoint")
    rollout.add_argument("--episodes", type=int, default=5)
    rollout.set_defaults(func=run_rollout)
    train = commands.add_parser("train", help="采样、更新、保存和评估的完整闭环")
    train.add_argument("--iterations", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--actor-lr", type=float, default=1.0)
    train.add_argument("--critic-lr", type=float, default=0.2)
    train.add_argument("--eval-episodes", type=int, default=500)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--out", default="runs/first")
    train.set_defaults(func=run_train)
    evaluation = commands.add_parser("evaluate", help="独立加载保存的策略进行评估")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--episodes", type=int, default=500)
    evaluation.set_defaults(func=run_evaluate)
    for command in (returns, rollout, train):
        command.add_argument("--gamma", type=float, default=0.9)
    for command in (rollout, train, evaluation):
        command.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"{key} 必须是有限数")
    if hasattr(args, "gamma") and not 0 <= args.gamma <= 1:
        parser.error("gamma 必须在 [0, 1] 之间")
    for key in ("episodes", "batch_size", "eval_episodes", "log_every"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"{key} 必须大于 0")
    for key in ("iterations", "steps", "lr", "actor_lr", "critic_lr"):
        if hasattr(args, key) and getattr(args, key) < 0:
            parser.error(f"{key} 不能小于 0")
    args.func(args)


if __name__ == "__main__":
    main()
