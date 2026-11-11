"""辅助观察工具：复现当前固定动作实验，保存真实渲染帧和慢放 GIF。"""

import argparse
import csv
import math
import os
from pathlib import Path

# 在远程终端中离屏渲染；不需要图形桌面或音频设备。
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action", type=int, choices=[0, 1], default=0)
    parser.add_argument("--out", default="runs/cartpole-first")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames, rows = [], []

    def record(step, observation, reward, terminated, truncated):
        angle = math.degrees(float(observation[2]))
        rows.append({
            "step": step, "action": "" if step == 0 else args.action,
            "cart_position": float(observation[0]),
            "cart_velocity": float(observation[1]), "pole_angle_deg": angle,
            "pole_angular_velocity": float(observation[3]), "reward": reward,
            "terminated": terminated, "truncated": truncated,
        })
        frame = Image.fromarray(env.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 48), fill="white")
        draw.text((12, 8), f"step={step}   action={'initial' if step == 0 else args.action}"
                  f"   angle={angle:.2f} deg   limit=12 deg", fill="black")
        draw.text((12, 28), f"terminated={terminated}   truncated={truncated}"
                  "   playback: 10x slower", fill="red" if terminated else "black")
        frames.append(frame)

    try:
        observation, _ = env.reset(seed=args.seed)
        record(0, observation, 0.0, False, False)
        step = 0
        while True:
            observation, reward, terminated, truncated, _ = env.step(args.action)
            step += 1
            record(step, observation, reward, terminated, truncated)
            if terminated or truncated:
                break
    finally:
        env.close()

    # 模拟每步 20ms，这里每帧显示 200ms，慢放 10 倍，便于观察。
    frames[0].save(output / "episode.gif", save_all=True, append_images=frames[1:],
                   duration=200, loop=0)
    # 静态对照图便于在编辑器中检查初始、第 3、6、8、9 步等关键帧。
    indices = sorted({0, len(frames)//3, 2*len(frames)//3, len(frames)-2, len(frames)-1})
    sheet = Image.new("RGB", (frames[0].width * len(indices), frames[0].height), "white")
    for column, index in enumerate(indices):
        sheet.paste(frames[index], (column * frames[0].width, 0))
    sheet.save(output / "keyframes.png")
    with (output / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"seed={args.seed}，固定动作={args.action}，总步数={step}")
    print(f"最终角度={rows[-1]['pole_angle_deg']:.4f} 度，terminated={terminated}，truncated={truncated}")
    print(f"GIF：{output / 'episode.gif'}")
    print(f"关键帧：{output / 'keyframes.png'}")
    print(f"逐步数据：{output / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
