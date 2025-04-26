# eval_policy.py
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from raspbot_rl_env.env import RaspbotEnv  

def make_env():
    return RaspbotEnv(namespace="RaspbotV2_1")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument("model_path", type=str, help="Path to the saved model .zip file.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vec_env = DummyVecEnv([make_env])

    model = PPO.load(
        args.model_path,
        env=vec_env,
        device=device,
    )

    success = 0
    collisions = 0
    total_rewards = []
    total_steps = []

    for i in range(args.episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done:
            steps += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]

        print(f"[EP {i+1}] Reward: {ep_reward:.2f} | Steps: {steps} | Done: {done}")
        if info[0].get("is_success"):
            success += 1
        if info[0].get("collided"):
            collisions += 1
        total_rewards.append(ep_reward)
        total_steps.append(steps)

    print(f"\nSuccess Rate: {success}/{args.episodes}")
    print(f"Collision Rate: {collisions}/{args.episodes}")
    print(f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Avg Steps: {sum(total_steps)/len(total_steps):.2f}")

if __name__ == "__main__":
    main()
