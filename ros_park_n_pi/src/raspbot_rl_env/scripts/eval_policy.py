# eval_policy.py
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from raspbot_rl_env.env import RaspbotEnv  

def make_env():
    return RaspbotEnv(namespace="RaspbotV2_1") 

def main():
    vec_env = VecNormalize.load("logs/checkpoints/vecnormalize_final.pkl", DummyVecEnv([make_env]))
    vec_env.training = False
    vec_env.norm_reward = False


    model = PPO.load("logs/checkpoints/ppo_raspbot_final.zip", env=vec_env, device="cuda" if torch.cuda.is_available() else "cpu")

    episodes = 10
    success = 0
    collisions = 0
    total_rewards = []

    for i in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            #print(action)
            obs, reward, done, info = vec_env.step(action)
            print(f"Test loop obs distance to goal: {obs["goal_distance"]}")
            print(f"Reward: {reward}")
            ep_reward += reward[0]

        print(f"[EP {i+1}] Reward: {ep_reward:.2f} | Info: {info[0]}")
        if info[0].get("is_success"): success += 1
        if info[0].get("collided"): collisions += 1
        total_rewards.append(ep_reward)

    print(f"\nSuccess Rate: {success}/{episodes}")
    print(f"Collision Rate: {collisions}/{episodes}")
    print(f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")

if __name__ == "__main__":
    main()
