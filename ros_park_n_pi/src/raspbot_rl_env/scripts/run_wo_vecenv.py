import gymnasium as gym
from raspbot_rl_env.env import RaspbotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import os
from collections import defaultdict
import argparse


# === Custom Feature Extractor ===
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        lidar_shape = observation_space["lidar"].shape[0]
        goal_rel_shape = observation_space["goal_relative"].shape[0]
        goal_dist_shape = observation_space["goal_distance"].shape[0]

        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            sample = th.zeros((1, 1, lidar_shape))
            n_lidar_features = self.lidar_net(sample).shape[1]

        self.goal_net = nn.Sequential(
            nn.Linear(goal_rel_shape + goal_dist_shape, 32),
            nn.ReLU()
        )

        self.combined_net = nn.Sequential(
            nn.Linear(n_lidar_features + 32, 256),
            nn.ReLU()
        )

        self._features_dim = 256

    def forward(self, obs):
        lidar = obs["lidar"].unsqueeze(1)
        goal_info = th.cat([obs["goal_relative"], obs["goal_distance"]], dim=1)
        return self.combined_net(th.cat([self.lidar_net(lidar), self.goal_net(goal_info)], dim=1))


# === Enhanced Logging Callback ===
class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.successes = []
        self.collisions = []
        self.reward_components = defaultdict(list)
        self.total_successes = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.successes.append(info.get("is_success", 0))
                self.collisions.append(info.get("collided", 0))
                self.total_successes += int(info.get("is_success", 0))
                for key, value in info.get("reward_breakdown", {}).items():
                    self.reward_components[key].append(value)
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            self.logger.record("custom/avg_reward", sum(self.episode_rewards) / len(self.episode_rewards))
            self.logger.record("custom/success_rate", sum(self.successes) / len(self.successes))
            self.logger.record("custom/collision_rate", sum(self.collisions) / len(self.collisions))
            self.logger.record("custom/ent_coef", self.model.ent_coef)
            for key, values in self.reward_components.items():
                self.logger.record(f"reward_components/{key}", sum(values) / len(values))
            self.logger.record("custom/total_successes", self.total_successes)
            self.episode_rewards.clear()
            self.successes.clear()
            self.collisions.clear()
            self.reward_components.clear()


# === Environment Factory ===
def make_env(rank):
    def _init():
        return RaspbotEnv(namespace=f"RaspbotV2_{rank}")
    return _init


# === Main Training Script ===
def main():
    parser = argparse.ArgumentParser(description="Train PPO on Raspbot RL environment.")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments (robots).")
    parser.add_argument("--total_timesteps", type=int, default=1_600_000, help="Total training timesteps.")
    parser.add_argument("--tb_log_name", type=str, default="PPO_default", help="Tensorboard log folder name.")
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    log_dir = "./logs"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_fns = [make_env(i) for i in range(1, args.num_envs + 1)]
    vec_env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2000,
        batch_size=500,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_raspbot",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    total_steps = args.total_timesteps - 100_000

    def calculate_entropy_coef(step):
        initial_coef = 0.01
        final_coef = 0.001
        progress = step / total_steps
        return initial_coef + progress * (final_coef - initial_coef)

    checkpoint_interval = 100_000
    for i in range(0, args.total_timesteps, checkpoint_interval):
        model.ent_coef = calculate_entropy_coef(i)
        model.ent_coef_tensor = th.tensor(model.ent_coef, device=device)
        model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, EnhancedLoggingCallback()],
            tb_log_name=args.tb_log_name,
        )
        model.save(os.path.join(checkpoint_dir, f"ppo_raspbot_checkpoint_{i}"))
        print(f"Checkpoint saved at timestep {i}")

    model.save(os.path.join(checkpoint_dir, "ppo_raspbot_final"))
    vec_env.close()


if __name__ == "__main__":
    main()
