import os
from raspbot_rl_env.env import RaspbotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym
from collections import defaultdict


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)  # will be set below

        # --- Extract shapes ---
        lidar_shape = observation_space["lidar"].shape[0]            # 180
        goal_rel_shape = observation_space["goal_relative"].shape[0] # 3
        goal_dist_shape = observation_space["goal_distance"].shape[0] # 1

        # --- LiDAR: 1D CNN ---
        self.lidar_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output dim
        with th.no_grad():
            sample = th.zeros((1, 1, lidar_shape))
            n_lidar_features = self.lidar_net(sample).shape[1]

        # --- Goal-related data (relative + distance): small MLP ---
        self.goal_net = nn.Sequential(
            nn.Linear(goal_rel_shape + goal_dist_shape, 32),
            nn.ReLU()
        )

        # --- Final combined net ---
        self.combined_net = nn.Sequential(
            nn.Linear(n_lidar_features + 32, 256),
            nn.ReLU()
        )

        self._features_dim = 256  # Important for SB3

    def forward(self, obs):
        # LiDAR input: (batch, 180) → (batch, 1, 180)
        lidar = obs["lidar"].unsqueeze(1)
        lidar_feat = self.lidar_net(lidar)

        # Combine goal_relative and goal_distance
        goal_info = th.cat([obs["goal_relative"], obs["goal_distance"]], dim=1)
        goal_feat = self.goal_net(goal_info)

        # Concatenate and process
        combined = th.cat([lidar_feat, goal_feat], dim=1)
        return self.combined_net(combined)


# === Enhanced Logging Callback ===

class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.successes = []
        self.collisions = []
        self.reward_components = defaultdict(list)

        self.total_successes = 0 

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.successes.append(info.get("is_success", 0))
                self.collisions.append(info.get("collided", 0))

                self.total_successes += int(info.get("is_success", 0))  # 👈 Increment global

                breakdown = info.get("reward_breakdown", {})
                for key, value in breakdown.items():
                    self.reward_components[key].append(value)
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            success_rate = sum(self.successes) / len(self.successes)
            collision_rate = sum(self.collisions) / len(self.collisions)

            self.logger.record("custom/avg_reward", avg_reward)
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/collision_rate", collision_rate)
            self.logger.record("custom/ent_coef", self.model.ent_coef)

            # Log component-wise reward breakdown
            for key, values in self.reward_components.items():
                self.logger.record(f"reward_components/{key}", sum(values) / len(values))

            # Log cumulative successes
            self.logger.record("custom/total_successes", self.total_successes)

            # Clear episode stats
            self.episode_rewards.clear()
            self.successes.clear()
            self.collisions.clear()
            self.reward_components.clear()


# === Env Factory ===
def make_env(rank):
    def _init():
        return RaspbotEnv(namespace=f"RaspbotV2_{rank}")
    return _init


def main():
    # === Setup ===
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    log_dir = "./logs"
    tb_log = os.path.join(log_dir, "tensorboard", "PPO_15_0")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_name = "ppo_raspbot_checkpoint_1000000.zip"
    vecnormalize_path = os.path.join(checkpoint_dir, "vecnormalize_1000000.pkl")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Training Parameters ===
    start_timestep = 1_100_000
    total_timesteps = 2_100_000


    # === Load VecNormalize + Env ===
    num_envs = 6
    env_fns = [make_env(i) for i in range(1, num_envs + 1)]
    raw_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize.load(vecnormalize_path, raw_env)
    vec_env.norm_obs = True
    vec_env.norm_reward = True
    vec_env.training = True
    new_logger = configure(tb_log, ["stdout", "tensorboard"])
    
    # === Load model from checkpoint ===
    old_model_path = os.path.join(checkpoint_dir, checkpoint_name)
    # old_model = PPO.load(old_model_path, env=vec_env, device=device, custom_objects={"CustomFeatureExtractor": CustomFeatureExtractor})
    old_model = PPO.load(
        old_model_path,
        device=device,
        custom_objects={
            "CustomFeatureExtractor": CustomFeatureExtractor,
            "policy_kwargs": dict(features_extractor_class=CustomFeatureExtractor),
        }
    )
    old_model.set_env(vec_env)
    old_model.set_logger(new_logger)

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="ppo_raspbot",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # === Training Loop ===
    checkpoint_interval = 100_000
    for step in range(start_timestep, total_timesteps, checkpoint_interval):
        print(f"Step {step} | Entropy Coef: {old_model.ent_coef}")

        old_model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, EnhancedLoggingCallback()],
        )
        old_model.save(os.path.join(checkpoint_dir, f"ppo_raspbot_checkpoint_{step}"))
        vec_env.save(os.path.join(checkpoint_dir, f"vecnormalize_{step}.pkl"))
        print(f"Checkpoint saved at timestep {step}")

    vec_env.close()


if __name__ == "__main__":
    main()
