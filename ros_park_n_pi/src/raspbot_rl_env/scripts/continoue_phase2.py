import gymnasium as gym
from raspbot_rl_env.env import RaspbotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import os
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

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


# === Logging Callback ===
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


# === Env Factory ===
def make_env(rank):
    def _init():
        return RaspbotEnv(namespace=f"RaspbotV2_{rank}")
    return _init


# === Continue Training ===
def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Continuing training on device: {device}")

    log_dir = "./logs"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_envs = 8
    env_fns = [make_env(i) for i in range(1, num_envs + 1)]
    vec_env = SubprocVecEnv(env_fns)

    # === Load from previous checkpoint ===
    model_path = os.path.join(checkpoint_dir, "working_phase_1.zip")
    model = PPO.load(
        model_path,
        env=vec_env,
        device=device,
        custom_objects={"features_extractor_class": CustomFeatureExtractor}
    )

    def calculate_entropy_coef(step):
        initial_coef = 0.01
        final_coef = 0.001
        progress = step / 600_000
        return initial_coef + progress * (final_coef - initial_coef)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_raspbot_phase2",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    total_timesteps = 600_000  # Continue for more steps
    checkpoint_interval = 100_000

    for i in range(0, total_timesteps, checkpoint_interval):
        model.ent_coef = calculate_entropy_coef(i)
        model.ent_coef_tensor = th.tensor(model.ent_coef, device=device)
        model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, EnhancedLoggingCallback()],
            tb_log_name="PPO_31_no_norm_phase2_2"
        )
        model.save(os.path.join(checkpoint_dir, f"ppo_raspbot_phase2_checkpoint_{i}"))

    model.save(os.path.join(checkpoint_dir, "ppo_raspbot_phase2_final"))
    vec_env.close()


if __name__ == "__main__":
    main()
