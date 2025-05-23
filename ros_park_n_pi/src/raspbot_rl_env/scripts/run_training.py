import gymnasium as gym
from raspbot_rl_env.env import RaspbotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import os
from stable_baselines3.common.callbacks import BaseCallback
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


def make_env(rank):
    def _init():
        env = RaspbotEnv(namespace=f"RaspbotV2_{rank}")
        return env
    return _init


def main():
    # GPU Check
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Paths
    log_dir = "./logs"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Environment Setup
    num_envs = 6
    env_fns = [make_env(i) for i in range(1, num_envs + 1)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)


    # Learning Rate (Constant)
    def constant_schedule(lr=3e-4):
        return lambda _: lr

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="ppo_raspbot",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    total_timesteps = 1_100_000
    total_steps = 1_000_000

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
    )

    

    # PPO Model
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        #policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4, 
        n_steps=2000,            # per environment if using VecEnv
        batch_size=240, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        ent_coef=0.01,  # Increased entropy for better exploration
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device=device,
    )

    # Training Loop
    checkpoint_interval = 100_000
    
    def calculate_entropy_coef(step):
        initial_coef = 0.01
        final_coef = 0.001
        progress = step / total_steps
        return initial_coef + progress * (final_coef - initial_coef)

    for i in range(0, total_timesteps, checkpoint_interval):
        print(f"Step {i} | Entropy Coef: {model.ent_coef}")
        model.ent_coef = calculate_entropy_coef(i / total_timesteps)
        model.ent_coef_tensor = th.tensor(model.ent_coef, device=device)
        model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, EnhancedLoggingCallback()],
            tb_log_name="PPO_27",
        )
        model.save(os.path.join(checkpoint_dir, f"ppo_raspbot_checkpoint_{i}"))
        vec_env.save(os.path.join(checkpoint_dir, f"vecnormalize_{i}.pkl"))
        print(f"Checkpoint saved at timestep {i}")

    model.save(os.path.join(checkpoint_dir, "ppo_raspbot_final"))
    vec_env.save(os.path.join(checkpoint_dir, "vecnormalize_final.pkl"))

    vec_env.close()


if __name__ == "__main__":
    main()
