import gymnasium as gym
from raspbot_rl_env.env import RaspbotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch


def make_env(rank, log_dir=None):
    def _init():
        env = RaspbotEnv(namespace=f"RaspbotV2_{rank}")
        # Optional: Add environment wrappers here if needed
        return env

    return _init


def main():
    # GPU Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Environment Setup
    num_envs = 4
    vec_env = SubprocVecEnv([make_env(i) for i in range(1, num_envs + 1)])

    # Learning Rate Schedule (Linear Decay)
    def lr_schedule(progress_remaining):
        return 3e-4 * progress_remaining

    # Checkpoint and Evaluation Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50,000 steps
        save_path="./logs/checkpoints/",
        name_prefix="ppo_raspbot",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Model Initialization
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=lr_schedule,
        n_steps=4096,  # Increased from previous version
        batch_size=1024,
        n_epochs=20,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Generalized Advantage Estimation lambda
        clip_range=0.2,  # PPO clipping parameter
        ent_coef=0.01,  # Entropy coefficient for exploration
        tensorboard_log="./logs/tensorboard/",
        device=device,
    )

    # Training
    total_timesteps = 10_000_000
    CHECKPOINT_INTERVAL = 100_000

    for i in range(0, total_timesteps, CHECKPOINT_INTERVAL):
        model.learn(
            total_timesteps=CHECKPOINT_INTERVAL,
            reset_num_timesteps=False,
            callback=checkpoint_callback,
        )
        model.save(f"./logs/checkpoints/ppo_raspbot_checkpoint_{i}")
        print(f"Checkpoint saved at timestep {i}")

    vec_env.close()


if __name__ == '__main__':
    main()
