# 🚗 park_n_pi
> 🎓 MSc Thesis Project - University of Pannonia

## 📝 Project Overview

This project focuses on developing a reinforcement learning (RL)-based goal navigation system for an autonomous car platform using a Raspberry Pi 5 and the Raspbot V2 chassis.  
The core objective is to enable the robot to reach designated goal poses efficiently and reliably using deep reinforcement learning techniques, serving as a foundation for future autonomous parking maneuvers.

All experiments and evaluations presented in this project were conducted within a high-fidelity Webots simulation environment, integrated with ROS 2 Jazzy and Gymnasium.

Real-world deployment onto the Raspberry Pi 5 is planned for future extensions.

## ✨ Features

- PPO-based reinforcement learning policy (Stable-Baselines3)
- ROS 2 Jazzy full simulation integration
- Custom vectorized Webots environments for faster training
- Curriculum learning for progressive goal-reaching difficulty
- Dockerized setup for easy reproducibility
- TensorBoard support for training monitoring
- Modular architecture supporting future real-world deployment

## 📁 Project Structure

```plaintext
park_n_pi/
├── docker/
│   └── Dockerfile
├── ros_park_n_pi/
│   ├── src/
│   │   ├── raspbot_rl_env/
│   │   ├── raspbot_rl_interface/
│   │   ├── webots_py_ros2_driver/
│   │   └── raspbotv2_bringup/
├── scripts/
│   ├── run_wo_vecenv.py
│   ├── eval_wo_norm.py
│   └── (etc.)
├── logs/
│   ├── checkpoints/
│   ├── tensorboard/
├── resource/
├── README.md
├── .gitignore
└── LICENSE
```

## 🚀 Getting Started

First, build the Docker image:

```
docker build -f docker/Dockerfile -t park_n_pi_docker .
```

Then start the container using Docker Compose:

```
cd docker
docker compose up -d
```

Enter the container:

```
docker exec -it park_n_pi_dev bash
```

Inside the container, launch the Webots simulation and ROS2 nodes:

```
ros2 launch raspbotv2_bringup rl_training.launch.py num_robots:=8
```

Open a new terminal and connect again to the container:

```
docker exec -it park_n_pi_dev bash
cd src/park_n_pi/ros_park_n_pi/src/raspbot_rl_env/scripts/
python3 run_wo_vecenv.py --num_envs 8 --total_timesteps 1_500_000 --tb_log_name RunName
```

This starts the reinforcement learning training loop!
## ⚙️ Usage

- Simulation and ROS2 nodes start automatically.

- Training runs with Stable-Baselines3 agents.

- Logs and models are saved under /home/dev_ws/src/park_n_pi/logs/

Stopping everything:

```
docker compose down
```

Rebuilding if needed:
```
docker compose down
docker compose build --no-cache
docker compose up -d
```


## 🛠️ Technologies Used

- Docker

- ROS2 Jazzy

- Webots R2025a

- Stable-Baselines3

- Python 3.10+

- Gymnasium

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/targabor/park_n_pi/blob/main/LICENSE) file for details.
