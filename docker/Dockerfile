# Start from official ROS 2 Jazzy base
FROM ros:jazzy

# Set environment variables for ROS 2
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    ros-dev-tools \
    git \
    sudo \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /home/dev_ws/src
WORKDIR /home/dev_ws/src

# Copy your project into the container
COPY . /home/dev_ws/src/park_n_pi

# Install Webots R2025a from tar.bz2 manually
RUN wget https://github.com/cyberbotics/webots/releases/download/R2025a/webots-R2025a-x86-64.tar.bz2 && \
    tar -xjf webots-*.tar.bz2 && \
    mv webots /usr/local/webots && \
    rm webots-*.tar.bz2

ENV WEBOTS_HOME=/usr/local/webots
ENV PATH=${WEBOTS_HOME}:${PATH}


# Install Python requirements
WORKDIR /home/dev_ws/src/park_n_pi
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Add ROS2 apt repository inside Docker build
RUN apt update && apt install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list && \
    apt update

# Install ROS dependencies
WORKDIR /home/dev_ws
RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# Source the environment automatically when opening shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source /home/dev_ws/install/setup.bash" >> ~/.bashrc

# Set default working directory
WORKDIR /home/dev_ws

# Default command
CMD ["/bin/bash"]
