# What is the base image to pull
FROM python:3.7.2-stretch
# FROM nvidia/vulkan:1.1.121
# FROM nvidia/opengl:runtime
# FROM nvidia/cuda

# Update package lists
RUN apt-get update

# Install xvfb to execute the Unity build in the container
RUN apt-get install -y xvfb
RUN apt-get install -y screen

# Copy the executable and _Data directory to /unity_player
RUN mkdir /unity_player
# COPY Player/Linux.x86_64 /unity_player/linux_build.x86_64
# COPY Player/Linux_Data /unity_player/linux_build_Data

# for 2019.2+ version of the Unity Editor
# COPY Player/UnityPlayer.so /unity_player/UnityPlayer.so

WORKDIR unity_player
CMD screen -d -m ./start.sh
