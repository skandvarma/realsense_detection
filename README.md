# realsense_detection

> A project for real-time object detection and Simultaneous Localization and Mapping (SLAM) using Intel RealSense cameras, specifically optimized for the Intel RealSense 435i. This project utilizes YOLOv5 and DETR for object detection and integrates voice commands using OpenAI's Whisper for a more interactive experience. This project also introduces ROS2 for camera sharing functionalities. Further development can be done on the SLAM pipeline.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running Detection Pipeline](#running-the-detection-pipeline)
  - [Running Obstacle Avoidance Pipeline](#running-the-obstacle-avoidance-pipeline)
  - [Running the SLAM Pipeline](#running-the-slam-pipeline)
  - [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project integrates object detection and SLAM to provide a comprehensive understanding of the environment using Intel RealSense cameras. It is specifically tested and optimized for use with the Intel RealSense 435i. Object detection is implemented using YOLOv5 and DETR. The project also integrates voice commands using OpenAI's Whisper for a more interactive object detection experience. This project also introduces ROS2 for camera sharing functionalities.

The project has been tested in the following environment:
- Python 3.10
- ROS2 Jazzy
- Ubuntu 24.04

## Features

- Real-time object detection using YOLOv5 and DETR models
- SLAM-based localization and mapping
- Integration with Intel RealSense cameras (optimized for 435i)
- Modular design for easy extension and customization
- Configurable parameters for camera and detection modules
- Example configuration files for quick setup
- ROS2 integration for camera sharing
- Voice command integration using OpenAI's Whisper for specifying objects to detect

## Getting Started

These instructions will guide you through setting up the project on your local machine.

### Prerequisites

- **Hardware:**
    - Intel RealSense Camera D435i
    - NVIDIA GPU with CUDA support (Recommended: NVIDIA 3060 or higher)
    - At least 16 GB RAM

- **Software:**
    - Ubuntu 24.04 or higher
    - Python 3.10 or higher
    - Intel RealSense SDK 2.0 ([Installation Guide](https://www.intelrealsense.com/how-to-install-the-intel-realsense-sdk/))
    - CUDA Toolkit 11.0 or higher ([Download CUDA Drivers](https://developer.nvidia.com/cuda-downloads))
    - cuDNN (matching CUDA version)
    - Pip package manager
    - ROS2 Jazzy
    - OpenAI's Whisper (install via pip, see instructions below)

> Ensure that the CUDA and cuDNN versions are compatible with your NVIDIA driver.

### Installation

1. Clone the repository:

bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate  # On Windows
      [https://github.com/skandvarma/ros2_ws](https://github.com/skandvarma/ros2_ws)

   > **Important:** This setup is designed to run on two computers: a host (e.g., Intel NUC) and a client. The host provides the camera feed using the rs-server from the specified repository. Make sure your ROS2 environment is properly sourced on both machines after installing the package. Follow the instructions at [https://github.com/skandvarma/ros2_ws](https://github.com/skandvarma/ros2_ws) for setting up the client. Our pipelines will work on the client side.

5. **Download Models:** The YOLO and Detection Transformer pipelines require downloading models. Ensure you have an active internet connection when running these for the first time, as the models will be downloaded to your host laptop. You are required to manually download the necessary YOLOv5 models (e.g., `yolov5s.pt`, `yolov5m.pt`, etc.) from the [official YOLOv5 repository](https://github.com/ultralytics/yolov5/releases). Place the downloaded model file in the `models/yolo` directory.

## Usage

> This pipeline is designed to run on two computers: the host and the client. The host, in our case an Intel NUC, provides the camera feed. The following instructions apply to the client side.

### Running Detection Pipeline

To run the object detection pipeline, execute the following command:

> When running the detection pipeline with DETR in `config.yaml`, the application will prompt you for a voice command to specify the objects you want to find. Ensure your microphone is set up correctly.

> **Note:**
>
> - Replace `<camera_serial_number>` with the actual serial number of your Intel RealSense camera. If left empty, the system will attempt to automatically detect the camera.
> - Adjust the `confidence_threshold` and `iou_threshold` in the `detection` section to optimize the object detection performance.
> - If you switch the `detector_type` to `detr`, make sure that the confidence threshold is set accordingly. DETR typically requires higher confidence thresholds than YOLOv5.

### Running Obstacle Avoidance Pipeline

> To run the error calculation that is performed via PD control mathematics to find where the robot needs to go run error_calc.py this will subscribe to topics provided by https://github.com/skandvarma/ros2_ws package .

### Running the SLAM Pipeline

> ros2 launch rtabmap_launch rtabmap.launch.py     rtabmap_args:="--delete_db_on_start"     depth_topic:=/camera/camera/aligned_depth_to_color/image_raw     rgb_topic:=/camera/camera/color/image_raw     camera_info_topic:=/camera/camera/color/camera_info     approx_sync:=true     frame_id:=camera_link     use_odom:=false

> Run this command for SLAM and change the parameters accordingly to either save the map or just for viewing the map .

> If you want to run the open3d pipeline , the camera needs to be connected to the computer and can be run through here "python realsense_slam/src/Open3D/examples/python/t_reconstruction_system/enhanced_slam_v2.py"

### Configuration

> Details about the `config.yaml` file and its parameters will be added here.  This is a placeholder.

## Project Structure

-   `data/`: This directory is intended for storing map data, calibration files, or any other persistent data required by the system.
-   `models/`: This directory contains the object detection models. YOLO models should be placed in the `yolo/` subdirectory, while DETR models are automatically downloaded to the `detr/` subdirectory.
-   `scripts/`: This directory can contain utility scripts for tasks such as data processing or evaluation.
-   `src/`: This directory contains the core source code, organized into submodules for camera interface, object detection, SLAM, and utility functions.
-   `config.yaml`: This is the main configuration file that controls the behavior of the system.
-   `requirements.txt`: This file lists the Python dependencies required to run the project.

## Contributing

Contributions are welcome! Please follow these steps:
email on skandvarma2004@gmail.com
