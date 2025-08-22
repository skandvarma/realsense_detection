
# realsense_detection

> A project for real-time object detection and Simultaneous Localization and Mapping (SLAM) using Intel RealSense cameras, specifically optimized for the Intel RealSense 435i.

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
  - [Running the Unified System](#running-the-unified-system)
  - [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project integrates object detection and SLAM to provide a comprehensive understanding of the environment using Intel RealSense cameras. It is specifically tested and optimized for use with the Intel RealSense 435i. The primary entry point for utilizing both SLAM and detection functionalities is the `run_unified_system.py` script.

## Features

- Real-time object detection using YOLO and DETR models
- SLAM-based localization and mapping
- Integration with Intel RealSense cameras (optimized for 435i)
- Modular design for easy extension and customization

> Add specific details on the object detection models and SLAM algorithms used.  For example: "Utilizes YOLOv5 for object detection and ORB-SLAM2 for SLAM."

## Getting Started

These instructions will guide you through setting up the project on your local machine.

### Prerequisites

- Python 3.10 or higher
- Intel RealSense SDK 2.0 ([Installation Guide](https://www.intelrealsense.com/how-to-install-the-intel-realsense-sdk/))
- CUDA 12.9
- NVIDIA 3060
- Pip package manager

> List any other specific hardware or software requirements.  Also, you can add links to download CUDA drivers from NVIDIA.

### Installation

1. Clone the repository:

bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows
4.  **Download YOLO Models:** You are required to manually download the necessary YOLO models and place them in the appropriate directory (e.g., `models/yolo`).  Refer to the documentation for specific instructions on obtaining these models.

5.  **DETR Models:** The DETR models will be automatically downloaded during the first run of the detection module.

### Usage

#### Running the Unified System

To run the integrated SLAM and object detection system, execute the following command:

The system's behavior can be configured using a configuration file (e.g., `config.yaml`).

yaml
# Example configuration file (config.yaml)
camera:
  serial_number: "<camera_serial_number>"
  resolution: [640, 480]
  fps: 30

detection:
  model_path: "path/to/your/detection/model.pth" # Path to your YOLO model
  confidence_threshold: 0.5

slam:
  # SLAM related parameters here
  pass
> Modify the `config.yaml` file to suit your specific needs, such as camera settings, detection thresholds, and SLAM parameters.

## Project Structure

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Submit a pull request.

## License

