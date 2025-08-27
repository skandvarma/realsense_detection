# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import os
import sys
import numpy as np
import open3d as o3d
import time

from config import ConfigParser
from common import (get_default_dataset, load_rgbd_file_names, save_poses,
                    load_intrinsic, extract_trianglemesh, extract_rgbd_frames)

# Import the D435iCamera class
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
sys.path.insert(0, project_root)

# Import D435iCamera from shared camera manager
from src.camera.realsense_manager import D435iCamera


def slam_with_camera(camera, intrinsic, config, max_frames=1000, map_update_interval=30):
    """SLAM with live camera input, IMU integration, and continuous map visualization."""
    device = o3d.core.Device(config.device)

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config.voxel_size, 16,
                                       config.block_count, T_frame_to_model,
                                       device)

    # Get initial frame to set up dimensions
    rgb_frame, depth_frame = camera.get_frames()
    while rgb_frame is None or depth_frame is None:
        print("Waiting for initial frames...")
        time.sleep(0.1)
        rgb_frame, depth_frame = camera.get_frames()

    height, width = depth_frame.shape
    input_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)

    poses = []
    imu_data = []
    frame_count = 0

    # IMU integration variables
    prev_accel = None
    prev_gyro = None
    prev_timestamp = time.time()
    velocity = np.zeros(3)
    orientation = np.eye(3)  # Rotation matrix

    # Initialize visualizer for continuous map display
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live SLAM Map", width=800, height=600)

    # Initialize with empty geometry
    mesh = o3d.geometry.TriangleMesh()
    vis.add_geometry(mesh)

    print("Starting SLAM with live camera and IMU...")
    print("Press Ctrl+C to stop")
    print(f"Map will update every {map_update_interval} frames")

    try:
        while frame_count < max_frames:
            start = time.time()
            current_timestamp = time.time()

            # Get frames from camera
            rgb_frame, depth_frame = camera.get_frames()

            if rgb_frame is None or depth_frame is None:
                time.sleep(0.01)
                continue

            # Get IMU data
            accel, gyro = camera.get_imu_data()
            dt = current_timestamp - prev_timestamp

            # IMU-based motion prediction for better initial pose estimate
            imu_transform_prediction = np.eye(4)
            if accel is not None and gyro is not None and prev_accel is not None and prev_gyro is not None and dt > 0:
                try:
                    # Simple IMU integration for motion prediction
                    # Angular velocity integration (simplified)
                    if np.linalg.norm(gyro) > 0.01:  # Only if significant rotation
                        angle = np.linalg.norm(gyro) * dt
                        axis = gyro / np.linalg.norm(gyro) if np.linalg.norm(gyro) > 0 else np.array([0, 0, 1])

                        # Rodrigues' rotation formula for small angles
                        K = np.array([[0, -axis[2], axis[1]],
                                      [axis[2], 0, -axis[0]],
                                      [-axis[1], axis[0], 0]])
                        R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                        orientation = orientation @ R_delta

                        # Update rotation part of transform
                        imu_transform_prediction[:3, :3] = R_delta

                    # Linear acceleration integration (with gravity compensation)
                    gravity_compensated_accel = accel - np.array([0, 0, -9.81])  # Remove gravity
                    velocity += gravity_compensated_accel * dt
                    translation = velocity * dt

                    # Only use translation if velocity is reasonable (not too high)
                    if np.linalg.norm(velocity) < 2.0:  # Max 2 m/s
                        imu_transform_prediction[:3, 3] = translation * 0.1  # Scale down for stability

                    # Store IMU data for logging
                    imu_data.append({
                        'frame': frame_count,
                        'timestamp': current_timestamp,
                        'accel': accel.copy() if accel is not None else None,
                        'gyro': gyro.copy() if gyro is not None else None,
                        'predicted_translation': translation,
                        'predicted_rotation': R_delta if 'R_delta' in locals() else np.eye(3)
                    })

                except Exception as e:
                    print(f"IMU integration error: {e}")
                    imu_transform_prediction = np.eye(4)

            # Convert numpy arrays to Open3D Images (ensure CPU creation, then move to device)
            depth_tensor = o3d.core.Tensor(depth_frame.astype(np.uint16), device=o3d.core.Device('CPU:0'))
            color_tensor = o3d.core.Tensor(rgb_frame.astype(np.uint8), device=o3d.core.Device('CPU:0'))

            depth = o3d.t.geometry.Image(depth_tensor).to(device)
            color = o3d.t.geometry.Image(color_tensor).to(device)

            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)

            if frame_count > 0:
                # Use IMU prediction as initial guess for tracking (if available)
                if 'imu_transform_prediction' in locals() and np.any(imu_transform_prediction != np.eye(4)):
                    # Convert IMU prediction to tensor and use as prior
                    T_imu_pred = o3d.core.Tensor(imu_transform_prediction.astype(np.float64))
                    T_initial_guess = T_frame_to_model @ T_imu_pred
                else:
                    T_initial_guess = T_frame_to_model

                result = model.track_frame_to_model(input_frame, raycast_frame,
                                                    config.depth_scale,
                                                    config.depth_max,
                                                    config.odometry_distance_thr)
                T_frame_to_model = T_frame_to_model @ result.transformation

            poses.append(T_frame_to_model.cpu().numpy())
            model.update_frame_pose(frame_count, T_frame_to_model)
            model.integrate(input_frame, config.depth_scale, config.depth_max,
                            config.trunc_voxel_multiplier)
            model.synthesize_model_frame(raycast_frame, config.depth_scale,
                                         config.depth_min, config.depth_max,
                                         config.trunc_voxel_multiplier, False)

            # Update map visualization periodically
            if frame_count % map_update_interval == 0 and frame_count > 0:
                try:
                    # Extract current mesh
                    current_mesh = extract_trianglemesh(model.voxel_grid, config, None)

                    # Update visualization
                    vis.remove_geometry(mesh)
                    mesh = current_mesh
                    vis.add_geometry(mesh)
                    vis.update_geometry(mesh)

                    print(f"Map updated at frame {frame_count}")
                except Exception as e:
                    print(f"Map update failed: {e}")

            # Update visualizer
            vis.poll_events()
            vis.update_renderer()

            # Update previous IMU values and timestamp
            prev_accel = accel.copy() if accel is not None else None
            prev_gyro = gyro.copy() if gyro is not None else None
            prev_timestamp = current_timestamp

            stop = time.time()
            imu_info = ""
            if accel is not None and gyro is not None:
                imu_info = f" | IMU: a={np.linalg.norm(accel):.2f} g={np.linalg.norm(gyro):.2f}"

            print('{:04d} slam takes {:.4}s{}'.format(frame_count, stop - start, imu_info))

            frame_count += 1

    except KeyboardInterrupt:
        print(f"\nStopping SLAM after {frame_count} frames")

    finally:
        vis.destroy_window()

        # Save IMU data log
        if imu_data:
            try:
                import json
                imu_log_file = 'imu_data.json'
                with open(imu_log_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    imu_data_serializable = []
                    for entry in imu_data:
                        serializable_entry = {}
                        for key, value in entry.items():
                            if isinstance(value, np.ndarray):
                                serializable_entry[key] = value.tolist()
                            else:
                                serializable_entry[key] = value
                        imu_data_serializable.append(serializable_entry)

                    json.dump(imu_data_serializable, f, indent=2)
                print(f"IMU data saved to {imu_log_file}")
            except Exception as e:
                print(f"Failed to save IMU data: {e}")

    return model.voxel_grid, poses


def slam(depth_file_names, color_file_names, intrinsic, config):
    n_files = len(color_file_names)
    device = o3d.core.Device(config.device)

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config.voxel_size, 16,
                                       config.block_count, T_frame_to_model,
                                       device)
    depth_ref = o3d.t.io.read_image(depth_file_names[0])
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                             intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                               depth_ref.columns, intrinsic,
                                               device)

    poses = []

    for i in range(n_files):
        start = time.time()

        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)

        if i > 0:
            result = model.track_frame_to_model(input_frame, raycast_frame,
                                                config.depth_scale,
                                                config.depth_max,
                                                config.odometry_distance_thr)
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, config.depth_scale, config.depth_max,
                        config.trunc_voxel_multiplier)
        model.synthesize_model_frame(raycast_frame, config.depth_scale,
                                     config.depth_min, config.depth_max,
                                     config.trunc_voxel_multiplier, False)
        stop = time.time()
        print('{:04d}/{:04d} slam takes {:.4}s'.format(i, n_files,
                                                       stop - start))

    return model.voxel_grid, poses


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
             'reference. It overrides the default config file, but will be '
             'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
                    'Default dataset may be selected from the following options: '
                    '[lounge, bedroom, jack_jack]',
               default='lounge')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='output.npz')
    parser.add('--use_camera',
               help='Use live D435i camera instead of dataset files.',
               action='store_true')
    parser.add('--max_frames',
               help='Maximum number of frames to process from camera.',
               type=int,
               default=1000)
    parser.add('--map_update_interval',
               help='Number of frames between map visualization updates.',
               type=int,
               default=30)
    config = parser.get_config()

    if config.use_camera:
        # Use live camera
        print("Initializing D435i camera...")

        # Create camera config (adjust as needed for your setup)
        camera_config = {
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }

        try:
            camera = D435iCamera(camera_config)
            intrinsic = o3d.core.Tensor(camera.get_intrinsics()).to(o3d.core.Device(config.device))

            print("Camera initialized successfully")
            print(f"Intrinsics matrix:\n{camera.get_intrinsics()}")

            volume, poses = slam_with_camera(camera, intrinsic, config, config.max_frames, config.map_update_interval)

            print('Saving to {}...'.format(config.path_npz))
            volume.save(config.path_npz)
            save_poses('output_camera.log', poses)
            print('Saving finished')

            # Stop camera
            camera.stop()

        except Exception as e:
            print(f"Camera initialization failed: {e}")
            sys.exit(1)

    else:
        # Use dataset files (original functionality)
        if config.path_dataset == '':
            config = get_default_dataset(config)

        # Extract RGB-D frames and intrinsic from bag file.
        if config.path_dataset.endswith(".bag"):
            assert os.path.isfile(
                config.path_dataset), f"File {config.path_dataset} not found."
            print("Extracting frames from RGBD video file")
            config.path_dataset, config.path_intrinsic, config.depth_scale = extract_rgbd_frames(
                config.path_dataset)

        depth_file_names, color_file_names = load_rgbd_file_names(config)
        intrinsic = load_intrinsic(config)

        if not os.path.exists(config.path_npz):
            volume, poses = slam(depth_file_names, color_file_names, intrinsic,
                                 config)
            print('Saving to {}...'.format(config.path_npz))
            volume.save(config.path_npz)
            save_poses('output.log', poses)
            print('Saving finished')
        else:
            volume = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)

    # Load volume if it wasn't created above
    if not config.use_camera and os.path.exists(config.path_npz):
        volume = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)

    mesh = extract_trianglemesh(volume, config, 'output.ply')
    o3d.visualization.draw([mesh])