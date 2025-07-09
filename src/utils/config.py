"""
Configuration management and validation system for RealSense object detection project.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigManager:
    """Configuration manager for loading and validating YAML configuration files."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with comprehensive error handling.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the loaded configuration

        Raises:
            ConfigurationError: If configuration cannot be loaded or is invalid
        """
        try:
            config_file = Path(config_path)

            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ConfigurationError(f"Configuration file is empty: {config_path}")

            # Validate the loaded configuration
            ConfigManager._validate_configuration(config)

            # Apply environment variable overrides
            config = ConfigManager._apply_env_overrides(config)

            # Fill in default values
            config = ConfigManager._apply_defaults(config)

            logging.info(f"Configuration loaded successfully from {config_path}")
            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Unexpected error loading configuration: {e}")

    @staticmethod
    def _validate_configuration(config: Dict[str, Any]) -> None:
        """
        Validate the loaded configuration structure and values.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['camera', 'detection', 'integration', 'output']

        # Check for required top-level sections
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required configuration section: {section}")

        # Validate camera configuration
        ConfigManager._validate_camera_config(config['camera'])

        # Validate detection configuration
        ConfigManager._validate_detection_config(config['detection'])

        # Validate integration configuration
        ConfigManager._validate_integration_config(config['integration'])

        # Validate output configuration
        ConfigManager._validate_output_config(config['output'])

        # Cross-section validation
        ConfigManager._validate_cross_section_compatibility(config)

    @staticmethod
    def _validate_camera_config(camera_config: Dict[str, Any]) -> None:
        """Validate camera configuration section."""
        required_fields = ['device_id', 'resolution', 'streams']

        for field in required_fields:
            if field not in camera_config:
                raise ConfigurationError(f"Missing required camera field: {field}")

        # Validate resolution
        resolution = camera_config['resolution']
        if not isinstance(resolution.get('width'), int) or not isinstance(resolution.get('height'), int):
            raise ConfigurationError("Camera resolution width and height must be integers")

        if resolution['width'] <= 0 or resolution['height'] <= 0:
            raise ConfigurationError("Camera resolution must be positive")

        # Validate streams
        streams = camera_config['streams']
        required_streams = ['color', 'depth']

        for stream in required_streams:
            if stream not in streams:
                raise ConfigurationError(f"Missing required stream configuration: {stream}")

            stream_config = streams[stream]
            if 'framerate' in stream_config:
                if not isinstance(stream_config['framerate'], int) or stream_config['framerate'] <= 0:
                    raise ConfigurationError(f"Invalid framerate for {stream} stream")

        # Validate depth settings
        if 'depth' in camera_config:
            depth_config = camera_config['depth']
            min_dist = depth_config.get('min_distance', 0)
            max_dist = depth_config.get('max_distance', 10)

            if min_dist >= max_dist:
                raise ConfigurationError("Depth min_distance must be less than max_distance")

    @staticmethod
    def _validate_detection_config(detection_config: Dict[str, Any]) -> None:
        """Validate detection configuration section."""
        if 'active_model' not in detection_config:
            raise ConfigurationError("Missing active_model in detection configuration")

        active_model = detection_config['active_model']
        if active_model not in ['yolo', 'detr']:
            raise ConfigurationError("active_model must be either 'yolo' or 'detr'")

        # Validate model-specific configuration
        if active_model in detection_config:
            model_config = detection_config[active_model]

            # Validate confidence threshold
            if 'confidence_threshold' in model_config:
                conf_thresh = model_config['confidence_threshold']
                if not 0.0 <= conf_thresh <= 1.0:
                    raise ConfigurationError("confidence_threshold must be between 0.0 and 1.0")

            # Validate input size
            if 'input_size' in model_config:
                input_size = model_config['input_size']
                if not isinstance(input_size, list) or len(input_size) != 2:
                    raise ConfigurationError("input_size must be a list of 2 integers")

                if not all(isinstance(x, int) and x > 0 for x in input_size):
                    raise ConfigurationError("input_size values must be positive integers")

    @staticmethod
    def _validate_integration_config(integration_config: Dict[str, Any]) -> None:
        """Validate integration configuration section."""
        if 'tracking' in integration_config:
            tracking = integration_config['tracking']

            # Validate threshold values
            if 'distance_threshold' in tracking:
                if tracking['distance_threshold'] <= 0:
                    raise ConfigurationError("distance_threshold must be positive")

            if 'disappearance_threshold' in tracking:
                if not isinstance(tracking['disappearance_threshold'], int) or tracking['disappearance_threshold'] <= 0:
                    raise ConfigurationError("disappearance_threshold must be a positive integer")

        if 'performance' in integration_config:
            performance = integration_config['performance']

            if 'target_fps' in performance:
                if not isinstance(performance['target_fps'], int) or performance['target_fps'] <= 0:
                    raise ConfigurationError("target_fps must be a positive integer")

    @staticmethod
    def _validate_output_config(output_config: Dict[str, Any]) -> None:
        """Validate output configuration section."""
        if 'directories' in output_config:
            directories = output_config['directories']

            # Validate directory paths are strings
            for dir_name, dir_path in directories.items():
                if not isinstance(dir_path, str):
                    raise ConfigurationError(f"Directory path for {dir_name} must be a string")

        if 'position_logging' in output_config:
            pos_logging = output_config['position_logging']

            if 'format' in pos_logging:
                valid_formats = ['json', 'csv', 'txt']
                if pos_logging['format'] not in valid_formats:
                    raise ConfigurationError(f"position_logging format must be one of: {valid_formats}")

            if 'coordinate_precision' in pos_logging:
                precision = pos_logging['coordinate_precision']
                if not isinstance(precision, int) or precision < 0:
                    raise ConfigurationError("coordinate_precision must be a non-negative integer")

    @staticmethod
    def _validate_cross_section_compatibility(config: Dict[str, Any]) -> None:
        """Validate compatibility between different configuration sections."""
        # Check camera resolution compatibility with detection input size
        camera_res = config['camera']['resolution']
        active_model = config['detection']['active_model']

        if active_model in config['detection']:
            model_config = config['detection'][active_model]
            input_size = model_config.get('input_size', [640, 640])

            # Warning if input size is significantly different from camera resolution
            width_ratio = abs(camera_res['width'] - input_size[0]) / camera_res['width']
            height_ratio = abs(camera_res['height'] - input_size[1]) / camera_res['height']

            if width_ratio > 0.5 or height_ratio > 0.5:
                logging.warning(
                    f"Camera resolution ({camera_res['width']}x{camera_res['height']}) "
                    f"differs significantly from model input size ({input_size[0]}x{input_size[1]}). "
                    f"This may impact performance."
                )

    @staticmethod
    def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Environment variable mappings
        env_mappings = {
            'REALSENSE_DEVICE_ID': ['camera', 'device_id'],
            'DETECTION_MODEL': ['detection', 'active_model'],
            'CONFIDENCE_THRESHOLD': ['detection', 'yolo', 'confidence_threshold'],
            'OUTPUT_DIR': ['output', 'directories', 'base_output'],
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the correct nested dictionary
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                # Convert value to appropriate type
                final_key = config_path[-1]
                if env_var == 'REALSENSE_DEVICE_ID':
                    current[final_key] = int(env_value)
                elif env_var == 'CONFIDENCE_THRESHOLD':
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value

                logging.info(f"Applied environment override: {env_var} = {env_value}")

        return config

    @staticmethod
    def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing optional configuration fields."""
        defaults = {
            'camera': {
                'filters': {
                    'spatial': {'enabled': False},
                    'temporal': {'enabled': False},
                    'hole_filling': {'enabled': False}
                },
                'alignment': {
                    'align_depth_to_color': True
                }
            },
            'detection': {
                'yolo': {
                    'confidence_threshold': 0.5,
                    'iou_threshold': 0.45,
                    'max_detections': 100
                },
                'detr': {
                    'confidence_threshold': 0.7,
                    'max_detections': 100
                }
            },
            'integration': {
                'performance': {
                    'use_threading': True,
                    'target_fps': 30
                },
                'visualization': {
                    'display_rgb': True,
                    'display_depth': True,
                    'show_fps': True
                }
            },
            'output': {
                'session': {
                    'auto_create_directories': True,
                    'max_log_files': 10
                }
            }
        }

        # Recursively apply defaults
        def apply_recursive_defaults(config_section: Dict[str, Any], defaults_section: Dict[str, Any]) -> None:
            for key, default_value in defaults_section.items():
                if key not in config_section:
                    config_section[key] = default_value
                elif isinstance(default_value, dict) and isinstance(config_section[key], dict):
                    apply_recursive_defaults(config_section[key], default_value)

        for section, section_defaults in defaults.items():
            if section in config:
                apply_recursive_defaults(config[section], section_defaults)

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the configuration file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2, sort_keys=False)

            logging.info(f"Configuration saved to {output_path}")

        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")

    @staticmethod
    def get_nested_value(config: Dict[str, Any], key_path: List[str], default: Any = None) -> Any:
        """
        Get a nested value from configuration using a list of keys.

        Args:
            config: Configuration dictionary
            key_path: List of keys to navigate to the desired value
            default: Default value if key path is not found

        Returns:
            The value at the specified key path or default value
        """
        current = config
        try:
            for key in key_path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default