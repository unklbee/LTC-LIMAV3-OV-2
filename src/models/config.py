# src/models/config.py
"""
Application configuration model with validation using Pydantic.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, DirectoryPath, FilePath
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import os
from enum import Enum
import json


class DeviceType(str, Enum):
    """Available device types"""
    CPU = "CPU"
    GPU = "GPU"
    AUTO = "AUTO"
    NPU = "NPU"


class Theme(str, Enum):
    """UI themes"""
    DARK = "dark"
    LIGHT = "light"


class Language(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    INDONESIAN = "id"
    CHINESE = "zh"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppConfig(BaseSettings):
    """
    Main application configuration with validation.
    Can be configured via environment variables with LIMA_ prefix.
    """

    # === Paths Configuration ===
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Project root directory"
    )

    model_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing model files"
    )

    data_dir: Optional[Path] = Field(
        default=None,
        description="Directory for data storage"
    )

    db_path: Optional[Path] = Field(
        default=None,
        description="SQLite database file path"
    )

    weights_path: Optional[Path] = Field(
        default=None,
        description="Default model weights path"
    )

    log_dir: Optional[Path] = Field(
        default=None,
        description="Directory for log files"
    )

    # === Detection Settings ===
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )

    nms_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Non-maximum suppression threshold"
    )

    vehicle_classes: List[str] = Field(
        default=["bicycle", "car", "motorbike", "bus", "truck"],
        description="Vehicle classes to detect"
    )

    input_size: tuple[int, int] = Field(
        default=(640, 640),
        description="Model input size (width, height)"
    )

    # === Performance Settings ===
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Inference device"
    )

    use_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )

    batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Batch size for inference"
    )

    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of worker threads"
    )

    buffer_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Frame buffer size"
    )

    # === Database Settings ===
    host_id: str = Field(
        default="default-host",
        description="Host identifier for multi-camera setup"
    )

    camera_id: str = Field(
        default="camera-1",
        description="Camera identifier"
    )

    db_pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Database connection pool size"
    )

    count_save_interval: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Interval for saving counts to database (seconds)"
    )

    # === API Settings ===
    api_url: Optional[str] = Field(
        default=None,
        description="API endpoint for remote data submission"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API authentication key"
    )

    api_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="API request timeout (seconds)"
    )

    api_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for API requests"
    )

    # === UI Settings ===
    theme: Theme = Field(
        default=Theme.DARK,
        description="UI theme"
    )

    language: Language = Field(
        default=Language.ENGLISH,
        description="UI language"
    )

    window_width: int = Field(
        default=1400,
        ge=800,
        le=3840,
        description="Default window width"
    )

    window_height: int = Field(
        default=800,
        ge=600,
        le=2160,
        description="Default window height"
    )

    # === Logging Settings ===
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )

    log_to_file: bool = Field(
        default=True,
        description="Enable logging to file"
    )

    log_max_size: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum log file size in bytes"
    )

    log_backup_count: int = Field(
        default=5,
        description="Number of log file backups to keep"
    )

    # === Tracking Settings ===
    track_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Tracking confidence threshold"
    )

    track_buffer: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Track buffer size (frames)"
    )

    max_age: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Maximum frames to keep track alive without detection"
    )

    # === Counting Settings ===
    enable_speed_estimation: bool = Field(
        default=True,
        description="Enable vehicle speed estimation"
    )

    pixel_per_meter: float = Field(
        default=10.0,
        gt=0,
        description="Pixels per meter for speed calculation"
    )

    speed_limit: float = Field(
        default=60.0,
        gt=0,
        description="Speed limit for violation detection (km/h)"
    )

    # === Export Settings ===
    export_format: str = Field(
        default="excel",
        pattern="^(excel|csv|json|pdf|html)$",
        description="Default export format"
    )

    include_charts: bool = Field(
        default=True,
        description="Include charts in exports"
    )

    # === Advanced Settings ===
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    profile_performance: bool = Field(
        default=False,
        description="Enable performance profiling"
    )

    save_detections: bool = Field(
        default=False,
        description="Save detection images"
    )

    detections_dir: Optional[Path] = Field(
        default=None,
        description="Directory for saving detection images"
    )

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_prefix = "LIMA_"
        case_sensitive = False

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to set up paths"""
        # Set up directory paths
        if self.model_dir is None:
            self.model_dir = self.project_root / "models"

        if self.data_dir is None:
            self.data_dir = self.project_root / "data"

        if self.log_dir is None:
            self.log_dir = self.project_root / "logs"

        # Create directories if they don't exist
        for dir_path in [self.model_dir, self.data_dir, self.log_dir]:
            if dir_path and not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

        # Set database path
        if self.db_path is None:
            self.db_path = self.data_dir / "traffic_counts.db"

        # Ensure database parent directory exists
        if self.db_path and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Set weights path
        if self.weights_path is None and self.model_dir:
            # Look for model files
            for ext in ['.xml', '.onnx', '.engine']:
                model_files = list(self.model_dir.glob(f'*{ext}'))
                if model_files:
                    self.weights_path = model_files[0]
                    break

            if self.weights_path is None:
                self.weights_path = self.model_dir / "yolov7-tiny.onnx"

        # Set detections directory
        if self.save_detections and self.detections_dir is None:
            self.detections_dir = self.data_dir / "detections"
            if not self.detections_dir.exists():
                self.detections_dir.mkdir(parents=True, exist_ok=True)

    @field_validator("vehicle_classes")
    @classmethod
    def validate_vehicle_classes(cls, v):
        """Ensure vehicle classes are valid"""
        valid_classes = {
            "person", "bicycle", "car", "motorbike", "aeroplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe"
        }

        # Check if all specified classes are valid
        invalid = set(v) - valid_classes
        if invalid:
            raise ValueError(f"Invalid vehicle classes: {invalid}")

        return v

    def get_inference_device(self) -> str:
        """Get device string for inference backend"""
        if self.device == DeviceType.AUTO:
            return "AUTO"
        elif self.device == DeviceType.GPU:
            return "GPU"
        elif self.device == DeviceType.NPU:
            return "NPU"
        else:
            return "CPU"

    def get_log_level_int(self) -> int:
        """Get logging level as integer"""
        import logging
        return getattr(logging, self.log_level.value)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return self.model_dump(exclude_none=True, mode='json')

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file"""
        path = Path(path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = self.model_dump(exclude_none=True, mode='json')

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        data = convert_paths(data)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AppConfig":
        """Load configuration from file"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert string paths back to Path objects
            def convert_strings_to_paths(obj, path_fields):
                if isinstance(obj, dict):
                    result = {}
                    for k, v in obj.items():
                        if k in path_fields and isinstance(v, str):
                            result[k] = Path(v)
                        elif isinstance(v, dict):
                            result[k] = convert_strings_to_paths(v, path_fields)
                        else:
                            result[k] = v
                    return result
                return obj

            # List of fields that should be Path objects
            path_fields = {
                'project_root', 'model_dir', 'data_dir', 'db_path',
                'weights_path', 'log_dir', 'detections_dir'
            }

            data = convert_strings_to_paths(data, path_fields)

            return cls(**data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {path}: {e}")


class CountingLineConfig(BaseSettings):
    """Configuration for a counting line"""

    id: str = Field(..., description="Unique line identifier")
    start_point: tuple[float, float] = Field(..., description="Line start point (x, y)")
    end_point: tuple[float, float] = Field(..., description="Line end point (x, y)")
    direction_vector: Optional[tuple[float, float]] = Field(None, description="Direction vector")
    count_forward: bool = Field(True, description="Count forward crossings")
    count_backward: bool = Field(True, description="Count backward crossings")
    active: bool = Field(True, description="Whether line is active")


class CountingZoneConfig(BaseSettings):
    """Configuration for a counting zone"""

    id: str = Field(..., description="Unique zone identifier")
    polygon: List[tuple[float, float]] = Field(..., description="Zone polygon points")
    entry_lines: List[str] = Field(default_factory=list, description="Entry line IDs")
    exit_lines: List[str] = Field(default_factory=list, description="Exit line IDs")
    max_time_inside: float = Field(300.0, description="Maximum time allowed in zone (seconds)")


class CameraConfig(BaseSettings):
    """Configuration for a camera source"""

    id: str = Field(..., description="Camera identifier")
    name: str = Field(..., description="Camera name")
    source: Union[int, str] = Field(..., description="Camera source (index or URL)")
    roi_polygon: Optional[List[tuple[float, float]]] = Field(None, description="ROI polygon")
    counting_lines: List[CountingLineConfig] = Field(default_factory=list)
    counting_zones: List[CountingZoneConfig] = Field(default_factory=list)
    fps: float = Field(30.0, description="Camera FPS")
    resolution: tuple[int, int] = Field((1920, 1080), description="Camera resolution")


def create_default_config() -> AppConfig:
    """Create default configuration"""
    return AppConfig()


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables"""
    return AppConfig()


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = load_config_from_env()
    return _config


def set_config(config: AppConfig):
    """Set global configuration instance"""
    global _config
    _config = config