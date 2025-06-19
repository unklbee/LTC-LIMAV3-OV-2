# src/ui/controllers/app_controller.py
"""
Main application controller that coordinates all components.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import numpy as np

from PySide6.QtCore import QObject, Signal, Slot, QTimer
from PySide6.QtWidgets import QMessageBox, QFileDialog

from src.core.pipeline import create_pipeline, OptimizedPipeline
from src.core.detector import create_detector
from src.core.tracker import create_tracker
from src.core.counter import create_counter, CountingLine, VehicleCount
from src.services.database_service import create_database_service, DatabaseService, CountRecord
from src.services.api_service import APIService
from src.models.config import AppConfig
from src.ui.views.main_view import ModernMainWindow
from src.ui.views.dashboard_view import DashboardView
from src.ui.views.video_view import VideoView
from src.ui.views.settings_view import SettingsView

import structlog

logger = structlog.get_logger()


class AppController(QObject):
    """
    Main application controller.

    Responsibilities:
    - Initialize and coordinate all components
    - Handle user interactions
    - Manage application state
    - Process pipeline data
    """

    # Signals
    stats_updated = Signal(dict)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config

        # Components
        self.pipeline: Optional[OptimizedPipeline] = None
        self.db_service: Optional[DatabaseService] = None
        self.api_service: Optional[APIService] = None

        # UI
        self.main_window = ModernMainWindow()
        self.dashboard_view: Optional[DashboardView] = None
        self.video_view: Optional[VideoView] = None
        self.settings_view: Optional[SettingsView] = None

        # State
        self.is_running = False
        self.current_source = None
        self.roi_polygon = None
        self.counting_lines = []

        # Statistics
        self.stats = {
            'fps': 0.0,
            'total_counts': {},
            'active_tracks': 0,
            'count_rate': 0.0,
            'detections': [],
            'avg_speed': None,
            'max_speed': None
        }

        # Setup
        self._setup_ui()
        self._connect_signals()

    async def initialize(self):
        """Initialize all async components"""
        try:
            # Initialize database service
            self.db_service = await create_database_service({
                'db_path': str(self.config.db_path),
                'host_id': self.config.host_id,
                'camera_id': self.config.camera_id,
                'pool_size': 5
            })

            # Initialize API service if configured
            if self.config.api_url:
                self.api_service = APIService(
                    base_url=self.config.api_url,
                    api_key=self.config.api_key
                )
                await self.api_service.initialize()

            # Load saved configuration
            await self._load_configuration()

            logger.info("Application controller initialized")

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.error_occurred.emit(str(e))

    def show(self):
        """Show main window"""
        self.main_window.show()

    # ==================== UI Setup ====================

    def _setup_ui(self):
        """Setup UI components"""
        # Get views from main window
        self.dashboard_view = self.main_window.pages.get('dashboard')
        self.video_view = self.main_window.pages.get('video')
        self.settings_view = self.main_window.pages.get('settings')

        # Setup video view
        if self.video_view:
            self.video_view.set_vehicle_classes(self.config.vehicle_classes)
            self.video_view.set_available_models(self._get_available_models())

    def _connect_signals(self):
        """Connect UI signals"""
        # Navigation
        self.main_window.sidebar.item_clicked.connect(self._on_navigation_changed)

        # Video view signals
        if self.video_view:
            self.video_view.source_changed.connect(self._on_source_changed)
            self.video_view.model_changed.connect(self._on_model_changed)
            self.video_view.start_requested.connect(self._on_start_requested)
            self.video_view.stop_requested.connect(self._on_stop_requested)
            self.video_view.roi_defined.connect(self._on_roi_defined)
            self.video_view.line_defined.connect(self._on_line_defined)

        # Dashboard signals
        if self.dashboard_view:
            self.dashboard_view.update_requested.connect(self._update_dashboard)

        # Settings signals
        if self.settings_view:
            self.settings_view.settings_changed.connect(self._on_settings_changed)

        # Controller signals
        self.stats_updated.connect(self._on_stats_updated)

    # ==================== Navigation ====================

    @Slot(str)
    def _on_navigation_changed(self, page_id: str):
        """Handle navigation change"""
        logger.debug(f"Navigation changed to: {page_id}")

        # Update toolbar based on page
        if page_id == 'video':
            self._setup_video_toolbar()
        elif page_id == 'dashboard':
            self._setup_dashboard_toolbar()

    def _setup_video_toolbar(self):
        """Setup video page toolbar"""
        # Add any video-specific toolbar items
        pass

    def _setup_dashboard_toolbar(self):
        """Setup dashboard page toolbar"""
        # Add any dashboard-specific toolbar items
        pass

    # ==================== Video Source Management ====================

    @Slot(object)
    def _on_source_changed(self, source):
        """Handle video source change"""
        logger.info(f"Video source changed: {source}")
        self.current_source = source

        # Stop current pipeline if running
        if self.is_running:
            asyncio.create_task(self._stop_pipeline())

        # Update UI
        self.status_changed.emit("Source selected")

    @Slot(str)
    def _on_model_changed(self, model_name: str):
        """Handle model change"""
        logger.info(f"Model changed: {model_name}")

        # Reload pipeline with new model
        if self.pipeline:
            asyncio.create_task(self._reload_pipeline())

    # ==================== Pipeline Control ====================

    @Slot()
    def _on_start_requested(self):
        """Handle start request"""
        if not self.current_source:
            self.error_occurred.emit("Please select a video source first")
            return

        if not self.roi_polygon:
            self.error_occurred.emit("Please define ROI first")
            return

        if not self.counting_lines:
            self.error_occurred.emit("Please define at least one counting line")
            return

        asyncio.create_task(self._start_pipeline())

    @Slot()
    def _on_stop_requested(self):
        """Handle stop request"""
        asyncio.create_task(self._stop_pipeline())

    async def _start_pipeline(self):
        """Start processing pipeline"""
        try:
            self.status_changed.emit("Starting pipeline...")

            # Create pipeline configuration
            pipeline_config = {
                'detector': {
                    'model_path': self._get_current_model_path(),
                    'device': self.config.device,
                    'conf_threshold': self.config.confidence_threshold,
                    'nms_threshold': self.config.nms_threshold
                },
                'tracker': {
                    'type': 'bytetrack',
                    'track_thresh': 0.5,
                    'track_buffer': 30,
                    'match_thresh': 0.8
                },
                'counter': {
                    'counting_lines': [
                        {
                            'id': f"line_{i}",
                            'start': line.start_point,
                            'end': line.end_point
                        }
                        for i, line in enumerate(self.counting_lines)
                    ],
                    'enable_speed_estimation': True,
                    'fps': 30.0
                },
                'buffer_size': 3,
                'batch_size': self.config.batch_size,
                'use_gpu_decode': self.config.use_gpu
            }

            # Create pipeline
            self.pipeline = await create_pipeline(pipeline_config)

            # Add callbacks
            counter = self.pipeline.counter
            counter.add_count_callback(self._on_vehicle_counted)

            # Create video source
            if isinstance(self.current_source, int):
                # Webcam
                import cv2
                video_source = cv2.VideoCapture(self.current_source)
            else:
                # File or stream
                import cv2
                video_source = cv2.VideoCapture(self.current_source)

            # Start processing
            self.is_running = True
            self.status_changed.emit("Pipeline running")

            # Process frames
            async for frame_data in self.pipeline.start(video_source):
                if not self.is_running:
                    break

                # Update statistics
                await self._update_statistics(frame_data)

                # Display frame
                if self.video_view:
                    self.video_view.display_frame(frame_data)

            logger.info("Pipeline stopped")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.error_occurred.emit(f"Pipeline error: {str(e)}")

        finally:
            self.is_running = False
            self.status_changed.emit("Pipeline stopped")

    async def _stop_pipeline(self):
        """Stop processing pipeline"""
        self.is_running = False

        if self.pipeline:
            await self.pipeline.stop()
            self.pipeline = None

        self.status_changed.emit("Stopped")

    async def _reload_pipeline(self):
        """Reload pipeline with new configuration"""
        was_running = self.is_running

        if was_running:
            await self._stop_pipeline()

        # Small delay to ensure cleanup
        await asyncio.sleep(0.5)

        if was_running:
            await self._start_pipeline()

    # ==================== ROI and Line Management ====================

    @Slot(list)
    def _on_roi_defined(self, polygon: List[Tuple[float, float]]):
        """Handle ROI definition"""
        logger.info(f"ROI defined with {len(polygon)} points")
        self.roi_polygon = polygon

        # Apply ROI to pipeline if running
        if self.pipeline:
            # TODO: Apply ROI filter to pipeline
            pass

        self.status_changed.emit("ROI defined")

    @Slot(object)
    def _on_line_defined(self, line_data: Dict):
        """Handle counting line definition"""
        logger.info("Counting line defined")

        # Create counting line
        line = CountingLine(
            line_id=f"line_{len(self.counting_lines)}",
            start_point=tuple(line_data['start']),
            end_point=tuple(line_data['end']),
            direction_vector=tuple(line_data.get('direction', [])) or None
        )

        self.counting_lines.append(line)

        # Add to counter if pipeline is running
        if self.pipeline and self.pipeline.counter:
            self.pipeline.counter.add_line(line)

        self.status_changed.emit(f"{len(self.counting_lines)} lines defined")

    # ==================== Statistics and Updates ====================

    async def _update_statistics(self, frame_data):
        """Update statistics from frame data"""
        # Get pipeline stats
        pipeline_stats = self.pipeline.get_stats()

        # Update FPS
        self.stats['fps'] = pipeline_stats.avg_fps

        # Update active tracks
        if frame_data.tracks:
            self.stats['active_tracks'] = len(frame_data.tracks)

        # Update detections for heatmap
        detections = []
        if frame_data.detections is not None:
            for det in frame_data.detections:
                detections.append({
                    'x': det.center[0],
                    'y': det.center[1],
                    'confidence': det.confidence
                })
        self.stats['detections'] = detections

        # Get counter statistics
        if self.pipeline.counter:
            counter_stats = self.pipeline.counter.get_statistics()
            self.stats['total_counts'] = counter_stats['total_counts']
            self.stats['count_rate'] = counter_stats['recent_count_rate']
            self.stats['avg_speed'] = counter_stats['avg_speed']
            self.stats['max_speed'] = counter_stats['max_speed']

        # Emit update
        self.stats_updated.emit(self.stats)

    @Slot(dict)
    def _on_stats_updated(self, stats: Dict):
        """Handle statistics update"""
        # Update dashboard
        if self.dashboard_view:
            self.dashboard_view.update_stats(stats)

        # Update sidebar quick stats
        self.main_window.sidebar.stats_widget.update_stats(
            fps=stats.get('fps', 0),
            total=sum(stats.get('total_counts', {}).values()),
            active=stats.get('active_tracks', 0)
        )

    def _on_vehicle_counted(self, count: VehicleCount):
        """Handle vehicle count event"""
        logger.info(f"Vehicle counted: {count.vehicle_class} "
                    f"on {count.line_id} going {count.direction.name}")

        # Create database record
        record = CountRecord(
            id=None,
            host_id=self.config.host_id,
            camera_id=self.config.camera_id,
            line_id=count.line_id,
            timestamp=count.timestamp,
            vehicle_type=self._get_vehicle_type_name(count.vehicle_class),
            count=1,
            direction=count.direction.name,
            metadata={
                'track_id': count.track_id,
                'confidence': count.confidence,
                'speed_kmh': count.speed
            }
        )

        # Buffer for batch insertion
        if self.db_service:
            asyncio.create_task(self.db_service.buffer_count(record))

        # Send to API if configured
        if self.api_service:
            asyncio.create_task(self._send_count_to_api(record))

    # ==================== Dashboard Updates ====================

    @Slot()
    def _update_dashboard(self):
        """Update dashboard with latest data"""
        if not self.db_service:
            return

        asyncio.create_task(self._fetch_dashboard_data())

    async def _fetch_dashboard_data(self):
        """Fetch data for dashboard"""
        try:
            # Get last hour statistics
            from datetime import datetime, timedelta
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)

            stats = await self.db_service.get_statistics(start_time, end_time)

            # Add current live stats
            stats.update(self.stats)

            # Update dashboard
            if self.dashboard_view:
                self.dashboard_view.update_stats(stats)

        except Exception as e:
            logger.error(f"Dashboard update error: {e}")

    # ==================== Settings Management ====================

    @Slot(dict)
    def _on_settings_changed(self, settings: Dict):
        """Handle settings change"""
        logger.info("Settings changed")

        # Update configuration
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Save configuration
        asyncio.create_task(self._save_configuration())

        # Reload components if needed
        if self.pipeline and any(key in settings for key in
                                 ['confidence_threshold', 'device', 'batch_size']):
            asyncio.create_task(self._reload_pipeline())

    async def _save_configuration(self):
        """Save current configuration"""
        try:
            config_data = {
                'source': self.current_source,
                'roi_polygon': self.roi_polygon,
                'counting_lines': [
                    {
                        'start': line.start_point,
                        'end': line.end_point,
                        'direction': line.direction_vector
                    }
                    for line in self.counting_lines
                ],
                'settings': self.config.dict()
            }

            config_path = self.config.data_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info("Configuration saved")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    async def _load_configuration(self):
        """Load saved configuration"""
        try:
            config_path = self.config.data_dir / 'config.json'
            if not config_path.exists():
                return

            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Apply loaded configuration
            if 'source' in config_data:
                self.current_source = config_data['source']

            if 'roi_polygon' in config_data:
                self.roi_polygon = config_data['roi_polygon']

            if 'counting_lines' in config_data:
                self.counting_lines = []
                for line_data in config_data['counting_lines']:
                    line = CountingLine(
                        line_id=f"line_{len(self.counting_lines)}",
                        start_point=tuple(line_data['start']),
                        end_point=tuple(line_data['end']),
                        direction_vector=tuple(line_data.get('direction', [])) or None
                    )
                    self.counting_lines.append(line)

            logger.info("Configuration loaded")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    # ==================== API Integration ====================

    async def _send_count_to_api(self, record: CountRecord):
        """Send count record to API"""
        if not self.api_service:
            return

        try:
            # Prepare payload
            payload = {
                'host_id': record.host_id,
                'camera_id': record.camera_id,
                'line_id': record.line_id,
                'timestamp': record.timestamp.isoformat(),
                'vehicle_type': record.vehicle_type,
                'count': record.count,
                'direction': record.direction,
                'metadata': record.metadata
            }

            # Send to API
            await self.api_service.send_count(payload)

        except Exception as e:
            logger.error(f"API send error: {e}")

    # ==================== Export Functions ====================

    async def export_data(self, start_time: datetime, end_time: datetime,
                          format: str = 'excel'):
        """Export data for time range"""
        try:
            # Get data from database
            df = await self.db_service.get_counts(start_time, end_time)

            if df.empty:
                self.error_occurred.emit("No data to export")
                return

            # Get export path
            default_name = f"traffic_data_{start_time:%Y%m%d}_{end_time:%Y%m%d}"

            if format == 'excel':
                file_path, _ = QFileDialog.getSaveFileName(
                    self.main_window,
                    "Export Data",
                    f"{default_name}.xlsx",
                    "Excel Files (*.xlsx)"
                )

                if file_path:
                    # Export to Excel with charts
                    from src.services.export_service import ExportService
                    export_service = ExportService()
                    await export_service.export_to_excel(
                        df, Path(file_path), include_charts=True
                    )

            elif format == 'csv':
                file_path, _ = QFileDialog.getSaveFileName(
                    self.main_window,
                    "Export Data",
                    f"{default_name}.csv",
                    "CSV Files (*.csv)"
                )

                if file_path:
                    df.to_csv(file_path, index=False)

            self.status_changed.emit(f"Data exported to {file_path}")

        except Exception as e:
            logger.error(f"Export error: {e}")
            self.error_occurred.emit(f"Export failed: {str(e)}")

    # ==================== Utility Functions ====================

    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        model_files = []

        # Check for model files
        for ext in ['.xml', '.onnx', '.engine']:
            model_files.extend(self.config.model_dir.glob(f'*{ext}'))

        return [f.stem for f in model_files]

    def _get_current_model_path(self) -> str:
        """Get current model path"""
        if self.video_view:
            model_name = self.video_view.get_selected_model()

            # Try different extensions
            for ext in ['.xml', '.onnx', '.engine']:
                model_path = self.config.model_dir / f"{model_name}{ext}"
                if model_path.exists():
                    return str(model_path)

        # Default model
        return str(self.config.weights_path)

    def _get_vehicle_type_name(self, class_id: int) -> str:
        """Get vehicle type name from class ID"""
        # Map class IDs to vehicle types
        # This mapping depends on your model
        class_to_type = {
            2: 'car',
            7: 'truck',
            5: 'bus',
            3: 'motorbike',
            1: 'bicycle'
        }

        return class_to_type.get(class_id, f'class_{class_id}')

    # ==================== Cleanup ====================

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up application controller")

        # Stop pipeline
        if self.pipeline:
            await self.pipeline.stop()

        # Close services
        if self.db_service:
            await self.db_service.close()

        if self.api_service:
            await self.api_service.close()

        logger.info("Cleanup completed")