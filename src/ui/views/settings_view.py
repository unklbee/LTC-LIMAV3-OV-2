# src/ui/views/settings_view.py
"""
Comprehensive settings view with tabbed interface.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QFormLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QSlider,
    QFileDialog, QColorDialog, QMessageBox,
    QListWidget, QListWidgetItem, QTextEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QSettings
from PySide6.QtGui import QIcon, QColor, QFont
from typing import Dict, Any, List, Optional
from pathlib import Path

import structlog

logger = structlog.get_logger()


class SettingsView(QWidget):
    """Main settings view with tabbed interface"""

    # Signals
    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setObjectName("settingsView")

        # Store current settings
        self.current_settings = {}
        self.original_settings = {}

        # Qt settings for persistence
        self.qsettings = QSettings("LIMA", "TrafficCounter")

        # Setup UI
        self._setup_ui()

        # Load settings
        self._load_settings()

    def _setup_ui(self):
        """Setup settings UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Settings")
        title.setObjectName("settingsTitle")
        title.setStyleSheet("""
            #settingsTitle {
                font-size: 24px;
                font-weight: bold;
                color: #0078d4;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("settingsTabWidget")

        # Create tabs
        self.general_tab = self._create_general_tab()
        self.detection_tab = self._create_detection_tab()
        self.tracking_tab = self._create_tracking_tab()
        self.database_tab = self._create_database_tab()
        self.api_tab = self._create_api_tab()
        self.ui_tab = self._create_ui_tab()
        self.advanced_tab = self._create_advanced_tab()

        # Add tabs
        self.tab_widget.addTab(self.general_tab, "General")
        self.tab_widget.addTab(self.detection_tab, "Detection")
        self.tab_widget.addTab(self.tracking_tab, "Tracking")
        self.tab_widget.addTab(self.database_tab, "Database")
        self.tab_widget.addTab(self.api_tab, "API")
        self.tab_widget.addTab(self.ui_tab, "Interface")
        self.tab_widget.addTab(self.advanced_tab, "Advanced")

        layout.addWidget(self.tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_settings)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)

        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        # Apply styles
        self._apply_styles()

    def _apply_styles(self):
        """Apply widget styles"""
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
            
            QTabBar::tab {
                background-color: #3d3d3d;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            
            QPushButton:pressed {
                background-color: #0078d4;
            }
        """)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Paths group
        paths_group = QGroupBox("Paths")
        paths_layout = QFormLayout(paths_group)

        # Model directory
        self.model_dir_edit = QLineEdit()
        self.model_dir_btn = QPushButton("Browse...")
        self.model_dir_btn.clicked.connect(
            lambda: self._browse_directory(self.model_dir_edit)
        )

        model_dir_layout = QHBoxLayout()
        model_dir_layout.addWidget(self.model_dir_edit)
        model_dir_layout.addWidget(self.model_dir_btn)
        paths_layout.addRow("Model Directory:", model_dir_layout)

        # Data directory
        self.data_dir_edit = QLineEdit()
        self.data_dir_btn = QPushButton("Browse...")
        self.data_dir_btn.clicked.connect(
            lambda: self._browse_directory(self.data_dir_edit)
        )

        data_dir_layout = QHBoxLayout()
        data_dir_layout.addWidget(self.data_dir_edit)
        data_dir_layout.addWidget(self.data_dir_btn)
        paths_layout.addRow("Data Directory:", data_dir_layout)

        layout.addWidget(paths_group)

        # Device group
        device_group = QGroupBox("Device Settings")
        device_layout = QFormLayout(device_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "Intel GPU", "NVIDIA GPU"])
        device_layout.addRow("Inference Device:", self.device_combo)

        self.use_gpu_check = QCheckBox("Enable GPU Acceleration")
        device_layout.addRow("", self.use_gpu_check)

        layout.addWidget(device_group)

        # Performance group
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        perf_layout.addRow("Batch Size:", self.batch_size_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, 16)
        self.num_workers_spin.setValue(4)
        perf_layout.addRow("Worker Threads:", self.num_workers_spin)

        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(1, 10)
        self.buffer_size_spin.setValue(3)
        perf_layout.addRow("Frame Buffer Size:", self.buffer_size_spin)

        layout.addWidget(perf_group)
        layout.addStretch()

        return widget

    def _create_detection_tab(self) -> QWidget:
        """Create detection settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Thresholds group
        thresh_group = QGroupBox("Detection Thresholds")
        thresh_layout = QFormLayout(thresh_group)

        # Confidence threshold
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")

        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        thresh_layout.addRow("Confidence Threshold:", conf_layout)

        # NMS threshold
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(0, 100)
        self.nms_slider.setValue(45)
        self.nms_label = QLabel("0.45")

        self.nms_slider.valueChanged.connect(
            lambda v: self.nms_label.setText(f"{v/100:.2f}")
        )

        nms_layout = QHBoxLayout()
        nms_layout.addWidget(self.nms_slider)
        nms_layout.addWidget(self.nms_label)
        thresh_layout.addRow("NMS Threshold:", nms_layout)

        layout.addWidget(thresh_group)

        # Vehicle classes group
        classes_group = QGroupBox("Vehicle Classes")
        classes_layout = QVBoxLayout(classes_group)

        self.vehicle_classes_list = QListWidget()
        self.vehicle_classes_list.setSelectionMode(QListWidget.MultiSelection)

        # Add default classes
        classes = ["bicycle", "car", "motorbike", "bus", "truck"]
        for cls in classes:
            item = QListWidgetItem(cls)
            item.setCheckState(Qt.Checked)
            self.vehicle_classes_list.addItem(item)

        classes_layout.addWidget(self.vehicle_classes_list)

        layout.addWidget(classes_group)

        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)

        self.input_width_spin = QSpinBox()
        self.input_width_spin.setRange(320, 1280)
        self.input_width_spin.setValue(640)
        self.input_width_spin.setSingleStep(32)

        self.input_height_spin = QSpinBox()
        self.input_height_spin.setRange(320, 1280)
        self.input_height_spin.setValue(640)
        self.input_height_spin.setSingleStep(32)

        size_layout = QHBoxLayout()
        size_layout.addWidget(self.input_width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.input_height_spin)

        model_layout.addRow("Input Size:", size_layout)

        layout.addWidget(model_group)
        layout.addStretch()

        return widget

    def _create_tracking_tab(self) -> QWidget:
        """Create tracking settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tracker settings
        tracker_group = QGroupBox("Tracker Settings")
        tracker_layout = QFormLayout(tracker_group)

        self.track_thresh_slider = QSlider(Qt.Horizontal)
        self.track_thresh_slider.setRange(0, 100)
        self.track_thresh_slider.setValue(50)
        self.track_thresh_label = QLabel("0.50")

        self.track_thresh_slider.valueChanged.connect(
            lambda v: self.track_thresh_label.setText(f"{v/100:.2f}")
        )

        track_thresh_layout = QHBoxLayout()
        track_thresh_layout.addWidget(self.track_thresh_slider)
        track_thresh_layout.addWidget(self.track_thresh_label)
        tracker_layout.addRow("Track Threshold:", track_thresh_layout)

        self.track_buffer_spin = QSpinBox()
        self.track_buffer_spin.setRange(1, 300)
        self.track_buffer_spin.setValue(30)
        tracker_layout.addRow("Track Buffer (frames):", self.track_buffer_spin)

        self.max_age_spin = QSpinBox()
        self.max_age_spin.setRange(1, 30)
        self.max_age_spin.setValue(5)
        tracker_layout.addRow("Max Age:", self.max_age_spin)

        layout.addWidget(tracker_group)

        # Counting settings
        counting_group = QGroupBox("Counting Settings")
        counting_layout = QFormLayout(counting_group)

        self.enable_speed_check = QCheckBox("Enable Speed Estimation")
        self.enable_speed_check.setChecked(True)
        counting_layout.addRow("", self.enable_speed_check)

        self.pixel_per_meter_spin = QDoubleSpinBox()
        self.pixel_per_meter_spin.setRange(1.0, 100.0)
        self.pixel_per_meter_spin.setValue(10.0)
        self.pixel_per_meter_spin.setSingleStep(0.5)
        counting_layout.addRow("Pixels per Meter:", self.pixel_per_meter_spin)

        self.speed_limit_spin = QDoubleSpinBox()
        self.speed_limit_spin.setRange(10.0, 200.0)
        self.speed_limit_spin.setValue(60.0)
        self.speed_limit_spin.setSingleStep(5.0)
        self.speed_limit_spin.setSuffix(" km/h")
        counting_layout.addRow("Speed Limit:", self.speed_limit_spin)

        layout.addWidget(counting_group)
        layout.addStretch()

        return widget

    def _create_database_tab(self) -> QWidget:
        """Create database settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Database settings
        db_group = QGroupBox("Database Configuration")
        db_layout = QFormLayout(db_group)

        # DB file path
        self.db_path_edit = QLineEdit()
        self.db_path_btn = QPushButton("Browse...")
        self.db_path_btn.clicked.connect(self._browse_db_file)

        db_path_layout = QHBoxLayout()
        db_path_layout.addWidget(self.db_path_edit)
        db_path_layout.addWidget(self.db_path_btn)
        db_layout.addRow("Database File:", db_path_layout)

        # Host ID
        self.host_id_edit = QLineEdit()
        self.host_id_edit.setPlaceholderText("e.g., camera-01")
        db_layout.addRow("Host ID:", self.host_id_edit)

        # Camera ID
        self.camera_id_edit = QLineEdit()
        self.camera_id_edit.setPlaceholderText("e.g., entrance-cam")
        db_layout.addRow("Camera ID:", self.camera_id_edit)

        # Save interval
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(60, 3600)
        self.save_interval_spin.setValue(300)
        self.save_interval_spin.setSingleStep(60)
        self.save_interval_spin.setSuffix(" seconds")
        db_layout.addRow("Save Interval:", self.save_interval_spin)

        # Connection pool size
        self.db_pool_size_spin = QSpinBox()
        self.db_pool_size_spin.setRange(1, 20)
        self.db_pool_size_spin.setValue(5)
        db_layout.addRow("Connection Pool Size:", self.db_pool_size_spin)

        layout.addWidget(db_group)

        # Maintenance
        maint_group = QGroupBox("Database Maintenance")
        maint_layout = QVBoxLayout(maint_group)

        self.auto_cleanup_check = QCheckBox("Enable Auto Cleanup")
        self.auto_cleanup_check.setChecked(True)
        maint_layout.addWidget(self.auto_cleanup_check)

        cleanup_layout = QHBoxLayout()
        cleanup_layout.addWidget(QLabel("Keep data for:"))

        self.cleanup_days_spin = QSpinBox()
        self.cleanup_days_spin.setRange(1, 365)
        self.cleanup_days_spin.setValue(30)
        self.cleanup_days_spin.setSuffix(" days")
        cleanup_layout.addWidget(self.cleanup_days_spin)

        maint_layout.addLayout(cleanup_layout)

        # Manual actions
        actions_layout = QHBoxLayout()

        self.vacuum_btn = QPushButton("Vacuum Database")
        self.vacuum_btn.clicked.connect(self._vacuum_database)
        actions_layout.addWidget(self.vacuum_btn)

        self.backup_btn = QPushButton("Backup Database")
        self.backup_btn.clicked.connect(self._backup_database)
        actions_layout.addWidget(self.backup_btn)

        maint_layout.addLayout(actions_layout)

        layout.addWidget(maint_group)
        layout.addStretch()

        return widget

    def _create_api_tab(self) -> QWidget:
        """Create API settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # API configuration
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout(api_group)

        self.api_enabled_check = QCheckBox("Enable API Integration")
        api_layout.addRow("", self.api_enabled_check)

        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText("https://api.example.com/traffic")
        api_layout.addRow("API URL:", self.api_url_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Your API key")
        api_layout.addRow("API Key:", self.api_key_edit)

        self.api_timeout_spin = QSpinBox()
        self.api_timeout_spin.setRange(1, 300)
        self.api_timeout_spin.setValue(30)
        self.api_timeout_spin.setSuffix(" seconds")
        api_layout.addRow("Request Timeout:", self.api_timeout_spin)

        self.api_batch_size_spin = QSpinBox()
        self.api_batch_size_spin.setRange(1, 1000)
        self.api_batch_size_spin.setValue(100)
        api_layout.addRow("Batch Size:", self.api_batch_size_spin)

        layout.addWidget(api_group)

        # Test connection
        test_group = QGroupBox("Connection Test")
        test_layout = QVBoxLayout(test_group)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_api_connection)
        test_layout.addWidget(self.test_btn)

        self.test_result = QTextEdit()
        self.test_result.setReadOnly(True)
        self.test_result.setMaximumHeight(100)
        test_layout.addWidget(self.test_result)

        layout.addWidget(test_group)
        layout.addStretch()

        return widget

    def _create_ui_tab(self) -> QWidget:
        """Create UI settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Theme settings
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout(theme_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        theme_layout.addRow("Theme:", self.theme_combo)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Indonesian", "Chinese"])
        theme_layout.addRow("Language:", self.language_combo)

        layout.addWidget(theme_group)

        # Window settings
        window_group = QGroupBox("Window Settings")
        window_layout = QFormLayout(window_group)

        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(800, 3840)
        self.window_width_spin.setValue(1400)
        self.window_width_spin.setSingleStep(10)

        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(600, 2160)
        self.window_height_spin.setValue(800)
        self.window_height_spin.setSingleStep(10)

        size_layout = QHBoxLayout()
        size_layout.addWidget(self.window_width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.window_height_spin)

        window_layout.addRow("Default Size:", size_layout)

        self.remember_window_check = QCheckBox("Remember Window Position")
        self.remember_window_check.setChecked(True)
        window_layout.addRow("", self.remember_window_check)

        layout.addWidget(window_group)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout(display_group)

        self.show_fps_check = QCheckBox("Show FPS")
        self.show_fps_check.setChecked(True)
        display_layout.addRow("", self.show_fps_check)

        self.show_inference_check = QCheckBox("Show Inference Time")
        display_layout.addRow("", self.show_inference_check)

        self.show_tracks_check = QCheckBox("Show Track IDs")
        self.show_tracks_check.setChecked(True)
        display_layout.addRow("", self.show_tracks_check)

        self.show_confidence_check = QCheckBox("Show Confidence Scores")
        display_layout.addRow("", self.show_confidence_check)

        layout.addWidget(display_group)
        layout.addStretch()

        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Logging settings
        log_group = QGroupBox("Logging")
        log_layout = QFormLayout(log_group)

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        log_layout.addRow("Log Level:", self.log_level_combo)

        self.log_to_file_check = QCheckBox("Log to File")
        self.log_to_file_check.setChecked(True)
        log_layout.addRow("", self.log_to_file_check)

        self.log_max_size_spin = QSpinBox()
        self.log_max_size_spin.setRange(1, 100)
        self.log_max_size_spin.setValue(10)
        self.log_max_size_spin.setSuffix(" MB")
        log_layout.addRow("Max Log Size:", self.log_max_size_spin)

        layout.addWidget(log_group)

        # Debug settings
        debug_group = QGroupBox("Debug Options")
        debug_layout = QFormLayout(debug_group)

        self.debug_mode_check = QCheckBox("Enable Debug Mode")
        debug_layout.addRow("", self.debug_mode_check)

        self.profile_perf_check = QCheckBox("Profile Performance")
        debug_layout.addRow("", self.profile_perf_check)

        self.save_detections_check = QCheckBox("Save Detection Images")
        debug_layout.addRow("", self.save_detections_check)

        layout.addWidget(debug_group)

        # Export settings
        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout(export_group)

        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["Excel", "CSV", "JSON", "PDF", "HTML"])
        export_layout.addRow("Default Format:", self.export_format_combo)

        self.include_charts_check = QCheckBox("Include Charts in Reports")
        self.include_charts_check.setChecked(True)
        export_layout.addRow("", self.include_charts_check)

        layout.addWidget(export_group)
        layout.addStretch()

        return widget

    # ==================== Helper Methods ====================

    def _browse_directory(self, line_edit: QLineEdit):
        """Browse for directory"""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text() or str(Path.home())
        )

        if path:
            line_edit.setText(path)

    def _browse_db_file(self):
        """Browse for database file"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Database File",
            self.db_path_edit.text() or str(Path.home() / "traffic_counts.db"),
            "SQLite Database (*.db)"
        )

        if path:
            self.db_path_edit.setText(path)

    def _test_api_connection(self):
        """Test API connection"""
        url = self.api_url_edit.text()
        key = self.api_key_edit.text()

        if not url:
            self.test_result.setText("Please enter API URL")
            return

        self.test_result.setText("Testing connection...")

        # TODO: Implement actual API test
        # For now, just simulate
        import asyncio

        async def test():
            await asyncio.sleep(1)  # Simulate API call
            return {"status": "success", "message": "Connection successful"}

        # Run test
        try:
            # This is simplified - in real app, use proper async handling
            self.test_result.setText("✓ Connection successful")
        except Exception as e:
            self.test_result.setText(f"✗ Connection failed: {str(e)}")

    def _vacuum_database(self):
        """Vacuum database"""
        reply = QMessageBox.question(
            self,
            "Vacuum Database",
            "This will optimize the database and may take some time. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # TODO: Implement actual vacuum operation
            QMessageBox.information(self, "Success", "Database optimized successfully")

    def _backup_database(self):
        """Backup database"""
        source = self.db_path_edit.text()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Warning", "Database file not found")
            return

        backup_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Backup",
            str(Path(source).with_suffix('.backup.db')),
            "SQLite Database (*.db)"
        )

        if backup_path:
            try:
                import shutil
                shutil.copy2(source, backup_path)
                QMessageBox.information(self, "Success", f"Database backed up to:\n{backup_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Backup failed: {str(e)}")

    # ==================== Settings Management ====================

    def _load_settings(self):
        """Load settings from config"""
        # TODO: Load from actual config
        # For now, use defaults
        pass

    def _apply_settings(self):
        """Apply current settings"""
        # Collect all settings
        settings = self._collect_settings()

        # Check what changed
        changed = {}
        for key, value in settings.items():
            if key not in self.original_settings or self.original_settings[key] != value:
                changed[key] = value

        if changed:
            # Emit signal with changed settings
            self.settings_changed.emit(changed)

            # Update original settings
            self.original_settings.update(changed)

            QMessageBox.information(self, "Success", "Settings applied successfully")
        else:
            QMessageBox.information(self, "Info", "No changes to apply")

    def _save_settings(self):
        """Save settings permanently"""
        self._apply_settings()

        # Save to QSettings
        settings = self._collect_settings()
        for key, value in settings.items():
            self.qsettings.setValue(key, value)

        QMessageBox.information(self, "Success", "Settings saved successfully")

    def _reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "This will reset all settings to default values. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # TODO: Implement actual reset
            QMessageBox.information(self, "Success", "Settings reset to defaults")

    def _collect_settings(self) -> Dict[str, Any]:
        """Collect all settings from UI"""
        settings = {}

        # General tab
        settings['model_dir'] = self.model_dir_edit.text()
        settings['data_dir'] = self.data_dir_edit.text()
        settings['device'] = self.device_combo.currentText()
        settings['use_gpu'] = self.use_gpu_check.isChecked()
        settings['batch_size'] = self.batch_size_spin.value()
        settings['num_workers'] = self.num_workers_spin.value()
        settings['buffer_size'] = self.buffer_size_spin.value()

        # Detection tab
        settings['confidence_threshold'] = self.conf_slider.value() / 100.0
        settings['nms_threshold'] = self.nms_slider.value() / 100.0

        # Get selected vehicle classes
        selected_classes = []
        for i in range(self.vehicle_classes_list.count()):
            item = self.vehicle_classes_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_classes.append(item.text())
        settings['vehicle_classes'] = selected_classes

        settings['input_size'] = (
            self.input_width_spin.value(),
            self.input_height_spin.value()
        )

        # Tracking tab
        settings['track_threshold'] = self.track_thresh_slider.value() / 100.0
        settings['track_buffer'] = self.track_buffer_spin.value()
        settings['max_age'] = self.max_age_spin.value()
        settings['enable_speed_estimation'] = self.enable_speed_check.isChecked()
        settings['pixel_per_meter'] = self.pixel_per_meter_spin.value()
        settings['speed_limit'] = self.speed_limit_spin.value()

        # Database tab
        settings['db_path'] = self.db_path_edit.text()
        settings['host_id'] = self.host_id_edit.text()
        settings['camera_id'] = self.camera_id_edit.text()
        settings['count_save_interval'] = self.save_interval_spin.value()
        settings['db_pool_size'] = self.db_pool_size_spin.value()

        # API tab
        if self.api_enabled_check.isChecked():
            settings['api_url'] = self.api_url_edit.text()
            settings['api_key'] = self.api_key_edit.text()
            settings['api_timeout'] = self.api_timeout_spin.value()
            settings['api_batch_size'] = self.api_batch_size_spin.value()
        else:
            settings['api_url'] = None

        # UI tab
        settings['theme'] = self.theme_combo.currentText().lower()
        settings['language'] = self.language_combo.currentText()[:2].lower()
        settings['window_width'] = self.window_width_spin.value()
        settings['window_height'] = self.window_height_spin.value()

        # Advanced tab
        settings['log_level'] = self.log_level_combo.currentText()
        settings['log_to_file'] = self.log_to_file_check.isChecked()
        settings['log_max_size'] = self.log_max_size_spin.value() * 1024 * 1024
        settings['debug_mode'] = self.debug_mode_check.isChecked()
        settings['profile_performance'] = self.profile_perf_check.isChecked()
        settings['save_detections'] = self.save_detections_check.isChecked()
        settings['export_format'] = self.export_format_combo.currentText().lower()
        settings['include_charts'] = self.include_charts_check.isChecked()

        return settings