# src/ui/views/detection_view.py
"""
Detection view for configuring and monitoring object detection.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSlider, QCheckBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSplitter, QTextEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor
from typing import Dict, List, Optional
import pyqtgraph as pg
from datetime import datetime
from collections import deque

import structlog

logger = structlog.get_logger()


class DetectionStatsWidget(QFrame):
    """Widget for displaying detection statistics"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Box)
        self.setObjectName("detectionStats")

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Detection Statistics")
        title.setObjectName("statsTitle")
        title.setStyleSheet("""
            #statsTitle {
                font-size: 16px;
                font-weight: bold;
                color: #0078d4;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)

        # Stats grid
        stats_layout = QHBoxLayout()

        # Total detections
        self.total_label = self._create_stat_widget("Total Detections", "0")
        stats_layout.addWidget(self.total_label)

        # Detection rate
        self.rate_label = self._create_stat_widget("Detection Rate", "0/s")
        stats_layout.addWidget(self.rate_label)

        # Average confidence
        self.conf_label = self._create_stat_widget("Avg Confidence", "0%")
        stats_layout.addWidget(self.conf_label)

        # Processing time
        self.time_label = self._create_stat_widget("Processing Time", "0ms")
        stats_layout.addWidget(self.time_label)

        layout.addLayout(stats_layout)

    def _create_stat_widget(self, label: str, value: str) -> QWidget:
        """Create individual stat widget"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 5px;
                padding: 10px;
            }
        """)

        layout = QVBoxLayout(widget)

        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #b0b0b0; font-size: 12px;")

        value_widget = QLabel(value)
        value_widget.setObjectName(f"{label.replace(' ', '')}Value")
        value_widget.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        value_widget.setAlignment(Qt.AlignCenter)

        layout.addWidget(label_widget)
        layout.addWidget(value_widget)

        return widget

    def update_stats(self, stats: Dict):
        """Update statistics display"""
        if 'total_detections' in stats:
            self.total_label.findChild(QLabel, "TotalDetectionsValue").setText(
                str(stats['total_detections'])
            )

        if 'detection_rate' in stats:
            self.rate_label.findChild(QLabel, "DetectionRateValue").setText(
                f"{stats['detection_rate']:.1f}/s"
            )

        if 'avg_confidence' in stats:
            self.conf_label.findChild(QLabel, "AvgConfidenceValue").setText(
                f"{stats['avg_confidence']:.0f}%"
            )

        if 'processing_time' in stats:
            self.time_label.findChild(QLabel, "ProcessingTimeValue").setText(
                f"{stats['processing_time']:.1f}ms"
            )


class ClassFilterWidget(QGroupBox):
    """Widget for filtering detection classes"""

    classes_changed = Signal(list)

    def __init__(self):
        super().__init__("Class Filter")

        layout = QVBoxLayout(self)

        # Select all/none buttons
        button_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self._select_none)
        button_layout.addWidget(self.select_none_btn)

        layout.addLayout(button_layout)

        # Class checkboxes
        self.class_checks = {}
        classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

        for cls in classes:
            check = QCheckBox(cls.capitalize())
            check.setChecked(cls in ["bicycle", "car", "motorbike", "bus", "truck"])
            check.stateChanged.connect(self._on_class_changed)
            self.class_checks[cls] = check
            layout.addWidget(check)

    def _select_all(self):
        """Select all classes"""
        for check in self.class_checks.values():
            check.setChecked(True)

    def _select_none(self):
        """Deselect all classes"""
        for check in self.class_checks.values():
            check.setChecked(False)

    def _on_class_changed(self):
        """Handle class selection change"""
        selected = [
            cls for cls, check in self.class_checks.items()
            if check.isChecked()
        ]
        self.classes_changed.emit(selected)

    def get_selected_classes(self) -> List[str]:
        """Get list of selected classes"""
        return [
            cls for cls, check in self.class_checks.items()
            if check.isChecked()
        ]


class DetectionLogWidget(QTableWidget):
    """Widget for displaying detection log"""

    def __init__(self):
        super().__init__()

        # Set columns
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels([
            "Time", "Class", "Confidence", "Track ID", "Location", "Speed"
        ])

        # Configure table
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)

        # Style
        self.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                alternate-background-color: #3d3d3d;
                color: white;
                gridline-color: #4d4d4d;
            }
            QHeaderView::section {
                background-color: #0078d4;
                color: white;
                padding: 5px;
                border: none;
            }
        """)

        # Limit rows
        self.max_rows = 100

    def add_detection(self, detection: Dict):
        """Add detection to log"""
        # Remove old rows if exceeding limit
        if self.rowCount() >= self.max_rows:
            self.removeRow(0)

        # Insert new row
        row = self.rowCount()
        self.insertRow(row)

        # Time
        time_item = QTableWidgetItem(
            datetime.now().strftime("%H:%M:%S.%f")[:-3]
        )
        self.setItem(row, 0, time_item)

        # Class
        class_item = QTableWidgetItem(detection.get('class', 'Unknown'))
        self.setItem(row, 1, class_item)

        # Confidence
        conf_item = QTableWidgetItem(f"{detection.get('confidence', 0):.2f}")
        self.setItem(row, 2, conf_item)

        # Track ID
        track_item = QTableWidgetItem(str(detection.get('track_id', '-')))
        self.setItem(row, 3, track_item)

        # Location
        bbox = detection.get('bbox', [0, 0, 0, 0])
        loc_item = QTableWidgetItem(
            f"({int(bbox[0])}, {int(bbox[1])}) - ({int(bbox[2])}, {int(bbox[3])})"
        )
        self.setItem(row, 4, loc_item)

        # Speed
        speed = detection.get('speed')
        speed_item = QTableWidgetItem(
            f"{speed:.1f} km/h" if speed else "-"
        )
        self.setItem(row, 5, speed_item)

        # Scroll to bottom
        self.scrollToBottom()


class ConfidenceGraphWidget(pg.PlotWidget):
    """Real-time confidence distribution graph"""

    def __init__(self):
        super().__init__()

        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('left', 'Count')
        self.setLabel('bottom', 'Confidence')
        self.setTitle('Confidence Distribution')

        # Data
        self.confidence_data = deque(maxlen=1000)

        # Histogram
        self.hist_item = pg.BarGraphItem(x=[], height=[], width=0.02, brush='b')
        self.addItem(self.hist_item)

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_histogram)
        self.timer.start(1000)  # Update every second

    def add_confidence(self, confidence: float):
        """Add confidence value"""
        self.confidence_data.append(confidence)

    def update_histogram(self):
        """Update histogram display"""
        if not self.confidence_data:
            return

        # Create histogram
        hist, bins = np.histogram(list(self.confidence_data), bins=20, range=(0, 1))

        # Update bar graph
        self.hist_item.setOpts(
            x=bins[:-1],
            height=hist,
            width=(bins[1] - bins[0]) * 0.8
        )


class DetectionView(QWidget):
    """Main detection configuration and monitoring view"""

    # Signals
    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setObjectName("detectionView")

        # Setup UI
        self._setup_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)  # Update at 10 Hz

        # Detection buffer
        self.detection_buffer = deque(maxlen=1000)

    def _setup_ui(self):
        """Setup detection view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Detection Configuration")
        title.setObjectName("viewTitle")
        title.setStyleSheet("""
            #viewTitle {
                font-size: 24px;
                font-weight: bold;
                color: #0078d4;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Configuration
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)

        # Right panel - Monitoring
        right_panel = self._create_monitoring_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])
        layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        """Create configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout(model_group)

        # Model selection
        model_label = QLabel("Detection Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "YOLOv7-tiny", "YOLOv7", "YOLOv8n", "YOLOv8s", "YOLOv8m"
        ])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

        # Backend selection
        backend_label = QLabel("Inference Backend:")
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "OpenVINO", "ONNX Runtime", "TensorRT"
        ])
        model_layout.addWidget(backend_label)
        model_layout.addWidget(self.backend_combo)

        layout.addWidget(model_group)

        # Detection settings
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QVBoxLayout(detect_group)

        # Confidence threshold
        conf_label = QLabel("Confidence Threshold:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_value = QLabel("0.25")

        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value.setText(f"{v/100:.2f}")
        )

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)

        detect_layout.addWidget(conf_label)
        detect_layout.addLayout(conf_layout)

        # NMS threshold
        nms_label = QLabel("NMS Threshold:")
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(0, 100)
        self.nms_slider.setValue(45)
        self.nms_value = QLabel("0.45")

        self.nms_slider.valueChanged.connect(
            lambda v: self.nms_value.setText(f"{v/100:.2f}")
        )

        nms_layout = QHBoxLayout()
        nms_layout.addWidget(self.nms_slider)
        nms_layout.addWidget(self.nms_value)

        detect_layout.addWidget(nms_label)
        detect_layout.addLayout(nms_layout)

        layout.addWidget(detect_group)

        # Class filter
        self.class_filter = ClassFilterWidget()
        self.class_filter.classes_changed.connect(self._on_classes_changed)
        layout.addWidget(self.class_filter)

        # Apply button
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self._apply_settings)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

        return widget

    def _create_monitoring_panel(self) -> QWidget:
        """Create monitoring panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Statistics
        self.stats_widget = DetectionStatsWidget()
        layout.addWidget(self.stats_widget)

        # Confidence graph
        self.conf_graph = ConfidenceGraphWidget()
        self.conf_graph.setMaximumHeight(200)
        layout.addWidget(self.conf_graph)

        # Detection log
        log_label = QLabel("Detection Log")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)

        self.log_widget = DetectionLogWidget()
        layout.addWidget(self.log_widget)

        return widget

    def add_detection(self, detection: Dict):
        """Add detection result"""
        self.detection_buffer.append(detection)

        # Add to log
        self.log_widget.add_detection(detection)

        # Add confidence to graph
        if 'confidence' in detection:
            self.conf_graph.add_confidence(detection['confidence'])

    def _update_display(self):
        """Update statistics display"""
        if not self.detection_buffer:
            return

        # Calculate statistics
        total = len(self.detection_buffer)

        # Detection rate (last 10 seconds)
        now = datetime.now()
        recent = [d for d in self.detection_buffer
                  if (now - d.get('timestamp', now)).total_seconds() < 10]
        rate = len(recent) / 10.0 if recent else 0

        # Average confidence
        confidences = [d.get('confidence', 0) for d in self.detection_buffer]
        avg_conf = sum(confidences) / len(confidences) * 100 if confidences else 0

        # Average processing time
        times = [d.get('processing_time', 0) for d in self.detection_buffer
                 if 'processing_time' in d]
        avg_time = sum(times) / len(times) if times else 0

        # Update stats widget
        stats = {
            'total_detections': total,
            'detection_rate': rate,
            'avg_confidence': avg_conf,
            'processing_time': avg_time
        }
        self.stats_widget.update_stats(stats)

    def _on_classes_changed(self, classes: List[str]):
        """Handle class filter change"""
        logger.info(f"Selected classes: {classes}")

    def _apply_settings(self):
        """Apply detection settings"""
        settings = {
            'model': self.model_combo.currentText(),
            'backend': self.backend_combo.currentText(),
            'confidence_threshold': self.conf_slider.value() / 100.0,
            'nms_threshold': self.nms_slider.value() / 100.0,
            'selected_classes': self.class_filter.get_selected_classes()
        }

        self.settings_changed.emit(settings)
        logger.info("Detection settings applied", settings=settings)