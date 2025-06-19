# src/ui/views/video_view.py
"""
Enhanced video view with overlay controls and interactive ROI/line drawing.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QToolBar, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsLineItem, QMenu, QInputDialog
)
from PySide6.QtCore import (
    Qt, Signal, QPointF, QRectF, QTimer, QDateTime
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QPolygonF,
    QColor, QAction, QKeySequence, QMouseEvent
)
import numpy as np
from typing import List, Optional, Dict, Tuple
import cv2

import structlog

logger = structlog.get_logger()


class InteractiveVideoWidget(QGraphicsView):
    """Interactive video display with drawing capabilities"""

    # Signals
    roi_defined = Signal(list)  # List of points
    line_defined = Signal(dict)  # Line data

    def __init__(self):
        super().__init__()

        # Scene setup
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Video display
        self.video_item = QGraphicsPixmapItem()
        self.scene.addItem(self.video_item)

        # Drawing state
        self.drawing_mode = None  # 'roi', 'line', None
        self.drawing_points = []
        self.temp_items = []

        # Permanent items
        self.roi_item = None
        self.line_items = []

        # Current frame info
        self.current_frame = None
        self.frame_size = (640, 480)

        # Styling
        self.roi_pen = QPen(QColor(0, 255, 0, 200), 2)
        self.roi_brush = QBrush(QColor(0, 255, 0, 50))
        self.line_pen = QPen(QColor(255, 0, 0, 200), 3)
        self.temp_pen = QPen(QColor(255, 255, 0, 200), 2, Qt.DashLine)

        # Setup
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def display_frame(self, frame: np.ndarray):
        """Display video frame"""
        self.current_frame = frame
        height, width = frame.shape[:2]
        self.frame_size = (width, height)

        if len(frame.shape) == 2:
            # Grayscale
            bytes_per_line = width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color
            bytes_per_line = 3 * width
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Update pixmap
        pixmap = QPixmap.fromImage(q_image)
        self.video_item.setPixmap(pixmap)

        # Fit in view
        self.fitInView(self.video_item, Qt.KeepAspectRatio)

    def start_roi_drawing(self):
        """Start ROI drawing mode"""
        self.drawing_mode = 'roi'
        self.drawing_points = []
        self._clear_temp_items()
        self.setCursor(Qt.CrossCursor)
        logger.info("Started ROI drawing mode")

    def start_line_drawing(self):
        """Start line drawing mode"""
        self.drawing_mode = 'line'
        self.drawing_points = []
        self._clear_temp_items()
        self.setCursor(Qt.CrossCursor)
        logger.info("Started line drawing mode")

    def cancel_drawing(self):
        """Cancel current drawing"""
        self.drawing_mode = None
        self.drawing_points = []
        self._clear_temp_items()
        self.setCursor(Qt.ArrowCursor)
        logger.info("Cancelled drawing mode")

    def clear_overlays(self):
        """Clear all overlays"""
        # Remove ROI
        if self.roi_item:
            self.scene.removeItem(self.roi_item)
            self.roi_item = None

        # Remove lines
        for item in self.line_items:
            self.scene.removeItem(item)
        self.line_items.clear()

        self._clear_temp_items()
        logger.info("Cleared all overlays")

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if self.drawing_mode and event.button() == Qt.LeftButton:
            # Convert to scene coordinates
            scene_pos = self.mapToScene(event.pos())

            # Ensure within video bounds
            if self.video_item.contains(scene_pos):
                self.drawing_points.append(scene_pos)
                self._update_temp_drawing()

                # Check if drawing is complete
                if self.drawing_mode == 'line' and len(self.drawing_points) == 2:
                    self._finish_line_drawing()

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double click to finish ROI"""
        if self.drawing_mode == 'roi' and len(self.drawing_points) >= 3:
            self._finish_roi_drawing()

        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for preview"""
        if self.drawing_mode and self.drawing_points:
            # Update preview with current mouse position
            scene_pos = self.mapToScene(event.pos())
            if self.video_item.contains(scene_pos):
                self._update_temp_drawing(scene_pos)

        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Escape:
            self.cancel_drawing()
        elif event.key() == Qt.Key_Return and self.drawing_mode == 'roi':
            if len(self.drawing_points) >= 3:
                self._finish_roi_drawing()

        super().keyPressEvent(event)

    def _update_temp_drawing(self, preview_point: Optional[QPointF] = None):
        """Update temporary drawing preview"""
        self._clear_temp_items()

        if not self.drawing_points:
            return

        if self.drawing_mode == 'roi':
            # Draw polygon preview
            points = self.drawing_points[:]
            if preview_point:
                points.append(preview_point)

            if len(points) >= 2:
                # Draw lines
                for i in range(len(points) - 1):
                    line = self.scene.addLine(
                        points[i].x(), points[i].y(),
                        points[i + 1].x(), points[i + 1].y(),
                        self.temp_pen
                    )
                    self.temp_items.append(line)

                # Close polygon if we have at least 3 points
                if len(points) >= 3 and preview_point:
                    line = self.scene.addLine(
                        points[-1].x(), points[-1].y(),
                        points[0].x(), points[0].y(),
                        self.temp_pen
                    )
                    self.temp_items.append(line)

        elif self.drawing_mode == 'line' and len(self.drawing_points) == 1:
            # Draw line preview
            if preview_point:
                line = self.scene.addLine(
                    self.drawing_points[0].x(), self.drawing_points[0].y(),
                    preview_point.x(), preview_point.y(),
                    self.temp_pen
                )
                self.temp_items.append(line)

        # Draw points
        for point in self.drawing_points:
            ellipse = self.scene.addEllipse(
                point.x() - 3, point.y() - 3, 6, 6,
                self.temp_pen, QBrush(QColor(255, 255, 0))
            )
            self.temp_items.append(ellipse)

    def _clear_temp_items(self):
        """Clear temporary drawing items"""
        for item in self.temp_items:
            self.scene.removeItem(item)
        self.temp_items.clear()

    def _finish_roi_drawing(self):
        """Finish ROI drawing"""
        if len(self.drawing_points) < 3:
            return

        # Remove old ROI if exists
        if self.roi_item:
            self.scene.removeItem(self.roi_item)

        # Create polygon
        polygon = QPolygonF(self.drawing_points)
        self.roi_item = QGraphicsPolygonItem(polygon)
        self.roi_item.setPen(self.roi_pen)
        self.roi_item.setBrush(self.roi_brush)
        self.scene.addItem(self.roi_item)

        # Convert to list of tuples
        roi_points = [(p.x(), p.y()) for p in self.drawing_points]
        self.roi_defined.emit(roi_points)

        # Reset drawing mode
        self.cancel_drawing()
        logger.info(f"ROI defined with {len(roi_points)} points")

    def _finish_line_drawing(self):
        """Finish line drawing"""
        if len(self.drawing_points) != 2:
            return

        # Create line
        line = QGraphicsLineItem(
            self.drawing_points[0].x(), self.drawing_points[0].y(),
            self.drawing_points[1].x(), self.drawing_points[1].y()
        )
        line.setPen(self.line_pen)
        self.scene.addItem(line)
        self.line_items.append(line)

        # Emit line data
        line_data = {
            'id': f"line_{len(self.line_items)}",
            'start': (self.drawing_points[0].x(), self.drawing_points[0].y()),
            'end': (self.drawing_points[1].x(), self.drawing_points[1].y())
        }
        self.line_defined.emit(line_data)

        # Reset drawing mode
        self.cancel_drawing()
        logger.info(f"Line defined: {line_data['id']}")

    def _show_context_menu(self, pos):
        """Show context menu"""
        menu = QMenu(self)

        if self.drawing_mode:
            cancel_action = menu.addAction("Cancel Drawing")
            cancel_action.triggered.connect(self.cancel_drawing)
        else:
            if self.roi_item:
                clear_roi_action = menu.addAction("Clear ROI")
                clear_roi_action.triggered.connect(self._clear_roi)

            if self.line_items:
                clear_lines_action = menu.addAction("Clear Lines")
                clear_lines_action.triggered.connect(self._clear_lines)

            menu.addSeparator()

            snapshot_action = menu.addAction("Take Snapshot")
            snapshot_action.triggered.connect(self._take_snapshot)

        menu.exec(self.mapToGlobal(pos))

    def _clear_roi(self):
        """Clear ROI"""
        if self.roi_item:
            self.scene.removeItem(self.roi_item)
            self.roi_item = None
            logger.info("ROI cleared")

    def _clear_lines(self):
        """Clear all lines"""
        for item in self.line_items:
            self.scene.removeItem(item)
        self.line_items.clear()
        logger.info("All lines cleared")

    def _take_snapshot(self):
        """Take snapshot of current frame"""
        if self.current_frame is not None:
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            filename = f"snapshot_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            logger.info(f"Snapshot saved: {filename}")

    def resizeEvent(self, event):
        """Handle resize to maintain aspect ratio"""
        super().resizeEvent(event)
        if self.video_item.pixmap():
            self.fitInView(self.video_item, Qt.KeepAspectRatio)


class VideoControlBar(QWidget):
    """Video control toolbar"""

    # Signals
    play_toggled = Signal(bool)
    speed_changed = Signal(float)
    snapshot_requested = Signal()
    record_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("videoControlBar")
        self.setFixedHeight(60)

        # State
        self.is_playing = False
        self.is_recording = False

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)

        # Play/Pause button
        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("playButton")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._toggle_play)

        # Speed control
        speed_label = QLabel("Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(25, 200)  # 0.25x to 2x
        self.speed_slider.setValue(100)  # 1x
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.speed_value = QLabel("1.0x")
        self.speed_value.setFixedWidth(40)

        # Snapshot button
        self.snapshot_btn = QPushButton("📷")
        self.snapshot_btn.setToolTip("Take Snapshot")
        self.snapshot_btn.clicked.connect(self.snapshot_requested)

        # Record button
        self.record_btn = QPushButton("⚫")
        self.record_btn.setObjectName("recordButton")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip("Record Video")
        self.record_btn.clicked.connect(self._toggle_record)

        # Statistics display
        self.stats_label = QLabel("FPS: 0 | Detected: 0")
        self.stats_label.setObjectName("statsLabel")

        # Add to layout
        layout.addWidget(self.play_btn)
        layout.addSpacing(20)
        layout.addWidget(speed_label)
        layout.addWidget(self.speed_slider)
        layout.addWidget(self.speed_value)
        layout.addSpacing(20)
        layout.addWidget(self.snapshot_btn)
        layout.addWidget(self.record_btn)
        layout.addStretch()
        layout.addWidget(self.stats_label)

        self._apply_styles()

    def _apply_styles(self):
        """Apply control bar styling"""
        self.setStyleSheet("""
            #videoControlBar {
                background-color: #2d2d2d;
                border-top: 1px solid #3d3d3d;
            }
            
            #playButton, #recordButton {
                background-color: #3d3d3d;
                border: none;
                border-radius: 20px;
                color: white;
                font-size: 18px;
            }
            
            #playButton:hover, #recordButton:hover {
                background-color: #4d4d4d;
            }
            
            #playButton:checked {
                background-color: #0078d4;
            }
            
            #recordButton:checked {
                background-color: #f44336;
                color: white;
            }
            
            #statsLabel {
                color: #b0b0b0;
                font-size: 12px;
            }
            
            QSlider::groove:horizontal {
                height: 4px;
                background: #3d3d3d;
                border-radius: 2px;
            }
            
            QSlider::handle:horizontal {
                background: #0078d4;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)

    def _toggle_play(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.play_btn.setText("⏸" if self.is_playing else "▶")
        self.play_toggled.emit(self.is_playing)

    def _toggle_record(self):
        """Toggle recording"""
        self.is_recording = not self.is_recording
        self.record_btn.setText("⏹" if self.is_recording else "⚫")
        self.record_toggled.emit(self.is_recording)

    def _on_speed_changed(self, value):
        """Handle speed change"""
        speed = value / 100.0
        self.speed_value.setText(f"{speed:.1f}x")
        self.speed_changed.emit(speed)

    def update_stats(self, fps: float, detected: int):
        """Update statistics display"""
        self.stats_label.setText(f"FPS: {fps:.1f} | Detected: {detected}")


class VideoView(QWidget):
    """Main video view with source selection and controls"""

    # Signals
    source_changed = Signal(object)  # Video source
    model_changed = Signal(str)  # Model name
    start_requested = Signal()
    stop_requested = Signal()
    roi_defined = Signal(list)
    line_defined = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setObjectName("videoView")

        # Available options
        self.vehicle_classes = []
        self.model_names = []

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup video view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video display (create first)
        self.video_widget = InteractiveVideoWidget()
        self.video_widget.roi_defined.connect(self.roi_defined)
        self.video_widget.line_defined.connect(self.line_defined)

        # Toolbar (create after video_widget)
        self.toolbar = self._create_toolbar()
        layout.addWidget(self.toolbar)

        # Add video widget
        layout.addWidget(self.video_widget, 1)

        # Control bar
        self.control_bar = VideoControlBar()
        layout.addWidget(self.control_bar)

    def _create_toolbar(self) -> QToolBar:
        """Create video toolbar"""
        toolbar = QToolBar()
        toolbar.setObjectName("videoToolbar")
        toolbar.setMovable(False)

        # Source selection
        source_label = QLabel("Source:")
        toolbar.addWidget(source_label)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "File", "RTSP Stream"])
        self.source_combo.currentTextChanged.connect(self._on_source_type_changed)
        toolbar.addWidget(self.source_combo)

        self.source_btn = QPushButton("Select")
        self.source_btn.clicked.connect(self._select_source)
        toolbar.addWidget(self.source_btn)

        toolbar.addSeparator()

        # Model selection
        model_label = QLabel("Model:")
        toolbar.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.model_changed)
        toolbar.addWidget(self.model_combo)

        toolbar.addSeparator()

        # ROI/Line tools
        self.roi_btn = QAction("Define ROI", self)
        self.roi_btn.setShortcut(QKeySequence("Ctrl+R"))
        self.roi_btn.triggered.connect(self.video_widget.start_roi_drawing)
        toolbar.addAction(self.roi_btn)

        self.line_btn = QAction("Define Line", self)
        self.line_btn.setShortcut(QKeySequence("Ctrl+L"))
        self.line_btn.triggered.connect(self.video_widget.start_line_drawing)
        toolbar.addAction(self.line_btn)

        self.clear_btn = QAction("Clear Overlays", self)
        self.clear_btn.triggered.connect(self.video_widget.clear_overlays)
        toolbar.addAction(self.clear_btn)

        toolbar.addSeparator()

        # Start/Stop
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setObjectName("startButton")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self._toggle_detection)
        toolbar.addWidget(self.start_btn)

        # Style
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border-bottom: 1px solid #3d3d3d;
                padding: 5px;
                spacing: 10px;
            }
            
            QLabel {
                color: white;
                margin: 0 5px;
            }
            
            #startButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            #startButton:hover {
                background-color: #45a049;
            }
            
            #startButton:checked {
                background-color: #f44336;
            }
        """)

        return toolbar

    def _on_source_type_changed(self, source_type: str):
        """Handle source type change"""
        if source_type == "Webcam":
            self.source_btn.setText("Select Camera")
        elif source_type == "File":
            self.source_btn.setText("Browse...")
        else:
            self.source_btn.setText("Enter URL")

    def _select_source(self):
        """Select video source"""
        source_type = self.source_combo.currentText()

        if source_type == "Webcam":
            # Show camera selection dialog
            cameras = self._get_available_cameras()
            if cameras:
                camera, ok = QInputDialog.getItem(
                    self, "Select Camera", "Available cameras:",
                    [f"Camera {i}" for i in range(len(cameras))],
                    0, False
                )
                if ok:
                    cam_index = int(camera.split()[-1])
                    self.source_changed.emit(cam_index)
            else:
                self.source_changed.emit(0)  # Default camera

        elif source_type == "File":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
            )

            if file_path:
                self.source_changed.emit(file_path)

        else:  # RTSP Stream
            url, ok = QInputDialog.getText(
                self, "Enter Stream URL",
                "Stream URL (rtsp://, http://, etc):",
                text="rtsp://localhost:8554/stream"
            )
            if ok and url:
                self.source_changed.emit(url)

    def _get_available_cameras(self) -> List[int]:
        """Get list of available cameras"""
        cameras = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

    def _toggle_detection(self):
        """Toggle detection on/off"""
        if self.start_btn.isChecked():
            self.start_btn.setText("Stop Detection")
            self.start_requested.emit()
        else:
            self.start_btn.setText("Start Detection")
            self.stop_requested.emit()

    def set_vehicle_classes(self, classes: List[str]):
        """Set available vehicle classes"""
        self.vehicle_classes = classes

    def set_available_models(self, models: List[str]):
        """Set available models"""
        self.model_names = models
        self.model_combo.clear()
        self.model_combo.addItems(models)

    def get_selected_model(self) -> str:
        """Get currently selected model"""
        return self.model_combo.currentText()

    def display_frame(self, frame_data):
        """Display processed frame"""
        # Extract frame and annotations
        frame = frame_data.raw_frame.copy()  # Make a copy to avoid modifying original

        # Draw overlays (detections, tracks, etc.)
        if hasattr(frame_data, 'tracks') and frame_data.tracks:
            for track in frame_data.tracks:
                # Draw bounding box
                x1, y1, x2, y2 = track.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw track ID and class
                label = f"ID: {track.track_id}"
                if hasattr(track, 'class_name'):
                    label += f" ({track.class_name})"

                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw background rectangle for text
                cv2.rectangle(frame,
                              (x1, y1 - text_height - 4),
                              (x1 + text_width, y1),
                              (0, 255, 0), -1)

                # Draw text
                cv2.putText(frame, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Update display
        self.video_widget.display_frame(frame)

        # Update stats
        if hasattr(frame_data, 'metadata'):
            fps = frame_data.metadata.get('fps', 0)
            detected = len(frame_data.tracks) if hasattr(frame_data, 'tracks') and frame_data.tracks else 0
            self.control_bar.update_stats(fps, detected)

    def set_detection_active(self, active: bool):
        """Update UI to reflect detection state"""
        self.start_btn.setChecked(active)
        self.start_btn.setText("Stop Detection" if active else "Start Detection")

    def get_roi_points(self) -> Optional[List[Tuple[float, float]]]:
        """Get current ROI points if defined"""
        if self.video_widget.roi_item:
            polygon = self.video_widget.roi_item.polygon()
            return [(p.x(), p.y()) for p in polygon]
        return None

    def get_counting_lines(self) -> List[Dict[str, any]]:
        """Get all defined counting lines"""
        lines = []
        for i, line_item in enumerate(self.video_widget.line_items):
            line = line_item.line()
            lines.append({
                'id': f"line_{i+1}",
                'start': (line.x1(), line.y1()),
                'end': (line.x2(), line.y2())
            })
        return lines