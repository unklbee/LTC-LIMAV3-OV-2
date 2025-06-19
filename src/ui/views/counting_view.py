# src/ui/views/counting_view.py
"""
Counting view for configuring counting lines/zones and monitoring counts.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSplitter, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPointF
from PySide6.QtGui import QColor, QIcon, QPen, QBrush, QPolygonF
import pyqtgraph as pg
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np

import structlog

logger = structlog.get_logger()


class CountingLineItem(QListWidgetItem):
    """List item for counting line"""

    def __init__(self, line_id: str, line_data: Dict):
        super().__init__()
        self.line_id = line_id
        self.line_data = line_data

        # Display text
        self.setText(f"Line {line_id}")

        # Set color based on direction
        if line_data.get('bidirectional', False):
            self.setForeground(QColor(0, 255, 255))  # Cyan for bidirectional
        else:
            self.setForeground(QColor(0, 255, 0))  # Green for unidirectional


class CountingZoneItem(QListWidgetItem):
    """List item for counting zone"""

    def __init__(self, zone_id: str, zone_data: Dict):
        super().__init__()
        self.zone_id = zone_id
        self.zone_data = zone_data

        # Display text
        self.setText(f"Zone {zone_id}")
        self.setForeground(QColor(255, 165, 0))  # Orange for zones


class CountingConfigWidget(QGroupBox):
    """Widget for counting configuration"""

    # Signals
    line_added = Signal(dict)
    line_removed = Signal(str)
    zone_added = Signal(dict)
    zone_removed = Signal(str)

    def __init__(self):
        super().__init__("Counting Configuration")

        layout = QVBoxLayout(self)

        # Line configuration
        line_label = QLabel("Counting Lines:")
        layout.addWidget(line_label)

        self.line_list = QListWidget()
        self.line_list.setMaximumHeight(150)
        self.line_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.line_list.customContextMenuRequested.connect(self._show_line_menu)
        layout.addWidget(self.line_list)

        # Line buttons
        line_btn_layout = QHBoxLayout()

        self.add_line_btn = QPushButton("Add Line")
        self.add_line_btn.clicked.connect(self._add_line)
        line_btn_layout.addWidget(self.add_line_btn)

        self.edit_line_btn = QPushButton("Edit Line")
        self.edit_line_btn.clicked.connect(self._edit_line)
        line_btn_layout.addWidget(self.edit_line_btn)

        layout.addLayout(line_btn_layout)

        # Zone configuration
        zone_label = QLabel("Counting Zones:")
        layout.addWidget(zone_label)

        self.zone_list = QListWidget()
        self.zone_list.setMaximumHeight(150)
        self.zone_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.zone_list.customContextMenuRequested.connect(self._show_zone_menu)
        layout.addWidget(self.zone_list)

        # Zone buttons
        zone_btn_layout = QHBoxLayout()

        self.add_zone_btn = QPushButton("Add Zone")
        self.add_zone_btn.clicked.connect(self._add_zone)
        zone_btn_layout.addWidget(self.add_zone_btn)

        self.edit_zone_btn = QPushButton("Edit Zone")
        self.edit_zone_btn.clicked.connect(self._edit_zone)
        zone_btn_layout.addWidget(self.edit_zone_btn)

        layout.addLayout(zone_btn_layout)

        # Speed estimation
        speed_group = QGroupBox("Speed Estimation")
        speed_layout = QVBoxLayout(speed_group)

        self.enable_speed_check = QCheckBox("Enable Speed Estimation")
        self.enable_speed_check.setChecked(True)
        speed_layout.addWidget(self.enable_speed_check)

        ppm_layout = QHBoxLayout()
        ppm_layout.addWidget(QLabel("Pixels per Meter:"))

        self.ppm_spin = QDoubleSpinBox()
        self.ppm_spin.setRange(1.0, 100.0)
        self.ppm_spin.setValue(10.0)
        self.ppm_spin.setSingleStep(0.5)
        ppm_layout.addWidget(self.ppm_spin)

        speed_layout.addLayout(ppm_layout)

        layout.addWidget(speed_group)

    def _add_line(self):
        """Add new counting line"""
        # TODO: Open line drawing dialog
        line_data = {
            'start': (100, 200),
            'end': (500, 200),
            'direction': 'both',
            'bidirectional': True
        }

        line_id = f"L{len(self.line_list) + 1}"
        item = CountingLineItem(line_id, line_data)
        self.line_list.addItem(item)

        self.line_added.emit(line_data)

    def _edit_line(self):
        """Edit selected line"""
        current = self.line_list.currentItem()
        if isinstance(current, CountingLineItem):
            # TODO: Open line editing dialog
            logger.info(f"Editing line: {current.line_id}")

    def _add_zone(self):
        """Add new counting zone"""
        # TODO: Open zone drawing dialog
        zone_data = {
            'polygon': [(100, 100), (200, 100), (200, 200), (100, 200)],
            'type': 'entry_exit'
        }

        zone_id = f"Z{len(self.zone_list) + 1}"
        item = CountingZoneItem(zone_id, zone_data)
        self.zone_list.addItem(item)

        self.zone_added.emit(zone_data)

    def _edit_zone(self):
        """Edit selected zone"""
        current = self.zone_list.currentItem()
        if isinstance(current, CountingZoneItem):
            # TODO: Open zone editing dialog
            logger.info(f"Editing zone: {current.zone_id}")

    def _show_line_menu(self, pos):
        """Show context menu for lines"""
        item = self.line_list.itemAt(pos)
        if not isinstance(item, CountingLineItem):
            return

        menu = QMenu(self)

        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(self._edit_line)

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_line(item))

        menu.exec(self.line_list.mapToGlobal(pos))

    def _show_zone_menu(self, pos):
        """Show context menu for zones"""
        item = self.zone_list.itemAt(pos)
        if not isinstance(item, CountingZoneItem):
            return

        menu = QMenu(self)

        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(self._edit_zone)

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_zone(item))

        menu.exec(self.zone_list.mapToGlobal(pos))

    def _delete_line(self, item: CountingLineItem):
        """Delete counting line"""
        reply = QMessageBox.question(
            self, "Delete Line",
            f"Delete {item.line_id}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.line_list.takeItem(self.line_list.row(item))
            self.line_removed.emit(item.line_id)

    def _delete_zone(self, item: CountingZoneItem):
        """Delete counting zone"""
        reply = QMessageBox.question(
            self, "Delete Zone",
            f"Delete {item.zone_id}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.zone_list.takeItem(self.zone_list.row(item))
            self.zone_removed.emit(item.zone_id)


class CountDisplayWidget(QFrame):
    """Widget for displaying current counts"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Box)
        self.setObjectName("countDisplay")

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Current Counts")
        title.setObjectName("countTitle")
        title.setStyleSheet("""
            #countTitle {
                font-size: 18px;
                font-weight: bold;
                color: #0078d4;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)

        # Count grid
        self.count_grid = QHBoxLayout()
        layout.addLayout(self.count_grid)

        # Vehicle type counts
        self.count_widgets = {}
        vehicle_types = ["Car", "Truck", "Bus", "Motorbike", "Bicycle"]
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"]

        for vtype, color in zip(vehicle_types, colors):
            widget = self._create_count_widget(vtype, color)
            self.count_widgets[vtype.lower()] = widget
            self.count_grid.addWidget(widget)

        # Total count
        self.total_widget = self._create_count_widget("Total", "#0078d4")
        self.count_grid.addWidget(self.total_widget)

        # Reset button
        self.reset_btn = QPushButton("Reset Counts")
        self.reset_btn.clicked.connect(self._reset_counts)
        layout.addWidget(self.reset_btn)

    def _create_count_widget(self, label: str, color: str) -> QWidget:
        """Create individual count widget"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border-radius: 8px;
                border-left: 4px solid {color};
                padding: 15px;
            }}
        """)

        layout = QVBoxLayout(widget)

        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #b0b0b0; font-size: 14px;")
        label_widget.setAlignment(Qt.AlignCenter)

        count_widget = QLabel("0")
        count_widget.setObjectName(f"{label}Count")
        count_widget.setStyleSheet(f"""
            color: {color};
            font-size: 36px;
            font-weight: bold;
        """)
        count_widget.setAlignment(Qt.AlignCenter)

        layout.addWidget(label_widget)
        layout.addWidget(count_widget)

        return widget

    def update_counts(self, counts: Dict[str, int]):
        """Update count display"""
        total = 0

        for vtype, widget in self.count_widgets.items():
            count = counts.get(vtype, 0)
            widget.findChild(QLabel, f"{vtype.capitalize()}Count").setText(str(count))
            total += count

        self.total_widget.findChild(QLabel, "TotalCount").setText(str(total))

    def _reset_counts(self):
        """Reset all counts"""
        reply = QMessageBox.question(
            self, "Reset Counts",
            "Reset all counts to zero?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for widget in self.count_widgets.values():
                widget.findChild(QLabel).setText("0")
            self.total_widget.findChild(QLabel, "TotalCount").setText("0")


class CountingGraphWidget(pg.PlotWidget):
    """Real-time counting graph"""

    def __init__(self):
        super().__init__()

        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('left', 'Count')
        self.setLabel('bottom', 'Time (minutes ago)')
        self.setTitle('Vehicle Count Timeline')

        # Data storage (last 30 minutes)
        self.time_window = 30  # minutes
        self.data_points = 60 * self.time_window  # 1 point per second

        # Time data
        self.time_data = deque(maxlen=self.data_points)

        # Count data per vehicle type
        self.count_data = defaultdict(lambda: deque(maxlen=self.data_points))

        # Plot lines
        self.plots = {}
        self.vehicle_colors = {
            'car': '#2196F3',
            'truck': '#FF9800',
            'bus': '#4CAF50',
            'motorbike': '#9C27B0',
            'bicycle': '#00BCD4'
        }

        # Legend
        self.addLegend(offset=(-10, 10))

        # Initialize plots
        for vtype, color in self.vehicle_colors.items():
            self.plots[vtype] = self.plot(
                pen=pg.mkPen(color, width=2),
                name=vtype.capitalize()
            )

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second

    def add_count(self, vehicle_type: str, count: int = 1):
        """Add count for vehicle type"""
        current_time = datetime.now()

        # Add time point if needed
        if not self.time_data or (current_time - self.time_data[-1]).total_seconds() >= 1:
            self.time_data.append(current_time)

            # Add count data
            for vtype in self.vehicle_colors:
                if vtype == vehicle_type:
                    current_count = self.count_data[vtype][-1] + count if self.count_data[vtype] else count
                else:
                    current_count = self.count_data[vtype][-1] if self.count_data[vtype] else 0

                self.count_data[vtype].append(current_count)

    def update_plot(self):
        """Update plot display"""
        if not self.time_data:
            return

        # Convert time to minutes ago
        current_time = datetime.now()
        time_minutes = [
            -(current_time - t).total_seconds() / 60
            for t in self.time_data
        ]

        # Update each plot
        for vtype, plot in self.plots.items():
            if vtype in self.count_data and self.count_data[vtype]:
                plot.setData(time_minutes, list(self.count_data[vtype]))

        # Set x-axis range
        self.setXRange(-self.time_window, 0)


class CountingEventLog(QTableWidget):
    """Log of counting events"""

    def __init__(self):
        super().__init__()

        # Set columns
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels([
            "Time", "Line/Zone", "Vehicle Type", "Direction", "Speed"
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
        self.max_rows = 200

    def add_event(self, event: Dict):
        """Add counting event to log"""
        # Remove old rows if exceeding limit
        if self.rowCount() >= self.max_rows:
            self.removeRow(0)

        # Insert new row
        row = self.rowCount()
        self.insertRow(row)

        # Time
        time_item = QTableWidgetItem(
            datetime.now().strftime("%H:%M:%S")
        )
        self.setItem(row, 0, time_item)

        # Line/Zone
        location_item = QTableWidgetItem(
            event.get('location', 'Unknown')
        )
        self.setItem(row, 1, location_item)

        # Vehicle type
        vtype_item = QTableWidgetItem(
            event.get('vehicle_type', 'Unknown')
        )
        self.setItem(row, 2, vtype_item)

        # Direction
        direction_item = QTableWidgetItem(
            event.get('direction', '-')
        )
        self.setItem(row, 3, direction_item)

        # Speed
        speed = event.get('speed')
        speed_item = QTableWidgetItem(
            f"{speed:.1f} km/h" if speed else "-"
        )
        self.setItem(row, 4, speed_item)

        # Scroll to bottom
        self.scrollToBottom()


class CountingView(QWidget):
    """Main counting configuration and monitoring view"""

    # Signals
    config_changed = Signal(dict)
    reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("countingView")

        # Current counts
        self.current_counts = defaultdict(int)

        # Setup UI
        self._setup_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(1000)  # Update every second

    def _setup_ui(self):
        """Setup counting view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Vehicle Counting")
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

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Configuration
        left_panel = self._create_config_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - Monitoring
        right_panel = self._create_monitoring_panel()
        main_splitter.addWidget(right_panel)

        main_splitter.setSizes([400, 800])
        layout.addWidget(main_splitter)

    def _create_config_panel(self) -> QWidget:
        """Create configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Counting configuration
        self.config_widget = CountingConfigWidget()
        self.config_widget.line_added.connect(self._on_line_added)
        self.config_widget.line_removed.connect(self._on_line_removed)
        self.config_widget.zone_added.connect(self._on_zone_added)
        self.config_widget.zone_removed.connect(self._on_zone_removed)

        layout.addWidget(self.config_widget)

        # Export settings
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout(export_group)

        # Auto-save interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Auto-save Interval:"))

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(5)
        self.interval_spin.setSuffix(" minutes")
        interval_layout.addWidget(self.interval_spin)

        export_layout.addLayout(interval_layout)

        # Export format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Export Format:"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel", "CSV", "JSON"])
        format_layout.addWidget(self.format_combo)

        export_layout.addLayout(format_layout)

        layout.addWidget(export_group)

        layout.addStretch()

        return widget

    def _create_monitoring_panel(self) -> QWidget:
        """Create monitoring panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Current counts
        self.count_display = CountDisplayWidget()
        layout.addWidget(self.count_display)

        # Counting graph
        self.count_graph = CountingGraphWidget()
        self.count_graph.setMaximumHeight(300)
        layout.addWidget(self.count_graph)

        # Event log
        log_label = QLabel("Counting Events")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)

        self.event_log = CountingEventLog()
        layout.addWidget(self.event_log)

        return widget

    def add_count_event(self, event: Dict):
        """Add counting event"""
        # Update counts
        vtype = event.get('vehicle_type', 'unknown')
        self.current_counts[vtype] += 1

        # Update graph
        self.count_graph.add_count(vtype)

        # Add to event log
        self.event_log.add_event(event)

        # Update display
        self._update_display()

    def _update_display(self):
        """Update count display"""
        self.count_display.update_counts(dict(self.current_counts))

    def _on_line_added(self, line_data: Dict):
        """Handle line added"""
        config = {
            'type': 'line_added',
            'data': line_data
        }
        self.config_changed.emit(config)

    def _on_line_removed(self, line_id: str):
        """Handle line removed"""
        config = {
            'type': 'line_removed',
            'line_id': line_id
        }
        self.config_changed.emit(config)

    def _on_zone_added(self, zone_data: Dict):
        """Handle zone added"""
        config = {
            'type': 'zone_added',
            'data': zone_data
        }
        self.config_changed.emit(config)

    def _on_zone_removed(self, zone_id: str):
        """Handle zone removed"""
        config = {
            'type': 'zone_removed',
            'zone_id': zone_id
        }
        self.config_changed.emit(config)

    def reset_counts(self):
        """Reset all counts"""
        self.current_counts.clear()
        self._update_display()
        self.reset_requested.emit()