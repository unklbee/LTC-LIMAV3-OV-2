# src/ui/views/dashboard_view.py
"""
Real-time dashboard with statistics, graphs, and heatmaps.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QScrollArea, QComboBox, QPushButton
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QFont, QLinearGradient, QPen
import pyqtgraph as pg
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


class StatCard(QFrame):
    """Modern statistics card with animations"""

    def __init__(self, title: str, value: str = "0",
                 subtitle: str = "", color: str = "#0078d4"):
        super().__init__()
        self.setObjectName("statCard")
        self.setFixedHeight(120)
        self.color = color

        # Animation for value changes
        self.value_animation = QPropertyAnimation(self, b"value")
        self.value_animation.setDuration(500)
        self.value_animation.setEasingCurve(QEasingCurve.OutCubic)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("statCardTitle")

        # Value
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statCardValue")
        self.value_label.setAlignment(Qt.AlignCenter)

        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("statCardSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        # Trend indicator
        self.trend_label = QLabel()
        self.trend_label.setObjectName("statCardTrend")
        self.trend_label.setAlignment(Qt.AlignRight)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.trend_label)
        layout.addStretch()

        self._apply_style()

    def _apply_style(self):
        """Apply card styling"""
        self.setStyleSheet(f"""
            #statCard {{
                background-color: #2d2d2d;
                border-radius: 10px;
                border-left: 4px solid {self.color};
            }}
            #statCardTitle {{
                color: #b0b0b0;
                font-size: 14px;
                font-weight: 500;
            }}
            #statCardValue {{
                color: {self.color};
                font-size: 32px;
                font-weight: bold;
                margin: 5px 0;
            }}
            #statCardSubtitle {{
                color: #808080;
                font-size: 12px;
            }}
            #statCardTrend {{
                color: #4CAF50;
                font-size: 12px;
                font-weight: bold;
            }}
        """)

    def update_value(self, value: str, animate: bool = True):
        """Update card value with optional animation"""
        if animate and value.isdigit() and self.value_label.text().isdigit():
            # Animate numeric values
            start = int(self.value_label.text())
            end = int(value)
            self.value_animation.setStartValue(start)
            self.value_animation.setEndValue(end)
            self.value_animation.valueChanged.connect(
                lambda v: self.value_label.setText(str(v))
            )
            self.value_animation.start()
        else:
            self.value_label.setText(value)

    def set_trend(self, value: float, is_percentage: bool = True):
        """Set trend indicator"""
        if value > 0:
            prefix = "↑"
            color = "#4CAF50"
        elif value < 0:
            prefix = "↓"
            color = "#f44336"
        else:
            prefix = "→"
            color = "#808080"

        if is_percentage:
            text = f"{prefix} {abs(value):.1f}%"
        else:
            text = f"{prefix} {abs(value):.1f}"

        self.trend_label.setText(text)
        self.trend_label.setStyleSheet(f"color: {color};")


class TimelineGraph(pg.PlotWidget):
    """Real-time timeline graph with smooth updates"""

    def __init__(self, title: str = "Vehicle Count Timeline",
                 window_minutes: int = 10):
        super().__init__()

        self.window_minutes = window_minutes
        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)

        # Configure axes
        self.setLabel('left', 'Vehicles/min')
        self.setLabel('bottom', 'Time')
        self.setTitle(title, color='w', size='14pt')

        # Time axis formatting
        self.getAxis('bottom').setStyle(showValues=False)

        # Data storage
        self.time_data = deque(maxlen=window_minutes * 60)  # 1 sample/sec
        self.count_data = defaultdict(lambda: deque(maxlen=window_minutes * 60))

        # Plot lines for each vehicle type
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

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second

    def add_data_point(self, counts: Dict[str, int]):
        """Add new data point"""
        current_time = datetime.now()
        self.time_data.append(current_time.timestamp())

        for vehicle_type, color in self.vehicle_colors.items():
            count = counts.get(vehicle_type, 0)
            self.count_data[vehicle_type].append(count)

            # Create plot if doesn't exist
            if vehicle_type not in self.plots:
                self.plots[vehicle_type] = self.plot(
                    pen=pg.mkPen(color, width=2),
                    name=vehicle_type.capitalize()
                )

    def update_plot(self):
        """Update plot with latest data"""
        if not self.time_data:
            return

        # Convert timestamps to relative time (seconds from start)
        time_array = np.array(self.time_data)
        time_relative = time_array - time_array[0]

        # Update each plot
        for vehicle_type, plot in self.plots.items():
            if vehicle_type in self.count_data:
                count_array = np.array(self.count_data[vehicle_type])
                plot.setData(time_relative, count_array)

        # Auto-range
        self.enableAutoRange()


class VehicleDistributionChart(pg.PlotWidget):
    """Pie/bar chart showing vehicle type distribution"""

    def __init__(self):
        super().__init__()
        self.setBackground('#1e1e1e')
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setTitle('Vehicle Distribution', color='w', size='14pt')

        # Bar chart items
        self.bars = {}
        self.vehicle_types = ['car', 'truck', 'bus', 'motorbike', 'bicycle']
        self.colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4']

        # Initialize bars
        self._init_bars()

    def _init_bars(self):
        """Initialize bar chart"""
        bar_width = 0.8

        for i, (vtype, color) in enumerate(zip(self.vehicle_types, self.colors)):
            bar = pg.BarGraphItem(
                x=[i],
                height=[0],
                width=bar_width,
                brush=color,
                pen='w'
            )
            self.addItem(bar)
            self.bars[vtype] = bar

        # Add labels
        self.getAxis('bottom').setTicks([
            [(i, vtype.capitalize()) for i, vtype in enumerate(self.vehicle_types)]
        ])
        self.showAxis('bottom')

    def update_data(self, counts: Dict[str, int]):
        """Update chart with new data"""
        total = sum(counts.values())

        for i, vtype in enumerate(self.vehicle_types):
            count = counts.get(vtype, 0)
            percentage = (count / total * 100) if total > 0 else 0

            # Update bar height
            self.bars[vtype].setOpts(height=[count])

            # Add percentage label
            # Note: pyqtgraph doesn't support text items well,
            # so we'd need custom implementation


class HeatmapWidget(QWidget):
    """Traffic density heatmap visualization"""

    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()
        self.setFixedSize(width // 2, height // 2)

        # Heatmap data
        self.heatmap_size = (height // 20, width // 20)
        self.heatmap_data = np.zeros(self.heatmap_size)
        self.decay_rate = 0.95  # Decay factor per update

        # Colormap
        self.colormap = self._create_colormap()

    def _create_colormap(self):
        """Create heatmap colormap"""
        # Blue -> Green -> Yellow -> Red
        colors = [
            (0, 0, 0),        # Black (no activity)
            (0, 0, 255),      # Blue
            (0, 255, 255),    # Cyan
            (0, 255, 0),      # Green
            (255, 255, 0),    # Yellow
            (255, 0, 0)       # Red (high activity)
        ]

        n_colors = 256
        colormap = []

        for i in range(n_colors):
            # Interpolate between colors
            idx = i / (n_colors - 1) * (len(colors) - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, len(colors) - 1)

            alpha = idx - idx_low

            r = int(colors[idx_low][0] * (1 - alpha) + colors[idx_high][0] * alpha)
            g = int(colors[idx_low][1] * (1 - alpha) + colors[idx_high][1] * alpha)
            b = int(colors[idx_low][2] * (1 - alpha) + colors[idx_high][2] * alpha)

            colormap.append(QColor(r, g, b))

        return colormap

    def add_detection(self, x: float, y: float, confidence: float = 1.0):
        """Add detection to heatmap"""
        # Convert to heatmap coordinates
        hx = int(x / self.width() * self.heatmap_size[1])
        hy = int(y / self.height() * self.heatmap_size[0])

        # Clamp to valid range
        hx = max(0, min(hx, self.heatmap_size[1] - 1))
        hy = max(0, min(hy, self.heatmap_size[0] - 1))

        # Add heat with Gaussian spread
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = hy + dy, hx + dx
                if 0 <= ny < self.heatmap_size[0] and 0 <= nx < self.heatmap_size[1]:
                    distance = np.sqrt(dx**2 + dy**2)
                    heat = confidence * np.exp(-distance**2 / 2)
                    self.heatmap_data[ny, nx] += heat

    def update_heatmap(self):
        """Update heatmap with decay"""
        # Apply decay
        self.heatmap_data *= self.decay_rate

        # Clamp values
        self.heatmap_data = np.clip(self.heatmap_data, 0, 1)

        # Trigger repaint
        self.update()

    def paintEvent(self, event):
        """Paint heatmap"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Cell size
        cell_width = self.width() / self.heatmap_size[1]
        cell_height = self.height() / self.heatmap_size[0]

        # Draw heatmap cells
        for y in range(self.heatmap_size[0]):
            for x in range(self.heatmap_size[1]):
                # Get heat value
                heat = self.heatmap_data[y, x]

                if heat > 0.01:  # Only draw if visible
                    # Map to color
                    color_idx = int(heat * (len(self.colormap) - 1))
                    color = self.colormap[color_idx]

                    # Set opacity based on heat
                    color.setAlphaF(min(heat * 2, 1.0))

                    # Draw cell
                    painter.fillRect(
                        x * cell_width,
                        y * cell_height,
                        cell_width,
                        cell_height,
                        color
                    )


class DashboardView(QWidget):
    """Main dashboard view with all visualizations"""

    update_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("dashboardView")

        # Data storage
        self.current_stats = {}
        self.historical_data = deque(maxlen=3600)  # 1 hour of data

        # Setup UI
        self._setup_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._request_update)
        self.update_timer.start(1000)  # Update every second

    def _setup_ui(self):
        """Setup dashboard UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header with title and controls
        header_widget = self._create_header()
        layout.addWidget(header_widget)

        # Statistics cards
        cards_widget = self._create_stat_cards()
        layout.addWidget(cards_widget)

        # Main content area
        content_widget = self._create_content_area()
        layout.addWidget(content_widget, 1)

    def _create_header(self) -> QWidget:
        """Create header section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Title
        title = QLabel("Real-time Traffic Dashboard")
        title.setObjectName("dashboardTitle")
        title.setStyleSheet("""
            #dashboardTitle {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)

        # Time range selector
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["Last 5 min", "Last 15 min",
                                        "Last 30 min", "Last 1 hour"])
        self.time_range_combo.currentTextChanged.connect(self._on_time_range_changed)

        # Export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self._export_data)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(QLabel("Time Range:"))
        layout.addWidget(self.time_range_combo)
        layout.addWidget(export_btn)

        return widget

    def _create_stat_cards(self) -> QWidget:
        """Create statistics cards section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(15)

        # Create stat cards
        self.fps_card = StatCard("Performance", "0", "FPS", "#4CAF50")
        self.total_card = StatCard("Total Vehicles", "0", "All Time", "#2196F3")
        self.current_card = StatCard("Active Tracks", "0", "Currently", "#FF9800")
        self.rate_card = StatCard("Count Rate", "0", "per minute", "#9C27B0")

        layout.addWidget(self.fps_card)
        layout.addWidget(self.total_card)
        layout.addWidget(self.current_card)
        layout.addWidget(self.rate_card)

        return widget

    def _create_content_area(self) -> QWidget:
        """Create main content area with graphs"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(15)

        # Timeline graph
        self.timeline_graph = TimelineGraph("Vehicle Count Timeline", 10)
        layout.addWidget(self.timeline_graph, 0, 0, 2, 2)

        # Vehicle distribution chart
        self.distribution_chart = VehicleDistributionChart()
        layout.addWidget(self.distribution_chart, 0, 2)

        # Heatmap
        heatmap_frame = QFrame()
        heatmap_frame.setObjectName("heatmapFrame")
        heatmap_frame.setStyleSheet("""
            #heatmapFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        heatmap_layout = QVBoxLayout(heatmap_frame)

        heatmap_title = QLabel("Traffic Density Heatmap")
        heatmap_title.setStyleSheet("color: white; font-weight: bold;")
        heatmap_layout.addWidget(heatmap_title)

        self.heatmap = HeatmapWidget()
        heatmap_layout.addWidget(self.heatmap)

        layout.addWidget(heatmap_frame, 1, 2)

        # Speed statistics
        self.speed_stats = self._create_speed_stats()
        layout.addWidget(self.speed_stats, 2, 0, 1, 3)

        return widget

    def _create_speed_stats(self) -> QWidget:
        """Create speed statistics widget"""
        widget = QFrame()
        widget.setObjectName("speedStats")
        widget.setStyleSheet("""
            #speedStats {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 15px;
            }
        """)

        layout = QHBoxLayout(widget)

        # Speed stat labels
        self.avg_speed_label = self._create_speed_label("Average Speed", "0 km/h")
        self.max_speed_label = self._create_speed_label("Max Speed", "0 km/h")
        self.speed_violations_label = self._create_speed_label("Speed Violations", "0")

        layout.addWidget(self.avg_speed_label)
        layout.addWidget(self.max_speed_label)
        layout.addWidget(self.speed_violations_label)

        return widget

    def _create_speed_label(self, title: str, value: str) -> QWidget:
        """Create speed statistic label"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")

        value_label = QLabel(value)
        value_label.setObjectName(f"{title.replace(' ', '')}Value")
        value_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        return widget

    @Slot()
    def _request_update(self):
        """Request data update from controller"""
        self.update_requested.emit()

    @Slot(str)
    def _on_time_range_changed(self, text: str):
        """Handle time range change"""
        # Parse time range
        if "5 min" in text:
            minutes = 5
        elif "15 min" in text:
            minutes = 15
        elif "30 min" in text:
            minutes = 30
        else:
            minutes = 60

        # Update graph window
        self.timeline_graph.window_minutes = minutes

    def _export_data(self):
        """Export dashboard data"""
        # Implementation for data export
        logger.info("Exporting dashboard data...")

    def update_stats(self, stats: Dict):
        """Update dashboard with new statistics"""
        self.current_stats = stats

        # Update stat cards
        if 'fps' in stats:
            self.fps_card.update_value(f"{stats['fps']:.1f}")

        if 'total_counts' in stats:
            total = sum(stats['total_counts'].values())
            self.total_card.update_value(str(total))

        if 'active_tracks' in stats:
            self.current_card.update_value(str(stats['active_tracks']))

        if 'count_rate' in stats:
            self.rate_card.update_value(f"{stats['count_rate']:.1f}")

        # Update graphs
        if 'counts_per_type' in stats:
            self.timeline_graph.add_data_point(stats['counts_per_type'])
            self.distribution_chart.update_data(stats['total_counts'])

        # Update heatmap
        if 'detections' in stats:
            for det in stats['detections']:
                self.heatmap.add_detection(det['x'], det['y'], det['confidence'])
            self.heatmap.update_heatmap()

        # Update speed stats
        if 'avg_speed' in stats:
            self.avg_speed_label.findChild(QLabel, "AverageSpeedValue").setText(
                f"{stats['avg_speed']:.1f} km/h" if stats['avg_speed'] is not None else "-"
            )
        if 'max_speed' in stats:
            self.max_speed_label.findChild(QLabel, "MaxSpeedValue").setText(
                f"{stats['avg_speed']:.1f} km/h" if stats['avg_speed'] is not None else "-"
            )