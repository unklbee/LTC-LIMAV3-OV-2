# src/ui/widgets/timeline_graph.py
"""
Real-time timeline graph widget using pyqtgraph.
"""

import pyqtgraph as pg
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QColor
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class TimelineGraph(pg.PlotWidget):
    """Real-time timeline graph with smooth updates"""

    # Signals
    time_range_changed = Signal(datetime, datetime)

    def __init__(self, title: str = "Timeline",
                 window_minutes: int = 10,
                 update_interval: int = 1000):
        super().__init__()

        # Configuration
        self.window_minutes = window_minutes
        self.update_interval = update_interval

        # Data storage
        self.max_points = window_minutes * 60  # 1 point per second
        self.time_data = deque(maxlen=self.max_points)
        self.data_series = {}  # series_name -> deque of values
        self.plot_items = {}   # series_name -> PlotDataItem

        # Colors for different series
        self.colors = [
            '#2196F3',  # Blue
            '#4CAF50',  # Green
            '#FF9800',  # Orange
            '#9C27B0',  # Purple
            '#00BCD4',  # Cyan
            '#F44336',  # Red
            '#FFEB3B',  # Yellow
        ]
        self.color_index = 0

        # Setup graph
        self._setup_graph(title)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(update_interval)

        # Auto-range timer
        self.auto_range_timer = QTimer()
        self.auto_range_timer.timeout.connect(self._auto_range)
        self.auto_range_timer.start(5000)  # Every 5 seconds

    def _setup_graph(self, title: str):
        """Setup graph appearance"""
        # Dark theme
        self.setBackground('#1e1e1e')

        # Configure axes
        self.setLabel('left', 'Value')
        self.setLabel('bottom', 'Time')
        self.setTitle(title, color='w', size='14pt')

        # Grid
        self.showGrid(x=True, y=True, alpha=0.3)

        # Time axis formatting
        self.getAxis('bottom').setStyle(tickTextOffset=10)

        # Legend
        self.addLegend(offset=(-10, 10))

        # Enable mouse interaction
        self.setMouseEnabled(x=True, y=True)

        # Anti-aliasing
        self.setAntialiasing(True)

    def add_series(self, name: str, color: Optional[str] = None):
        """Add a new data series"""
        if name in self.data_series:
            return

        # Initialize data storage
        self.data_series[name] = deque(maxlen=self.max_points)

        # Choose color
        if color is None:
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1

        # Create plot item
        pen = pg.mkPen(color, width=2)
        self.plot_items[name] = self.plot(
            pen=pen,
            name=name,
            symbol=None,
            symbolSize=4,
            symbolBrush=color
        )

    def add_data_point(self, values: Dict[str, float]):
        """Add new data point for all series"""
        current_time = datetime.now()
        self.time_data.append(current_time)

        # Add values for each series
        for name, value in values.items():
            if name not in self.data_series:
                self.add_series(name)

            self.data_series[name].append(value)

        # Pad missing values with None
        for name in self.data_series:
            if name not in values:
                self.data_series[name].append(None)

    def _update_display(self):
        """Update graph display"""
        if not self.time_data:
            return

        # Convert time to seconds from start
        time_array = np.array([t.timestamp() for t in self.time_data])
        if len(time_array) > 0:
            time_relative = time_array - time_array[0]
        else:
            time_relative = np.array([])

        # Update each series
        for name, plot_item in self.plot_items.items():
            values = list(self.data_series[name])

            # Filter out None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            if valid_indices:
                valid_times = time_relative[valid_indices]
                valid_values = [values[i] for i in valid_indices]
                plot_item.setData(valid_times, valid_values)

    def _auto_range(self):
        """Auto-range Y axis based on visible data"""
        # Get current X range
        x_range = self.viewRange()[0]

        if not self.time_data:
            return

        # Find visible data points
        time_array = np.array([t.timestamp() for t in self.time_data])
        time_relative = time_array - time_array[0]

        visible_mask = (time_relative >= x_range[0]) & (time_relative <= x_range[1])

        # Find min/max in visible range
        y_min = float('inf')
        y_max = float('-inf')

        for series in self.data_series.values():
            values = np.array(list(series))
            visible_values = values[visible_mask]

            # Filter out None values
            valid_values = visible_values[~np.isnan(visible_values)]

            if len(valid_values) > 0:
                y_min = min(y_min, np.min(valid_values))
                y_max = max(y_max, np.max(valid_values))

        # Set Y range with padding
        if y_min != float('inf') and y_max != float('-inf'):
            padding = (y_max - y_min) * 0.1
            self.setYRange(y_min - padding, y_max + padding)

    def set_window_size(self, minutes: int):
        """Set time window size in minutes"""
        self.window_minutes = minutes
        self.max_points = minutes * 60

        # Update data storage
        for series in self.data_series.values():
            series = deque(series, maxlen=self.max_points)

        self.time_data = deque(self.time_data, maxlen=self.max_points)

    def clear_data(self):
        """Clear all data"""
        self.time_data.clear()
        for series in self.data_series.values():
            series.clear()

        self._update_display()

    def export_data(self) -> Dict:
        """Export current data"""
        data = {
            'time': [t.isoformat() for t in self.time_data],
            'series': {}
        }

        for name, values in self.data_series.items():
            data['series'][name] = list(values)

        return data


class MultiAxisGraph(pg.GraphicsLayoutWidget):
    """Graph with multiple Y axes for different units"""

    def __init__(self, title: str = "Multi-Axis Graph"):
        super().__init__()

        self.setBackground('#1e1e1e')

        # Create plot with first Y axis
        self.plot1 = self.addPlot(title=title)
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        self.plot1.setLabel('bottom', 'Time')

        # Create additional Y axes
        self.plot2 = pg.ViewBox()
        self.plot1.scene().addItem(self.plot2)
        self.plot1.getAxis('right').linkToView(self.plot2)
        self.plot2.setXLink(self.plot1)

        # Show right axis
        self.plot1.showAxis('right')
        self.plot1.getAxis('right').setLabel('Secondary Axis')

        # Update views when plot is resized
        self.plot1.vb.sigResized.connect(self._update_views)

        # Data series
        self.primary_series = {}
        self.secondary_series = {}

    def _update_views(self):
        """Update secondary view to match primary"""
        self.plot2.setGeometry(self.plot1.vb.sceneBoundingRect())
        self.plot2.linkedViewChanged(self.plot1.vb, self.plot2.XAxis)

    def add_primary_series(self, name: str, color: str = 'b'):
        """Add series to primary axis"""
        pen = pg.mkPen(color, width=2)
        self.primary_series[name] = self.plot1.plot(pen=pen, name=name)

    def add_secondary_series(self, name: str, color: str = 'r'):
        """Add series to secondary axis"""
        pen = pg.mkPen(color, width=2)
        item = pg.PlotDataItem(pen=pen, name=name)
        self.plot2.addItem(item)
        self.secondary_series[name] = item

    def update_data(self, primary_data: Dict, secondary_data: Dict):
        """Update all data series"""
        for name, (x, y) in primary_data.items():
            if name in self.primary_series:
                self.primary_series[name].setData(x, y)

        for name, (x, y) in secondary_data.items():
            if name in self.secondary_series:
                self.secondary_series[name].setData(x, y)