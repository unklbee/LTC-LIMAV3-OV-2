# src/ui/widgets/dashboard_view.py
from PySide6.QtWidgets import QWidget, QGridLayout
from PySide6.QtCore import QTimer
import pyqtgraph as pg

class DashboardView(QWidget):
    """Real-time statistics dashboard"""

    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._setup_graphs()

    def _setup_ui(self):
        layout = QGridLayout(self)

        # Statistics cards
        self.fps_card = StatCard("FPS", "0", color="#4CAF50")
        self.total_card = StatCard("Total Vehicles", "0", color="#2196F3")
        self.current_card = StatCard("Current", "0", color="#FF9800")

        # Real-time graph
        self.timeline_graph = TimelineGraph()
        self.timeline_graph.setLabel('left', 'Vehicles/min')
        self.timeline_graph.setLabel('bottom', 'Time')

        # Vehicle type distribution
        self.type_chart = VehicleTypeChart()

        # Heatmap
        self.heatmap = TrafficHeatmap()

        # Layout
        layout.addWidget(self.fps_card, 0, 0)
        layout.addWidget(self.total_card, 0, 1)
        layout.addWidget(self.current_card, 0, 2)
        layout.addWidget(self.timeline_graph, 1, 0, 1, 3)
        layout.addWidget(self.type_chart, 2, 0, 1, 2)
        layout.addWidget(self.heatmap, 2, 2)

class TimelineGraph(pg.PlotWidget):
    """Real-time timeline graph with smooth updates"""

    def __init__(self):
        super().__init__()
        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)

        # Data buffers
        self.time_buffer = deque(maxlen=300)  # 5 minutes at 1Hz
        self.count_buffer = deque(maxlen=300)

        # Plot lines for each vehicle type
        self.plots = {}
        for vehicle_type, color in VEHICLE_COLORS.items():
            self.plots[vehicle_type] = self.plot(
                pen=pg.mkPen(color, width=2)
            )