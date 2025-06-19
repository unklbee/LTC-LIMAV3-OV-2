# src/ui/widgets/stat_card.py
"""
Statistics card widget for displaying metrics with animations.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, Property,
    Signal, QTimer
)
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QBrush, QPen

class StatCard(QFrame):
    """Animated statistics card widget"""

    # Signals
    clicked = Signal()
    value_changed = Signal(str)

    def __init__(self, title: str = "", value: str = "0",
                 subtitle: str = "", color: str = "#0078d4", parent=None):
        super().__init__(parent)

        self.setObjectName("statCard")
        self.setCursor(Qt.PointingHandCursor)

        # Properties
        self.title = title
        self._value = value
        self.subtitle = subtitle
        self.color = QColor(color)
        self._animated_value = 0
        self._target_value = 0

        # Setup UI
        self._setup_ui()

        # Setup animations
        self._setup_animations()

        # Apply initial style
        self._apply_style()

    def _setup_ui(self):
        """Setup card UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("cardTitle")

        # Value
        self.value_label = QLabel(self._value)
        self.value_label.setObjectName("cardValue")
        self.value_label.setAlignment(Qt.AlignCenter)

        # Subtitle
        self.subtitle_label = QLabel(self.subtitle)
        self.subtitle_label.setObjectName("cardSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        # Trend indicator
        trend_layout = QHBoxLayout()
        self.trend_icon = QLabel()
        self.trend_label = QLabel()
        self.trend_label.setObjectName("cardTrend")

        trend_layout.addStretch()
        trend_layout.addWidget(self.trend_icon)
        trend_layout.addWidget(self.trend_label)
        trend_layout.addStretch()

        # Add to layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)
        layout.addLayout(trend_layout)
        layout.addStretch()

    def _setup_animations(self):
        """Setup value animation"""
        self.value_animation = QPropertyAnimation(self, b"animated_value")
        self.value_animation.setDuration(1000)
        self.value_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.value_animation.valueChanged.connect(self._update_value_display)

    def _apply_style(self):
        """Apply card styling"""
        self.setStyleSheet(f"""
            #statCard {{
                background-color: #2d2d2d;
                border-radius: 10px;
                border-left: 4px solid {self.color.name()};
            }}
            
            #statCard:hover {{
                background-color: #3d3d3d;
            }}
            
            #cardTitle {{
                color: #b0b0b0;
                font-size: 14px;
                font-weight: 500;
            }}
            
            #cardValue {{
                color: {self.color.name()};
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            #cardSubtitle {{
                color: #808080;
                font-size: 12px;
            }}
            
            #cardTrend {{
                font-size: 12px;
                font-weight: bold;
            }}
        """)

    @Property(int)
    def animated_value(self):
        return self._animated_value

    @animated_value.setter
    def animated_value(self, value):
        self._animated_value = value
        self._update_value_display()

    def set_value(self, value: str, animate: bool = True):
        """Set card value with optional animation"""
        self._value = value

        # Check if value is numeric for animation
        try:
            numeric_value = float(value.replace(',', '').replace('%', ''))

            if animate and hasattr(self, '_target_value'):
                # Animate from current to new value
                self.value_animation.setStartValue(self._animated_value)
                self.value_animation.setEndValue(int(numeric_value))
                self.value_animation.start()
            else:
                self._animated_value = int(numeric_value)
                self.value_label.setText(value)

            self._target_value = numeric_value

        except ValueError:
            # Non-numeric value, just set it
            self.value_label.setText(value)

        self.value_changed.emit(value)

    def _update_value_display(self):
        """Update value label during animation"""
        # Format the animated value
        if '.' in self._value:
            display_value = f"{self._animated_value:,.1f}"
        else:
            display_value = f"{self._animated_value:,}"

        # Add suffix if present in original value
        if '%' in self._value:
            display_value += '%'

        self.value_label.setText(display_value)

    def set_trend(self, value: float, format: str = "percent"):
        """Set trend indicator"""
        if value > 0:
            icon = "↑"
            color = "#4CAF50"
        elif value < 0:
            icon = "↓"
            color = "#f44336"
        else:
            icon = "→"
            color = "#808080"

        self.trend_icon.setText(icon)
        self.trend_icon.setStyleSheet(f"color: {color}; font-size: 16px;")

        if format == "percent":
            text = f"{abs(value):.1f}%"
        else:
            text = f"{abs(value):.1f}"

        self.trend_label.setText(text)
        self.trend_label.setStyleSheet(f"color: {color};")

    def set_subtitle(self, subtitle: str):
        """Update subtitle text"""
        self.subtitle = subtitle
        self.subtitle_label.setText(subtitle)

    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        """Add hover effect"""
        # Slightly scale up
        self.setStyleSheet(self.styleSheet() + """
            #statCard {
                transform: scale(1.02);
                transition: transform 0.2s;
            }
        """)

    def leaveEvent(self, event):
        """Remove hover effect"""
        self._apply_style()


class MiniStatCard(QFrame):
    """Compact statistics card for sidebar"""

    def __init__(self, label: str = "", value: str = "0",
                 color: str = "#0078d4", parent=None):
        super().__init__(parent)

        self.setObjectName("miniStatCard")
        self.setFixedHeight(60)

        # Setup UI
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

        # Icon/Color indicator
        self.color_indicator = QFrame()
        self.color_indicator.setFixedSize(4, 40)
        self.color_indicator.setStyleSheet(f"background-color: {color};")

        # Labels
        label_layout = QVBoxLayout()
        label_layout.setSpacing(2)

        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet("color: #b0b0b0; font-size: 11px;")

        self.value_widget = QLabel(value)
        self.value_widget.setStyleSheet(f"color: white; font-size: 18px; font-weight: bold;")

        label_layout.addWidget(self.label_widget)
        label_layout.addWidget(self.value_widget)

        # Add to main layout
        layout.addWidget(self.color_indicator)
        layout.addLayout(label_layout)
        layout.addStretch()

        # Style
        self.setStyleSheet("""
            #miniStatCard {
                background-color: #2d2d2d;
                border-radius: 6px;
            }
        """)

    def set_value(self, value: str):
        """Update value"""
        self.value_widget.setText(value)


class LiveStatCard(StatCard):
    """Statistics card with live updates"""

    def __init__(self, *args, update_interval: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_interval = update_interval
        self.update_callback = None

        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_live_value)

    def set_update_callback(self, callback):
        """Set callback function for live updates"""
        self.update_callback = callback

    def start_live_updates(self):
        """Start live value updates"""
        if self.update_callback:
            self.update_timer.start(self.update_interval)

    def stop_live_updates(self):
        """Stop live value updates"""
        self.update_timer.stop()

    def _update_live_value(self):
        """Update value from callback"""
        if self.update_callback:
            try:
                new_value = self.update_callback()
                self.set_value(str(new_value))
            except Exception as e:
                print(f"Error updating live stat: {e}")