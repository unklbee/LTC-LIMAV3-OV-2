# src/ui/views/main_view.py
"""
Modern main window implementation with fluent design and animations.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QStackedWidget, QSystemTrayIcon, QMenu, QGraphicsDropShadowEffect
)
from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QRect, Signal, Slot, QPoint
)
from PySide6.QtGui import (
    QColor, QFont, QIcon, QPixmap
)

import structlog

logger = structlog.get_logger()


class ModernTitleBar(QWidget):
    """Custom title bar with modern design"""

    minimize_requested = Signal()
    maximize_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setObjectName("titleBar")

        # Mouse tracking for window dragging
        self.mouse_pos = None

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        # App icon and title
        self.icon_label = QLabel()
        self.icon_label.setPixmap(QPixmap("resources/icons/app_icon.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.title_label = QLabel("LIMA Traffic Counter")
        self.title_label.setObjectName("titleLabel")
        font = QFont("Segoe UI", 11)
        font.setWeight(QFont.DemiBold)
        self.title_label.setFont(font)

        # Window controls
        self.btn_minimize = self._create_window_button("−")
        self.btn_maximize = self._create_window_button("□")
        self.btn_close = self._create_window_button("×")
        self.btn_close.setObjectName("closeButton")

        # Connect signals
        self.btn_minimize.clicked.connect(self.minimize_requested)
        self.btn_maximize.clicked.connect(self.maximize_requested)
        self.btn_close.clicked.connect(self.close_requested)

        # Add to layout
        layout.addWidget(self.icon_label)
        layout.addSpacing(10)
        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.btn_minimize)
        layout.addWidget(self.btn_maximize)
        layout.addWidget(self.btn_close)

    def _create_window_button(self, text: str) -> QPushButton:
        """Create window control button"""
        btn = QPushButton(text)
        btn.setFixedSize(46, 32)
        btn.setObjectName("windowButton")
        return btn

    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.LeftButton:
            self.mouse_pos = event.globalPos() - self.window().pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if self.mouse_pos:
            self.window().move(event.globalPos() - self.mouse_pos)

    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        self.mouse_pos = None


class AnimatedSideBar(QWidget):
    """Animated sidebar with navigation items"""

    item_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sideBar")
        self.setFixedWidth(280)

        # Animation
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.InOutQuart)

        # State
        self.collapsed = False
        self.current_item = "dashboard"

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(60)
        header.setObjectName("sideBarHeader")
        header_layout = QHBoxLayout(header)

        self.logo_label = QLabel("LIMA")
        self.logo_label.setObjectName("logoLabel")
        font = QFont("Segoe UI", 18, QFont.Bold)
        self.logo_label.setFont(font)

        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setObjectName("toggleButton")
        self.toggle_btn.setFixedSize(40, 40)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        header_layout.addWidget(self.logo_label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_btn)

        # Navigation items
        self.nav_items = []
        nav_data = [
            ("dashboard", "📊", "Dashboard"),
            ("video", "🎥", "Video Source"),
            ("detection", "🎯", "Detection"),
            ("counting", "🔢", "Counting"),
            ("database", "💾", "Database"),
            ("settings", "⚙️", "Settings")
        ]

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(10, 20, 10, 20)
        nav_layout.setSpacing(5)

        for item_id, icon, text in nav_data:
            item = NavItem(item_id, icon, text)
            item.clicked.connect(self._on_nav_clicked)
            self.nav_items.append(item)
            nav_layout.addWidget(item)

        nav_layout.addStretch()

        # Stats widget
        self.stats_widget = QuickStatsWidget()
        nav_layout.addWidget(self.stats_widget)

        # Add to main layout
        layout.addWidget(header)
        layout.addWidget(nav_container)

        # Select first item
        self.nav_items[0].set_active(True)

    def toggle_sidebar(self):
        """Toggle sidebar collapsed state"""
        if self.collapsed:
            self.animation.setStartValue(60)
            self.animation.setEndValue(280)
            self.collapsed = False
        else:
            self.animation.setStartValue(280)
            self.animation.setEndValue(60)
            self.collapsed = True

        self.animation.start()

        # Update items
        for item in self.nav_items:
            item.set_collapsed(self.collapsed)

    @Slot(str)
    def _on_nav_clicked(self, item_id: str):
        """Handle navigation item click"""
        # Update active state
        for item in self.nav_items:
            item.set_active(item.item_id == item_id)

        self.current_item = item_id
        self.item_clicked.emit(item_id)


class NavItem(QWidget):
    """Navigation item with hover effects"""

    clicked = Signal(str)

    def __init__(self, item_id: str, icon: str, text: str):
        super().__init__()
        self.item_id = item_id
        self.icon = icon
        self.text = text
        self.active = False
        self.collapsed = False

        self.setFixedHeight(48)
        self.setCursor(Qt.PointingHandCursor)

        # Hover animation
        self.hover_animation = QPropertyAnimation(self, b"pos")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.OutCubic)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)

        self.icon_label = QLabel(icon)
        self.icon_label.setObjectName("navIcon")
        self.icon_label.setFixedWidth(30)

        self.text_label = QLabel(text)
        self.text_label.setObjectName("navText")

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addStretch()

    def set_active(self, active: bool):
        """Set active state"""
        self.active = active
        self.setProperty("active", active)
        self.style().polish(self)

    def set_collapsed(self, collapsed: bool):
        """Set collapsed state"""
        self.collapsed = collapsed
        self.text_label.setVisible(not collapsed)

    def enterEvent(self, event):
        """Mouse enter animation"""
        if not self.active:
            self.hover_animation.setStartValue(self.pos())
            self.hover_animation.setEndValue(self.pos() + QPoint(5, 0))
            self.hover_animation.start()

    def leaveEvent(self, event):
        """Mouse leave animation"""
        if not self.active:
            self.hover_animation.setStartValue(self.pos())
            self.hover_animation.setEndValue(self.pos() - QPoint(5, 0))
            self.hover_animation.start()

    def mousePressEvent(self, event):
        """Handle click"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.item_id)


class QuickStatsWidget(QFrame):
    """Quick statistics display"""

    def __init__(self):
        super().__init__()
        self.setObjectName("quickStats")
        self.setFixedHeight(150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Quick Stats")
        title.setObjectName("statsTitle")

        # Stats
        self.fps_label = self._create_stat_label("FPS", "0")
        self.total_label = self._create_stat_label("Total", "0")
        self.active_label = self._create_stat_label("Active", "0")

        layout.addWidget(title)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.total_label)
        layout.addWidget(self.active_label)

    def _create_stat_label(self, label: str, value: str) -> QWidget:
        """Create stat label widget"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label_w = QLabel(f"{label}:")
        label_w.setObjectName("statLabel")

        value_w = QLabel(value)
        value_w.setObjectName("statValue")
        value_w.setAlignment(Qt.AlignRight)

        layout.addWidget(label_w)
        layout.addStretch()
        layout.addWidget(value_w)

        return widget

    def update_stats(self, fps: float, total: int, active: int):
        """Update statistics"""
        self.fps_label.findChild(QLabel, "statValue").setText(f"{fps:.1f}")
        self.total_label.findChild(QLabel, "statValue").setText(str(total))
        self.active_label.findChild(QLabel, "statValue").setText(str(active))


class ModernMainWindow(QMainWindow):
    """Main application window with modern design"""

    def __init__(self):
        super().__init__()

        # Remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Window properties
        self.setMinimumSize(1280, 720)
        self.resize(1400, 800)

        # Setup UI
        self._setup_ui()
        self._apply_styles()
        self._setup_animations()

        # System tray
        self._setup_system_tray()

        # Show with animation
        self._show_with_animation()

    def _setup_ui(self):
        """Setup user interface"""
        # Main container with rounded corners
        self.container = QWidget()
        self.container.setObjectName("mainContainer")
        self.setCentralWidget(self.container)

        # Main layout
        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Custom title bar
        self.title_bar = ModernTitleBar()
        self.title_bar.minimize_requested.connect(self.showMinimized)
        self.title_bar.maximize_requested.connect(self._toggle_maximize)
        self.title_bar.close_requested.connect(self.close)

        # Content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar
        self.sidebar = AnimatedSideBar()
        self.sidebar.item_clicked.connect(self._on_nav_changed)

        # Stacked widget for content pages
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("contentStack")

        # Add pages
        from src.ui.views.dashboard_view import DashboardView
        from src.ui.views.video_view import VideoView
        from src.ui.views.detection_view import DetectionView
        from src.ui.views.counting_view import CountingView
        from src.ui.views.database_view import DatabaseView
        from src.ui.views.settings_view import SettingsView

        self.pages = {
            "dashboard": DashboardView(),
            "video": VideoView(),
            "detection": DetectionView(),
            "counting": CountingView(),
            "database": DatabaseView(),
            "settings": SettingsView()
        }

        for page in self.pages.values():
            self.content_stack.addWidget(page)

        # Add to layouts
        content_layout.addWidget(self.sidebar)
        content_layout.addWidget(self.content_stack)

        main_layout.addWidget(self.title_bar)
        main_layout.addWidget(content_widget)

        # Drop shadow effect
        self._add_shadow()

    def _apply_styles(self):
        """Apply modern dark theme styles"""
        style = """
        /* Main Container */
        #mainContainer {
            background-color: #1e1e1e;
            border-radius: 10px;
        }
        
        /* Title Bar */
        #titleBar {
            background-color: #2d2d2d;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        
        #titleLabel {
            color: #ffffff;
            padding-left: 5px;
        }
        
        #windowButton {
            background-color: transparent;
            border: none;
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
        }
        
        #windowButton:hover {
            background-color: #3d3d3d;
        }
        
        #closeButton:hover {
            background-color: #e81123;
        }
        
        /* Sidebar */
        #sideBar {
            background-color: #252525;
            border-right: 1px solid #3d3d3d;
        }
        
        #sideBarHeader {
            background-color: #2d2d2d;
            border-bottom: 1px solid #3d3d3d;
        }
        
        #logoLabel {
            color: #0078d4;
            padding-left: 20px;
        }
        
        #toggleButton {
            background-color: transparent;
            border: none;
            color: #ffffff;
            font-size: 18px;
        }
        
        #toggleButton:hover {
            background-color: #3d3d3d;
            border-radius: 5px;
        }
        
        /* Navigation Items */
        NavItem {
            background-color: transparent;
            border-radius: 8px;
            margin: 2px 0;
        }
        
        NavItem:hover {
            background-color: #2d2d2d;
        }
        
        NavItem[active="true"] {
            background-color: #0078d4;
        }
        
        #navIcon {
            font-size: 20px;
        }
        
        #navText {
            color: #ffffff;
            font-size: 14px;
            font-weight: 500;
        }
        
        /* Quick Stats */
        #quickStats {
            background-color: #2d2d2d;
            border-radius: 8px;
            border: 1px solid #3d3d3d;
        }
        
        #statsTitle {
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        #statLabel {
            color: #b0b0b0;
            font-size: 12px;
        }
        
        #statValue {
            color: #0078d4;
            font-size: 14px;
            font-weight: bold;
        }
        
        /* Content Stack */
        #contentStack {
            background-color: #1e1e1e;
            border-bottom-right-radius: 10px;
        }
        """

        self.setStyleSheet(style)

    def _add_shadow(self):
        """Add drop shadow to window"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.container.setGraphicsEffect(shadow)

    def _setup_animations(self):
        """Setup window animations"""
        # Fade in animation
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)

        # Scale animation
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(300)
        self.scale_animation.setEasingCurve(QEasingCurve.OutBack)

    def _show_with_animation(self):
        """Show window with animation"""
        # Center on screen
        screen = self.screen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2

        # Set initial position (slightly above final position)
        self.move(x, y - 20)

        # Setup scale animation
        start_rect = QRect(x + 50, y + 30, self.width() - 100, self.height() - 60)
        end_rect = QRect(x, y, self.width(), self.height())

        self.scale_animation.setStartValue(start_rect)
        self.scale_animation.setEndValue(end_rect)

        # Start animations
        self.show()
        self.fade_animation.start()
        self.scale_animation.start()

    def _setup_system_tray(self):
        """Setup system tray icon"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("resources/icons/app_icon.png"))
        self.tray_icon.setToolTip("LIMA Traffic Counter")

        # Tray menu
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)

        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Double click to show
        self.tray_icon.activated.connect(self._on_tray_activated)

    def _on_tray_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    @Slot(str)
    def _on_nav_changed(self, page_id: str):
        """Handle navigation change"""
        if page_id in self.pages:
            # Animate transition
            self.content_stack.setCurrentWidget(self.pages[page_id])

    def _toggle_maximize(self):
        """Toggle window maximized state"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def closeEvent(self, event):
        """Handle close event"""
        # Minimize to tray instead of closing
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "LIMA Traffic Counter",
            "Application minimized to tray",
            QSystemTrayIcon.Information,
            2000
        )