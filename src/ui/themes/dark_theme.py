# src/ui/themes/dark_theme.py
"""
Dark theme for LIMA Traffic Counter application.
"""

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

# Color palette
COLORS = {
    # Primary colors
    'primary': '#0078d4',
    'primary_hover': '#106ebe',
    'primary_pressed': '#005a9e',

    # Background colors
    'bg_primary': '#1e1e1e',
    'bg_secondary': '#252525',
    'bg_tertiary': '#2d2d2d',
    'bg_elevated': '#3d3d3d',

    # Text colors
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'text_disabled': '#666666',

    # Accent colors
    'success': '#4CAF50',
    'warning': '#FF9800',
    'error': '#f44336',
    'info': '#2196F3',

    # Border colors
    'border': '#3d3d3d',
    'border_hover': '#4d4d4d',

    # Special colors
    'selection': '#0078d4',
    'hover': 'rgba(255, 255, 255, 0.05)',
    'shadow': 'rgba(0, 0, 0, 0.3)'
}

# Main stylesheet
DARK_THEME_STYLESHEET = """
/* Global Styles */
QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
}

/* Main Window */
QMainWindow {
    background-color: #1e1e1e;
}

/* Buttons */
QPushButton {
    background-color: #0078d4;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 32px;
}

QPushButton:hover {
    background-color: #106ebe;
}

QPushButton:pressed {
    background-color: #005a9e;
}

QPushButton:disabled {
    background-color: #3d3d3d;
    color: #666666;
}

/* Secondary Buttons */
QPushButton[secondary="true"] {
    background-color: transparent;
    border: 2px solid #0078d4;
    color: #0078d4;
}

QPushButton[secondary="true"]:hover {
    background-color: #0078d4;
    color: white;
}

/* Labels */
QLabel {
    color: #ffffff;
    background-color: transparent;
}

QLabel[heading="true"] {
    font-size: 24px;
    font-weight: 600;
    color: #ffffff;
}

QLabel[subheading="true"] {
    font-size: 16px;
    color: #b0b0b0;
}

/* Line Edits */
QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 8px;
    color: #ffffff;
}

QLineEdit:focus {
    border-color: #0078d4;
    outline: none;
}

QLineEdit:disabled {
    background-color: #252525;
    color: #666666;
}

/* Combo Boxes */
QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 8px;
    min-height: 32px;
}

QComboBox:hover {
    border-color: #4d4d4d;
}

QComboBox:focus {
    border-color: #0078d4;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: url(resources/icons/arrow_down.png);
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    selection-background-color: #0078d4;
}

/* Spin Boxes */
QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 8px;
    min-height: 32px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #0078d4;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: #3d3d3d;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #0078d4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #106ebe;
}

/* Check Boxes */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #3d3d3d;
    background-color: #2d2d2d;
}

QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

QCheckBox::indicator:checked:hover {
    background-color: #106ebe;
    border-color: #106ebe;
}

/* Radio Buttons */
QRadioButton {
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #3d3d3d;
    background-color: #2d2d2d;
}

QRadioButton::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    background-color: #1e1e1e;
    color: #ffffff;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #3d3d3d;
    background-color: #2d2d2d;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #b0b0b0;
    padding: 10px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #0078d4;
    color: white;
}

QTabBar::tab:hover {
    background-color: #3d3d3d;
}

/* Tables */
QTableWidget {
    background-color: #2d2d2d;
    alternate-background-color: #252525;
    gridline-color: #3d3d3d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #0078d4;
}

QHeaderView::section {
    background-color: #1e1e1e;
    color: #ffffff;
    padding: 8px;
    border: none;
    font-weight: 600;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #4d4d4d;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5d5d5d;
}

QScrollBar:horizontal {
    background-color: #2d2d2d;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #4d4d4d;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::add-line, QScrollBar::sub-line {
    border: none;
    background: none;
}

/* Tool Bars */
QToolBar {
    background-color: #2d2d2d;
    border: none;
    spacing: 4px;
    padding: 4px;
}

QToolBar::separator {
    background-color: #3d3d3d;
    width: 1px;
    margin: 4px;
}

QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 6px;
}

QToolButton:hover {
    background-color: #3d3d3d;
}

QToolButton:pressed {
    background-color: #4d4d4d;
}

/* Menu Bar */
QMenuBar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #3d3d3d;
}

QMenuBar::item {
    padding: 8px 12px;
    background-color: transparent;
}

QMenuBar::item:selected {
    background-color: #3d3d3d;
}

/* Menus */
QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0078d4;
}

QMenu::separator {
    height: 1px;
    background-color: #3d3d3d;
    margin: 4px 0;
}

/* Progress Bars */
QProgressBar {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 3px;
}

/* Status Bar */
QStatusBar {
    background-color: #2d2d2d;
    border-top: 1px solid #3d3d3d;
}

/* Tool Tips */
QToolTip {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #3d3d3d;
    padding: 6px;
    border-radius: 4px;
}

/* Message Boxes */
QMessageBox {
    background-color: #1e1e1e;
}

QMessageBox QPushButton {
    min-width: 80px;
}

/* Dock Widgets */
QDockWidget {
    color: #ffffff;
}

QDockWidget::title {
    background-color: #2d2d2d;
    padding: 8px;
    border-bottom: 1px solid #3d3d3d;
}

QDockWidget::close-button, QDockWidget::float-button {
    background-color: transparent;
    border: none;
    padding: 4px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #3d3d3d;
    border-radius: 4px;
}

/* Custom Widgets */
#statCard {
    background-color: #2d2d2d;
    border-radius: 10px;
    border-left: 4px solid #0078d4;
}

#statCard:hover {
    background-color: #3d3d3d;
}

#miniStatCard {
    background-color: #2d2d2d;
    border-radius: 6px;
}

/* Splitters */
QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

QSplitter::handle:hover {
    background-color: #0078d4;
}
"""

def create_palette() -> QPalette:
    """Create QPalette for dark theme"""
    palette = QPalette()

    # Window colors
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)

    # Base colors (for input widgets)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))

    # Text colors
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)

    # Button colors
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)

    # Highlight colors
    palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
    palette.setColor(QPalette.HighlightedText, Qt.white)

    # Link colors
    palette.setColor(QPalette.Link, QColor(0, 120, 212))
    palette.setColor(QPalette.LinkVisited, QColor(255, 0, 255))

    # ToolTip colors
    palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipText, Qt.white)

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))

    return palette

def apply_dark_theme(app: QApplication):
    """Apply dark theme to application"""
    app.setPalette(create_palette())
    app.setStyleSheet(DARK_THEME_STYLESHEET)