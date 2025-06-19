# src/ui/themes/light_theme.py
"""
Light theme for LIMA Traffic Counter application.
"""

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Color palette
COLORS = {
    # Primary colors
    'primary': '#0078d4',
    'primary_hover': '#106ebe',
    'primary_pressed': '#005a9e',

    # Background colors
    'bg_primary': '#ffffff',
    'bg_secondary': '#f3f3f3',
    'bg_tertiary': '#e5e5e5',
    'bg_elevated': '#ffffff',

    # Text colors
    'text_primary': '#323130',
    'text_secondary': '#605e5c',
    'text_disabled': '#a19f9d',

    # Accent colors
    'success': '#107c10',
    'warning': '#ff8c00',
    'error': '#d13438',
    'info': '#0078d4',

    # Border colors
    'border': '#e1dfdd',
    'border_hover': '#c8c6c4',

    # Special colors
    'selection': '#0078d4',
    'hover': 'rgba(0, 0, 0, 0.02)',
    'shadow': 'rgba(0, 0, 0, 0.133)'
}

# Main stylesheet
LIGHT_THEME_STYLESHEET = """
/* Global Styles */
QWidget {
    background-color: #ffffff;
    color: #323130;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
}

/* Main Window */
QMainWindow {
    background-color: #ffffff;
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
    background-color: #f3f2f1;
    color: #a19f9d;
}

/* Secondary Buttons */
QPushButton[secondary="true"] {
    background-color: #ffffff;
    border: 1px solid #8a8886;
    color: #323130;
}

QPushButton[secondary="true"]:hover {
    background-color: #f3f2f1;
    border-color: #8a8886;
}

/* Labels */
QLabel {
    color: #323130;
    background-color: transparent;
}

QLabel[heading="true"] {
    font-size: 24px;
    font-weight: 600;
    color: #323130;
}

QLabel[subheading="true"] {
    font-size: 16px;
    color: #605e5c;
}

/* Line Edits */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #8a8886;
    border-radius: 4px;
    padding: 8px;
    color: #323130;
}

QLineEdit:focus {
    border-color: #0078d4;
    border-width: 2px;
    padding: 7px;
}

QLineEdit:disabled {
    background-color: #f3f2f1;
    color: #a19f9d;
}

/* Combo Boxes */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #8a8886;
    border-radius: 4px;
    padding: 8px;
    min-height: 32px;
}

QComboBox:hover {
    border-color: #323130;
}

QComboBox:focus {
    border-color: #0078d4;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: url(resources/icons/arrow_down_light.png);
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #e1dfdd;
    selection-background-color: #0078d4;
    selection-color: white;
}

/* Spin Boxes */
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #8a8886;
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
    background: #e1dfdd;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #0078d4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 2px solid #ffffff;
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
    border: 1px solid #8a8886;
    background-color: #ffffff;
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
    border: 1px solid #8a8886;
    background-color: #ffffff;
}

QRadioButton::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #e1dfdd;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    background-color: #ffffff;
    color: #323130;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #e1dfdd;
    background-color: #ffffff;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #f3f2f1;
    color: #323130;
    padding: 10px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    border: 1px solid #e1dfdd;
    border-bottom: none;
}

QTabBar::tab:hover {
    background-color: #e1dfdd;
}

/* Tables */
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #faf9f8;
    gridline-color: #e1dfdd;
    border: 1px solid #e1dfdd;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #0078d4;
    color: white;
}

QHeaderView::section {
    background-color: #f3f2f1;
    color: #323130;
    padding: 8px;
    border: none;
    border-right: 1px solid #e1dfdd;
    border-bottom: 1px solid #e1dfdd;
    font-weight: 600;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #f3f2f1;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #c8c6c4;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #8a8886;
}

QScrollBar:horizontal {
    background-color: #f3f2f1;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #c8c6c4;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::add-line, QScrollBar::sub-line {
    border: none;
    background: none;
}

/* Tool Bars */
QToolBar {
    background-color: #f3f2f1;
    border: none;
    border-bottom: 1px solid #e1dfdd;
    spacing: 4px;
    padding: 4px;
}

QToolBar::separator {
    background-color: #e1dfdd;
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
    background-color: #e1dfdd;
}

QToolButton:pressed {
    background-color: #c8c6c4;
}

/* Menu Bar */
QMenuBar {
    background-color: #f3f2f1;
    border-bottom: 1px solid #e1dfdd;
}

QMenuBar::item {
    padding: 8px 12px;
    background-color: transparent;
}

QMenuBar::item:selected {
    background-color: #e1dfdd;
}

/* Menus */
QMenu {
    background-color: #ffffff;
    border: 1px solid #e1dfdd;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #f3f2f1;
}

QMenu::separator {
    height: 1px;
    background-color: #e1dfdd;
    margin: 4px 0;
}

/* Progress Bars */
QProgressBar {
    background-color: #f3f2f1;
    border: 1px solid #e1dfdd;
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
    background-color: #f3f2f1;
    border-top: 1px solid #e1dfdd;
}

/* Tool Tips */
QToolTip {
    background-color: #323130;
    color: #ffffff;
    border: none;
    padding: 6px;
    border-radius: 4px;
}

/* Message Boxes */
QMessageBox {
    background-color: #ffffff;
}

QMessageBox QPushButton {
    min-width: 80px;
}

/* Custom Widgets */
#statCard {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #e1dfdd;
    border-left: 4px solid #0078d4;
}

#statCard:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.133);
}

#miniStatCard {
    background-color: #ffffff;
    border-radius: 6px;
    border: 1px solid #e1dfdd;
}
"""

def create_palette() -> QPalette:
    """Create QPalette for light theme"""
    palette = QPalette()

    # Window colors
    palette.setColor(QPalette.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.WindowText, QColor(50, 49, 48))

    # Base colors (for input widgets)
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(250, 249, 248))

    # Text colors
    palette.setColor(QPalette.Text, QColor(50, 49, 48))
    palette.setColor(QPalette.BrightText, Qt.red)

    # Button colors
    palette.setColor(QPalette.Button, QColor(243, 242, 241))
    palette.setColor(QPalette.ButtonText, QColor(50, 49, 48))

    # Highlight colors
    palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
    palette.setColor(QPalette.HighlightedText, Qt.white)

    # Link colors
    palette.setColor(QPalette.Link, QColor(0, 120, 212))
    palette.setColor(QPalette.LinkVisited, QColor(139, 69, 19))

    # ToolTip colors
    palette.setColor(QPalette.ToolTipBase, QColor(50, 49, 48))
    palette.setColor(QPalette.ToolTipText, Qt.white)

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(161, 159, 157))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(161, 159, 157))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(161, 159, 157))

    return palette

def apply_light_theme(app: QApplication):
    """Apply light theme to application"""
    app.setPalette(create_palette())
    app.setStyleSheet(LIGHT_THEME_STYLESHEET)