# src/ui/widgets/modern_button.py
"""
Modern button widget with animations and hover effects.
"""

from PySide6.QtWidgets import QPushButton, QGraphicsDropShadowEffect
from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QRect, Signal, Property
)
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QBrush, QPen

class ModernButton(QPushButton):
    """Modern button with smooth animations"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)

        # Animation properties
        self._animation_color = QColor(0, 120, 212)
        self._hover_progress = 0.0

        # Setup animations
        self._setup_animations()

        # Apply initial style
        self._apply_style()

        # Enable mouse tracking for smooth hover
        self.setMouseTracking(True)

        # Add shadow effect
        self._add_shadow()

    def _setup_animations(self):
        """Setup hover and click animations"""
        # Hover animation
        self.hover_animation = QPropertyAnimation(self, b"hover_progress")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Click animation
        self.click_animation = QPropertyAnimation(self, b"geometry")
        self.click_animation.setDuration(100)
        self.click_animation.setEasingCurve(QEasingCurve.OutBack)

    def _apply_style(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            ModernButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-height: 36px;
            }
            
            ModernButton:pressed {
                background-color: #106ebe;
            }
            
            ModernButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

    def _add_shadow(self):
        """Add drop shadow effect"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

    @Property(float)
    def hover_progress(self):
        return self._hover_progress

    @hover_progress.setter
    def hover_progress(self, value):
        self._hover_progress = value
        self.update()

    def enterEvent(self, event):
        """Mouse enter animation"""
        self.hover_animation.setStartValue(self._hover_progress)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.start()

        # Update shadow on hover
        shadow = self.graphicsEffect()
        if shadow:
            shadow.setBlurRadius(15)
            shadow.setYOffset(4)

    def leaveEvent(self, event):
        """Mouse leave animation"""
        self.hover_animation.setStartValue(self._hover_progress)
        self.hover_animation.setEndValue(0.0)
        self.hover_animation.start()

        # Reset shadow
        shadow = self.graphicsEffect()
        if shadow:
            shadow.setBlurRadius(10)
            shadow.setYOffset(2)

    def mousePressEvent(self, event):
        """Animate on click"""
        if event.button() == Qt.LeftButton:
            # Shrink animation
            geometry = self.geometry()
            self.click_animation.setStartValue(geometry)

            shrunk = QRect(
                geometry.x() + 2,
                geometry.y() + 2,
                geometry.width() - 4,
                geometry.height() - 4
            )
            self.click_animation.setEndValue(shrunk)
            self.click_animation.finished.connect(self._restore_geometry)
            self.click_animation.start()

        super().mousePressEvent(event)

    def _restore_geometry(self):
        """Restore original geometry after click"""
        self.click_animation.setDirection(QPropertyAnimation.Backward)
        self.click_animation.start()

    def paintEvent(self, event):
        """Custom paint with hover effect"""
        super().paintEvent(event)

        if self._hover_progress > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw hover overlay
            overlay_color = QColor(255, 255, 255, int(30 * self._hover_progress))
            painter.fillRect(self.rect(), overlay_color)


class IconButton(ModernButton):
    """Modern button with icon support"""

    def __init__(self, icon_text: str = "", text: str = "", parent=None):
        super().__init__(text, parent)
        self.icon_text = icon_text

    def paintEvent(self, event):
        """Paint with icon"""
        super().paintEvent(event)

        if self.icon_text:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw icon
            icon_rect = QRect(10, 0, 30, self.height())
            painter.setFont(self.font())
            painter.setPen(QPen(Qt.white))
            painter.drawText(icon_rect, Qt.AlignCenter, self.icon_text)


class PrimaryButton(ModernButton):
    """Primary action button with accent color"""

    def _apply_style(self):
        """Apply primary button style"""
        self.setStyleSheet("""
            PrimaryButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                min-height: 40px;
            }
            
            PrimaryButton:hover {
                background-color: #106ebe;
            }
            
            PrimaryButton:pressed {
                background-color: #005a9e;
            }
        """)


class SecondaryButton(ModernButton):
    """Secondary action button with outline style"""

    def _apply_style(self):
        """Apply secondary button style"""
        self.setStyleSheet("""
            SecondaryButton {
                background-color: transparent;
                color: #0078d4;
                border: 2px solid #0078d4;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-height: 36px;
            }
            
            SecondaryButton:hover {
                background-color: #0078d4;
                color: white;
            }
            
            SecondaryButton:pressed {
                background-color: #106ebe;
                border-color: #106ebe;
            }
        """)

    def _add_shadow(self):
        """No shadow for secondary buttons"""
        pass


class DangerButton(ModernButton):
    """Danger/destructive action button"""

    def _apply_style(self):
        """Apply danger button style"""
        self.setStyleSheet("""
            DangerButton {
                background-color: #d13438;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-height: 36px;
            }
            
            DangerButton:hover {
                background-color: #e81123;
            }
            
            DangerButton:pressed {
                background-color: #a80000;
            }
        """)