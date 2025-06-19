# src/ui/widgets/roi_editor.py
"""
Interactive ROI (Region of Interest) and line editor widget.
"""

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QMenu
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPolygonF, QPainter,
    QMouseEvent, QKeyEvent
)
from typing import List, Tuple, Optional, Dict
import numpy as np

class InteractiveHandle(QGraphicsItem):
    """Draggable handle for editing shapes"""

    def __init__(self, parent=None, index: int = 0):
        super().__init__(parent)
        self.index = index
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(1000)

        # Appearance
        self.radius = 6
        self.color = QColor(255, 255, 0)
        self.hover_color = QColor(255, 200, 0)
        self.is_hovered = False

    def boundingRect(self) -> QRectF:
        return QRectF(-self.radius, -self.radius,
                      2 * self.radius, 2 * self.radius)

    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw handle
        color = self.hover_color if self.is_hovered else self.color
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self.boundingRect())

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.setCursor(Qt.ClosedHandCursor)
        self.update()

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.setCursor(Qt.OpenHandCursor)
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.parentItem():
            # Notify parent shape about position change
            self.parentItem().handle_moved(self.index, value)

        return super().itemChange(change, value)


class EditablePolygon(QGraphicsItem):
    """Editable polygon shape for ROI"""

    def __init__(self, points: List[QPointF] = None):
        super().__init__()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        # Polygon data
        self.points = points or []
        self.handles = []

        # Appearance
        self.pen = QPen(QColor(0, 255, 0), 2)
        self.brush = QBrush(QColor(0, 255, 0, 50))
        self.selected_pen = QPen(QColor(255, 255, 0), 3)

        # Create handles if points exist
        if self.points:
            self._create_handles()

    def _create_handles(self):
        """Create draggable handles for each point"""
        # Remove existing handles
        for handle in self.handles:
            if handle.scene():
                handle.scene().removeItem(handle)

        self.handles = []

        # Create new handles
        for i, point in enumerate(self.points):
            handle = InteractiveHandle(self, i)
            handle.setPos(point)
            self.handles.append(handle)

    def add_point(self, point: QPointF):
        """Add a new point to polygon"""
        self.points.append(point)

        # Create handle for new point
        handle = InteractiveHandle(self, len(self.points) - 1)
        handle.setPos(point)
        self.handles.append(handle)

        self.update()

    def handle_moved(self, index: int, new_pos: QPointF):
        """Update polygon when handle is moved"""
        if 0 <= index < len(self.points):
            self.points[index] = new_pos
            self.update()

    def boundingRect(self) -> QRectF:
        if not self.points:
            return QRectF()

        # Calculate bounding rectangle
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]

        margin = 10
        return QRectF(min(xs) - margin, min(ys) - margin,
                      max(xs) - min(xs) + 2 * margin,
                      max(ys) - min(ys) + 2 * margin)

    def paint(self, painter: QPainter, option, widget):
        if not self.points:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        # Draw polygon
        pen = self.selected_pen if self.isSelected() else self.pen
        painter.setPen(pen)
        painter.setBrush(self.brush)

        polygon = QPolygonF(self.points)
        painter.drawPolygon(polygon)

        # Draw points
        painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(QBrush(Qt.white))
        for point in self.points:
            painter.drawEllipse(point, 3, 3)

    def remove_point(self, index: int):
        """Remove a point from polygon"""
        if len(self.points) > 3 and 0 <= index < len(self.points):
            self.points.pop(index)

            # Remove handle
            handle = self.handles.pop(index)
            if handle.scene():
                handle.scene().removeItem(handle)

            # Update remaining handle indices
            for i, handle in enumerate(self.handles):
                handle.index = i

            self.update()

    def get_points(self) -> List[Tuple[float, float]]:
        """Get polygon points as list of tuples"""
        return [(p.x(), p.y()) for p in self.points]


class EditableLine(QGraphicsItem):
    """Editable line for counting"""

    def __init__(self, start: QPointF, end: QPointF):
        super().__init__()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # Line data
        self.start_point = start
        self.end_point = end
        self.handles = []

        # Appearance
        self.pen = QPen(QColor(255, 0, 0), 3)
        self.selected_pen = QPen(QColor(255, 255, 0), 4)
        self.arrow_pen = QPen(QColor(255, 0, 0), 2)

        # Direction
        self.show_direction = True

        # Create handles
        self._create_handles()

    def _create_handles(self):
        """Create draggable handles for endpoints"""
        # Start handle
        self.start_handle = InteractiveHandle(self, 0)
        self.start_handle.setPos(self.start_point)
        self.handles.append(self.start_handle)

        # End handle
        self.end_handle = InteractiveHandle(self, 1)
        self.end_handle.setPos(self.end_point)
        self.handles.append(self.end_handle)

    def handle_moved(self, index: int, new_pos: QPointF):
        """Update line when handle is moved"""
        if index == 0:
            self.start_point = new_pos
        elif index == 1:
            self.end_point = new_pos

        self.update()

    def boundingRect(self) -> QRectF:
        margin = 20
        x1, y1 = self.start_point.x(), self.start_point.y()
        x2, y2 = self.end_point.x(), self.end_point.y()

        return QRectF(min(x1, x2) - margin, min(y1, y2) - margin,
                      abs(x2 - x1) + 2 * margin,
                      abs(y2 - y1) + 2 * margin)

    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw line
        pen = self.selected_pen if self.isSelected() else self.pen
        painter.setPen(pen)
        painter.drawLine(self.start_point, self.end_point)

        # Draw direction arrow
        if self.show_direction:
            self._draw_arrow(painter)

    def _draw_arrow(self, painter: QPainter):
        """Draw direction arrow at midpoint"""
        # Calculate midpoint
        mid_x = (self.start_point.x() + self.end_point.x()) / 2
        mid_y = (self.start_point.y() + self.end_point.y()) / 2
        mid_point = QPointF(mid_x, mid_y)

        # Calculate angle
        dx = self.end_point.x() - self.start_point.x()
        dy = self.end_point.y() - self.start_point.y()
        angle = np.arctan2(dy, dx)

        # Arrow parameters
        arrow_length = 15
        arrow_angle = 0.5

        # Calculate arrow points
        x1 = mid_x - arrow_length * np.cos(angle - arrow_angle)
        y1 = mid_y - arrow_length * np.sin(angle - arrow_angle)
        x2 = mid_x - arrow_length * np.cos(angle + arrow_angle)
        y2 = mid_y - arrow_length * np.sin(angle + arrow_angle)

        # Draw arrow
        painter.setPen(self.arrow_pen)
        painter.drawLine(mid_point, QPointF(x1, y1))
        painter.drawLine(mid_point, QPointF(x2, y2))

    def get_endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get line endpoints as tuples"""
        return ((self.start_point.x(), self.start_point.y()),
                (self.end_point.x(), self.end_point.y()))


class ROIEditor(QGraphicsView):
    """Interactive ROI and line editor"""

    # Signals
    roi_created = Signal(list)  # List of points
    line_created = Signal(dict)  # Line data
    shape_selected = Signal(object)  # Selected shape
    shape_deleted = Signal(object)  # Deleted shape

    def __init__(self):
        super().__init__()

        # Scene setup
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Editor state
        self.edit_mode = None  # 'roi', 'line', 'select'
        self.current_shape = None
        self.temp_points = []

        # Shapes
        self.rois = []
        self.lines = []

        # Appearance
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_edit_mode(self, mode: str):
        """Set editor mode"""
        self.edit_mode = mode
        self.temp_points = []

        # Update cursor
        if mode in ['roi', 'line']:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self.edit_mode == 'roi':
                self._handle_roi_click(scene_pos)
            elif self.edit_mode == 'line':
                self._handle_line_click(scene_pos)
            elif self.edit_mode == 'select':
                # Let default handling select items
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def _handle_roi_click(self, pos: QPointF):
        """Handle click in ROI mode"""
        if not self.current_shape:
            # Start new ROI
            self.current_shape = EditablePolygon()
            self.scene.addItem(self.current_shape)

        # Add point to ROI
        self.current_shape.add_point(pos)
        self.temp_points.append(pos)

    def _handle_line_click(self, pos: QPointF):
        """Handle click in line mode"""
        self.temp_points.append(pos)

        if len(self.temp_points) == 1:
            # First point - show preview
            pass
        elif len(self.temp_points) == 2:
            # Second point - create line
            line = EditableLine(self.temp_points[0], self.temp_points[1])
            self.scene.addItem(line)
            self.lines.append(line)

            # Emit signal
            line_data = {
                'start': (self.temp_points[0].x(), self.temp_points[0].y()),
                'end': (self.temp_points[1].x(), self.temp_points[1].y())
            }
            self.line_created.emit(line_data)

            # Reset
            self.temp_points = []

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double click to finish ROI"""
        if self.edit_mode == 'roi' and self.current_shape:
            if len(self.current_shape.points) >= 3:
                # Finish ROI
                self.rois.append(self.current_shape)
                self.roi_created.emit(self.current_shape.get_points())

                # Reset
                self.current_shape = None
                self.temp_points = []

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Escape:
            # Cancel current operation
            if self.current_shape and self.current_shape not in self.rois:
                self.scene.removeItem(self.current_shape)

            self.current_shape = None
            self.temp_points = []
            self.set_edit_mode('select')

        elif event.key() == Qt.Key_Delete:
            # Delete selected items
            for item in self.scene.selectedItems():
                self._delete_shape(item)

        super().keyPressEvent(event)

    def _show_context_menu(self, pos):
        """Show context menu"""
        menu = QMenu(self)

        # Check if clicking on an item
        scene_pos = self.mapToScene(pos)
        item = self.scene.itemAt(scene_pos, self.transform())

        if item and not isinstance(item, InteractiveHandle):
            # Shape-specific actions
            edit_action = menu.addAction("Edit Properties")
            edit_action.triggered.connect(lambda: self._edit_shape(item))

            delete_action = menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self._delete_shape(item))

            menu.addSeparator()

        # General actions
        if self.edit_mode != 'roi':
            roi_action = menu.addAction("Draw ROI")
            roi_action.triggered.connect(lambda: self.set_edit_mode('roi'))

        if self.edit_mode != 'line':
            line_action = menu.addAction("Draw Line")
            line_action.triggered.connect(lambda: self.set_edit_mode('line'))

        menu.addSeparator()

        clear_action = menu.addAction("Clear All")
        clear_action.triggered.connect(self.clear_all)

        menu.exec(self.mapToGlobal(pos))

    def _edit_shape(self, shape):
        """Edit shape properties"""
        self.shape_selected.emit(shape)

    def _delete_shape(self, shape):
        """Delete a shape"""
        if shape in self.rois:
            self.rois.remove(shape)
        elif shape in self.lines:
            self.lines.remove(shape)

        self.scene.removeItem(shape)
        self.shape_deleted.emit(shape)

    def clear_all(self):
        """Clear all shapes"""
        for item in self.scene.items():
            if not isinstance(item, InteractiveHandle):
                self.scene.removeItem(item)

        self.rois.clear()
        self.lines.clear()
        self.current_shape = None
        self.temp_points = []

    def get_all_shapes(self) -> Dict:
        """Get all shapes data"""
        return {
            'rois': [roi.get_points() for roi in self.rois],
            'lines': [line.get_endpoints() for line in self.lines]
        }

    def load_shapes(self, shapes_data: Dict):
        """Load shapes from data"""
        self.clear_all()

        # Load ROIs
        for roi_points in shapes_data.get('rois', []):
            points = [QPointF(x, y) for x, y in roi_points]
            roi = EditablePolygon(points)
            self.scene.addItem(roi)
            self.rois.append(roi)

        # Load lines
        for line_endpoints in shapes_data.get('lines', []):
            start = QPointF(*line_endpoints[0])
            end = QPointF(*line_endpoints[1])
            line = EditableLine(start, end)
            self.scene.addItem(line)
            self.lines.append(line)