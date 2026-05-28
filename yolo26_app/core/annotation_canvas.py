from typing import Optional

from dataclasses import dataclass, field

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPen, QBrush, QColor, QPolygonF, QPainter, QKeyEvent
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsRectItem,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneWheelEvent,
)


@dataclass
class AnnotationItem:
    class_index: int
    rect: QRectF = field(default_factory=QRectF)
    polygon: QPolygonF = field(default_factory=QPolygonF)
    item_type: str = "rect"


class _SignalHolder(QObject):
    annotations_changed = pyqtSignal()


class AnnotationScene(QGraphicsScene):
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._current_tool: str = "rect"
        self._current_class_index: int = 0
        self._annotations: list[AnnotationItem] = []
        self._signal_holder = _SignalHolder()
        self.annotations_changed = self._signal_holder.annotations_changed

        self._drawing: bool = False
        self._start_point: QPointF = QPointF()
        self._temp_rect_item: Optional[QGraphicsRectItem] = None
        self._polygon_points: list[QPointF] = []
        self._temp_polygon_item: Optional[QGraphicsPolygonItem] = None
        self._selected_index: int = -1
        self._class_colors: list[str] = []

    @property
    def current_tool(self) -> str:
        return self._current_tool

    @property
    def current_class_index(self) -> int:
        return self._current_class_index

    @property
    def annotations(self) -> list[AnnotationItem]:
        return list(self._annotations)

    def set_class_colors(self, colors: list[str]) -> None:
        self._class_colors = colors

    def _get_color(self, class_index: int) -> str:
        if 0 <= class_index < len(self._class_colors):
            return self._class_colors[class_index]
        return "#FF0000"

    def set_tool(self, tool: str) -> None:
        self._current_tool = tool
        self._cancel_drawing()

    def set_current_class(self, index: int) -> None:
        self._current_class_index = index

    def _cancel_drawing(self) -> None:
        self._drawing = False
        if self._temp_rect_item is not None:
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
        if self._temp_polygon_item is not None:
            self.removeItem(self._temp_polygon_item)
            self._temp_polygon_item = None
        self._polygon_points.clear()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.scenePos()

        if self._current_tool == "rect":
            if event.button() == Qt.MouseButton.LeftButton:
                self._drawing = True
                self._start_point = pos
                color = QColor(self._get_color(self._current_class_index))
                pen = QPen(color, 2)
                self._temp_rect_item = QGraphicsRectItem(QRectF(pos, pos))
                self._temp_rect_item.setPen(pen)
                self.addItem(self._temp_rect_item)

        elif self._current_tool == "polygon":
            if event.button() == Qt.MouseButton.LeftButton:
                self._polygon_points.append(pos)
                self._update_temp_polygon()

        elif self._current_tool == "select":
            if event.button() == Qt.MouseButton.LeftButton:
                self._select_at(pos)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._current_tool == "rect" and self._drawing and self._temp_rect_item is not None:
            rect = QRectF(self._start_point, event.scenePos()).normalized()
            self._temp_rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._current_tool == "rect" and self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self._temp_rect_item is not None:
                rect = self._temp_rect_item.rect()
                self.removeItem(self._temp_rect_item)
                self._temp_rect_item = None
                if rect.width() > 2 and rect.height() > 2:
                    ann = AnnotationItem(
                        class_index=self._current_class_index,
                        rect=rect,
                        item_type="rect",
                    )
                    self._annotations.append(ann)
                    self._draw_annotation(ann, len(self._annotations) - 1)
                    self.annotations_changed.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._current_tool == "polygon" and len(self._polygon_points) >= 3:
            polygon = QPolygonF(self._polygon_points)
            ann = AnnotationItem(
                class_index=self._current_class_index,
                polygon=polygon,
                item_type="polygon",
            )
            self._annotations.append(ann)
            if self._temp_polygon_item is not None:
                self.removeItem(self._temp_polygon_item)
                self._temp_polygon_item = None
            self._polygon_points.clear()
            self._draw_annotation(ann, len(self._annotations) - 1)
            self.annotations_changed.emit()
        super().mouseDoubleClickEvent(event)

    def _update_temp_polygon(self) -> None:
        if self._temp_polygon_item is not None:
            self.removeItem(self._temp_polygon_item)
        color = QColor(self._get_color(self._current_class_index))
        pen = QPen(color, 2)
        polygon = QPolygonF(self._polygon_points)
        self._temp_polygon_item = QGraphicsPolygonItem(polygon)
        self._temp_polygon_item.setPen(pen)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
        self._temp_polygon_item.setBrush(brush)
        self.addItem(self._temp_polygon_item)

    def _select_at(self, pos: QPointF) -> None:
        self._selected_index = -1
        for i in range(len(self._annotations) - 1, -1, -1):
            ann = self._annotations[i]
            if ann.item_type == "rect" and ann.rect.contains(pos):
                self._selected_index = i
                break
            elif ann.item_type == "polygon" and ann.polygon.containsPoint(pos, Qt.FillRule.OddEvenFill):
                self._selected_index = i
                break
        self._redraw_all()

    def _draw_annotation(self, ann: AnnotationItem, index: int) -> None:
        color = QColor(self._get_color(ann.class_index))
        pen = QPen(color, 2)
        if index == self._selected_index:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(3)

        if ann.item_type == "rect":
            item = QGraphicsRectItem(ann.rect)
            item.setPen(pen)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
            item.setBrush(brush)
            item.setData(0, index)
            self.addItem(item)
        elif ann.item_type == "polygon":
            item = QGraphicsPolygonItem(ann.polygon)
            item.setPen(pen)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
            item.setBrush(brush)
            item.setData(0, index)
            self.addItem(item)

        label_text = f"{ann.class_index}"
        label = QGraphicsTextItem(label_text)
        label.setDefaultTextColor(color)
        pos_x = ann.rect.left() if ann.item_type == "rect" else ann.polygon.boundingRect().left()
        pos_y = (ann.rect.top() - 20) if ann.item_type == "rect" else (ann.polygon.boundingRect().top() - 20)
        label.setPos(pos_x, pos_y)
        label.setData(0, index)
        self.addItem(label)

    def _redraw_all(self) -> None:
        for item in self.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsTextItem)):
                self.removeItem(item)
        for i, ann in enumerate(self._annotations):
            self._draw_annotation(ann, i)

    def delete_selected(self) -> None:
        if 0 <= self._selected_index < len(self._annotations):
            self._annotations.pop(self._selected_index)
            self._selected_index = -1
            self._redraw_all()
            self.annotations_changed.emit()

    def clear_annotations(self) -> None:
        self._annotations.clear()
        self._selected_index = -1
        self._cancel_drawing()
        for item in self.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsTextItem)):
                self.removeItem(item)

    def get_annotations(self) -> list[AnnotationItem]:
        return list(self._annotations)

    def load_annotations(self, annotations: list[AnnotationItem]) -> None:
        self._annotations = list(annotations)
        self._selected_index = -1
        self._cancel_drawing()
        self._redraw_all()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        super().keyPressEvent(event)


class AnnotationView(QGraphicsView):
    def __init__(self, scene: AnnotationScene, parent: Optional[QObject] = None) -> None:
        super().__init__(scene, parent)
        self._scene = scene
        self._scale_factor: float = 1.0
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def fit_to_item(self) -> None:
        rect = self._scene.itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._scale_factor = 1.0

    def wheelEvent(self, event: QGraphicsSceneWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._scale_factor *= factor
        self.scale(factor, factor)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)
