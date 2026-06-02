from typing import List, Optional

from dataclasses import dataclass, field

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPen, QBrush, QColor, QPolygonF, QPainter, QKeyEvent
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsRectItem,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneWheelEvent,
)


@dataclass
class AnnotationItem:
    class_index: int
    rect: QRectF = field(default_factory=QRectF)
    polygon: QPolygonF = field(default_factory=QPolygonF)
    item_type: str = "rect"
    keypoints: List[QPointF] = field(default_factory=list)


class _SignalHolder(QObject):
    annotations_changed = pyqtSignal()


class AnnotationScene(QGraphicsScene):
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._current_tool: str = "rect"
        self._current_class_index: int = 0
        self._annotations: list[AnnotationItem] = []
        self._graphics_items: list = []
        self._signal_holder = _SignalHolder()
        self.annotations_changed = self._signal_holder.annotations_changed

        self._drawing: bool = False
        self._start_point: QPointF = QPointF()
        self._temp_rect_item: Optional[QGraphicsRectItem] = None
        self._polygon_points: list[QPointF] = []
        self._temp_polygon_item: Optional[QGraphicsPolygonItem] = None
        self._selected_index: int = -1
        self._class_colors: list[str] = []
        self._class_names: list[str] = []
        self._sam_annotator = None
        self._sam_points: list[QPointF] = []
        self._sam_labels: list[int] = []
        self._temp_sam_items: list = []
        self._sam_encoding: bool = False
        self._keypoint_points: list[QPointF] = []
        self._temp_keypoint_items: list = []
        self._temp_keypoint_lines: list = []
        self._current_kpt_count: int = 0
        self._undo_stack: list = []
        self._redo_stack: list = []

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

    def set_class_names(self, names: list[str]) -> None:
        self._class_names = names

    def _get_color(self, class_index: int) -> str:
        if 0 <= class_index < len(self._class_colors):
            return self._class_colors[class_index]
        return "#FF0000"

    def set_tool(self, tool: str) -> None:
        self._current_tool = tool
        self._cancel_drawing()
        if tool != "sam":
            self.clear_sam_points()
        if tool != "keypoint":
            for item in self._temp_keypoint_items:
                self.removeItem(item)
            self._temp_keypoint_items.clear()
            for line in self._temp_keypoint_lines:
                self.removeItem(line)
            self._temp_keypoint_lines.clear()
            self._keypoint_points.clear()

    def set_current_class(self, index: int) -> None:
        self._current_class_index = index

    def set_sam_annotator(self, annotator) -> None:
        self._sam_annotator = annotator

    def set_kpt_count(self, count: int) -> None:
        self._current_kpt_count = count

    def clear_sam_points(self) -> None:
        self._sam_points.clear()
        self._sam_labels.clear()
        for item in self._temp_sam_items:
            self.removeItem(item)
        self._temp_sam_items.clear()

    def _cancel_drawing(self) -> None:
        self._drawing = False
        if self._temp_rect_item is not None:
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
        if self._temp_polygon_item is not None:
            self.removeItem(self._temp_polygon_item)
            self._temp_polygon_item = None
        self._polygon_points.clear()
        for item in self._temp_keypoint_items:
            self.removeItem(item)
        self._temp_keypoint_items.clear()
        for line in self._temp_keypoint_lines:
            self.removeItem(line)
        self._temp_keypoint_lines.clear()
        self._keypoint_points.clear()
        if self._temp_sam_items:
            for item in self._temp_sam_items:
                self.removeItem(item)
            self._temp_sam_items.clear()
            self._sam_points.clear()
            self._sam_labels.clear()

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

        elif self._current_tool == "sam":
            if self._sam_encoding:
                super().mousePressEvent(event)
                return
            if event.button() == Qt.MouseButton.LeftButton:
                self._sam_points.append(pos)
                self._sam_labels.append(1)
                self._draw_sam_point(pos, True)
            elif event.button() == Qt.MouseButton.RightButton:
                self._sam_points.append(pos)
                self._sam_labels.append(0)
                self._draw_sam_point(pos, False)

        elif self._current_tool == "keypoint":
            if event.button() == Qt.MouseButton.LeftButton:
                if self._current_kpt_count > 0 and len(self._keypoint_points) >= self._current_kpt_count:
                    return
                self._keypoint_points.append(pos)
                self._draw_temp_keypoint(pos, len(self._keypoint_points) - 1)
                if self._current_kpt_count > 0 and len(self._keypoint_points) >= self._current_kpt_count:
                    self._finish_keypoint()

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
                    self._undo_stack.append(("add", len(self._annotations) - 1, ann))
                    self._redo_stack.clear()
                    if len(self._undo_stack) > 50:
                        self._undo_stack.pop(0)
                    self.annotations_changed.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._current_tool == "keypoint" and len(self._keypoint_points) >= 1:
            self._finish_keypoint()
            return
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
            self._undo_stack.append(("add", len(self._annotations) - 1, ann))
            self._redo_stack.clear()
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
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

    def _draw_sam_point(self, pos: QPointF, is_foreground: bool) -> None:
        color = QColor("#00FF00") if is_foreground else QColor("#FF0000")
        item = QGraphicsEllipseItem(pos.x() - 4, pos.y() - 4, 8, 8)
        item.setBrush(QBrush(color))
        item.setPen(QPen(color, 1))
        self.addItem(item)
        self._temp_sam_items.append(item)

    def _draw_temp_keypoint(self, pos: QPointF, index: int) -> None:
        color = QColor(self._get_color(self._current_class_index))
        radius = 5
        ellipse = QGraphicsEllipseItem(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        ellipse.setBrush(QBrush(color))
        ellipse.setPen(QPen(color, 1))
        self.addItem(ellipse)
        self._temp_keypoint_items.append(ellipse)
        text = QGraphicsTextItem(str(index + 1))
        text.setDefaultTextColor(QColor(255, 255, 255))
        font = text.font()
        font.setPointSize(8)
        font.setBold(True)
        text.setFont(font)
        text.setPos(pos.x() - 4, pos.y() - 10)
        self.addItem(text)
        self._temp_keypoint_items.append(text)
        if index > 0:
            prev = self._keypoint_points[index - 1]
            line = self.addLine(prev.x(), prev.y(), pos.x(), pos.y(), QPen(color, 2))
            self._temp_keypoint_lines.append(line)

    def _finish_keypoint(self) -> None:
        if not self._keypoint_points:
            return
        xs = [p.x() for p in self._keypoint_points]
        ys = [p.y() for p in self._keypoint_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin = 5
        rect = QRectF(min_x - margin, min_y - margin, max_x - min_x + margin * 2, max_y - min_y + margin * 2)
        ann = AnnotationItem(
            class_index=self._current_class_index,
            rect=rect,
            item_type="keypoint",
            keypoints=list(self._keypoint_points),
        )
        for item in self._temp_keypoint_items:
            self.removeItem(item)
        for line in self._temp_keypoint_lines:
            self.removeItem(line)
        self._temp_keypoint_items.clear()
        self._temp_keypoint_lines.clear()
        self._keypoint_points.clear()
        self._annotations.append(ann)
        self._draw_annotation(ann, len(self._annotations) - 1)
        self._undo_stack.append(("add", len(self._annotations) - 1, ann))
        self._redo_stack.clear()
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self.annotations_changed.emit()

    def get_sam_input_points(self):
        if not self._sam_points:
            return None, None
        points = [[p.x(), p.y()] for p in self._sam_points]
        labels = list(self._sam_labels)
        self.clear_sam_points()
        return points, labels

    def apply_sam_result(self, masks, scores) -> None:
        import numpy as np
        import cv2
        if masks is None or len(masks) == 0:
            return
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8)
        h, w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        points = []
        for pt in approx:
            points.append(QPointF(float(pt[0][0]), float(pt[0][1])))
        if len(points) >= 3:
            ann = AnnotationItem(
                class_index=self._current_class_index,
                polygon=QPolygonF(points),
                item_type="polygon",
            )
            self._annotations.append(ann)
            self._draw_annotation(ann, len(self._annotations) - 1)
            self._undo_stack.append(("add", len(self._annotations) - 1, ann))
            self._redo_stack.clear()
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self.annotations_changed.emit()

    def _select_at(self, pos: QPointF) -> None:
        old_index = self._selected_index
        self._selected_index = -1
        for i in range(len(self._annotations) - 1, -1, -1):
            ann = self._annotations[i]
            if ann.item_type == "rect" and ann.rect.contains(pos):
                self._selected_index = i
                break
            elif ann.item_type == "polygon" and ann.polygon.containsPoint(pos, Qt.FillRule.OddEvenFill):
                self._selected_index = i
                break
            elif ann.item_type == "keypoint" and ann.rect.contains(pos):
                self._selected_index = i
                break

        if old_index == self._selected_index:
            return

        if old_index != -1 and old_index < len(self._graphics_items) and self._graphics_items[old_index] is not None:
            ann = self._annotations[old_index]
            color = QColor(self._get_color(ann.class_index))
            pen = QPen(color, 2)
            for item in self._graphics_items[old_index]:
                if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem)):
                    item.setPen(pen)

        if self._selected_index != -1 and self._selected_index < len(self._graphics_items) and self._graphics_items[self._selected_index] is not None:
            ann = self._annotations[self._selected_index]
            color = QColor(self._get_color(ann.class_index))
            pen = QPen(color, 3, Qt.PenStyle.DashLine)
            for item in self._graphics_items[self._selected_index]:
                if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem)):
                    item.setPen(pen)

    def _draw_annotation(self, ann: AnnotationItem, index: int) -> None:
        color = QColor(self._get_color(ann.class_index))
        pen = QPen(color, 2)
        if index == self._selected_index:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(3)

        items_added = []

        if ann.item_type == "rect":
            item = QGraphicsRectItem(ann.rect)
            item.setPen(pen)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
            item.setBrush(brush)
            item.setData(0, index)
            self.addItem(item)
            items_added.append(item)
        elif ann.item_type == "polygon":
            item = QGraphicsPolygonItem(ann.polygon)
            item.setPen(pen)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
            item.setBrush(brush)
            item.setData(0, index)
            self.addItem(item)
            items_added.append(item)
        elif ann.item_type == "keypoint":
            item = QGraphicsRectItem(ann.rect)
            item.setPen(pen)
            item.setData(0, index)
            self.addItem(item)
            items_added.append(item)
            for ki, kp in enumerate(ann.keypoints):
                radius = 5
                ellipse = QGraphicsEllipseItem(kp.x() - radius, kp.y() - radius, radius * 2, radius * 2)
                ellipse.setBrush(QBrush(color))
                ellipse.setPen(QPen(color, 1))
                ellipse.setData(0, index)
                self.addItem(ellipse)
                items_added.append(ellipse)
                text = QGraphicsTextItem(str(ki + 1))
                text.setDefaultTextColor(QColor(255, 255, 255))
                font = text.font()
                font.setPointSize(8)
                font.setBold(True)
                text.setFont(font)
                text.setPos(kp.x() - 4, kp.y() - 10)
                text.setData(0, index)
                self.addItem(text)
                items_added.append(text)
                if ki > 0:
                    prev = ann.keypoints[ki - 1]
                    line = self.addLine(prev.x(), prev.y(), kp.x(), kp.y(), QPen(color, 2))
                    line.setData(0, index)
                    items_added.append(line)

        name = self._class_names[ann.class_index] if 0 <= ann.class_index < len(self._class_names) else str(ann.class_index)
        label_text = name
        label = QGraphicsTextItem(label_text)
        label.setDefaultTextColor(color)
        pos_x = ann.rect.left() if ann.item_type == "rect" else ann.polygon.boundingRect().left()
        pos_y = (ann.rect.top() - 20) if ann.item_type == "rect" else (ann.polygon.boundingRect().top() - 20)
        label.setPos(pos_x, pos_y)
        label.setData(0, index)
        self.addItem(label)
        items_added.append(label)

        while len(self._graphics_items) <= index:
            self._graphics_items.append(None)
        self._graphics_items[index] = items_added

    def _remove_annotation_graphics(self, index: int) -> None:
        if 0 <= index < len(self._graphics_items) and self._graphics_items[index] is not None:
            for item in self._graphics_items[index]:
                self.removeItem(item)
            self._graphics_items[index] = None

    def _redraw_all(self) -> None:
        for item in self.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsEllipseItem)):
                self.removeItem(item)
        self._graphics_items.clear()
        for i, ann in enumerate(self._annotations):
            self._draw_annotation(ann, i)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        action_type, index, ann = self._undo_stack.pop()
        if action_type == "add":
            if 0 <= index < len(self._annotations):
                self._annotations.pop(index)
            self._selected_index = -1
            self._redraw_all()
            self.annotations_changed.emit()
            self._redo_stack.append(("delete", index, ann))
        elif action_type == "delete":
            self._annotations.insert(index, ann)
            self._selected_index = -1
            self._redraw_all()
            self.annotations_changed.emit()
            self._redo_stack.append(("add", index, ann))

    def redo(self) -> None:
        if not self._redo_stack:
            return
        action_type, index, ann = self._redo_stack.pop()
        if action_type == "add":
            self._annotations.insert(index, ann)
            self._selected_index = -1
            self._redraw_all()
            self.annotations_changed.emit()
            self._undo_stack.append(("delete", index, ann))
        elif action_type == "delete":
            if 0 <= index < len(self._annotations):
                self._annotations.pop(index)
            self._selected_index = -1
            self._redraw_all()
            self.annotations_changed.emit()
            self._undo_stack.append(("add", index, ann))

    def delete_selected(self) -> None:
        if 0 <= self._selected_index < len(self._annotations):
            deleted_ann = self._annotations[self._selected_index]
            self._undo_stack.append(("delete", self._selected_index, deleted_ann))
            self._redo_stack.clear()
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self._annotations.pop(self._selected_index)
            self._selected_index = -1
            self._redraw_all()
            self.annotations_changed.emit()

    def clear_annotations(self) -> None:
        self._annotations.clear()
        self._graphics_items.clear()
        self._selected_index = -1
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._cancel_drawing()
        for item in self.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsEllipseItem)):
                self.removeItem(item)

    def get_annotations(self) -> list[AnnotationItem]:
        return list(self._annotations)

    def load_annotations(self, annotations: list[AnnotationItem]) -> None:
        self._annotations = list(annotations)
        self._selected_index = -1
        self._cancel_drawing()
        self._redraw_all()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
        elif event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_Return and self._current_tool == "keypoint":
            if len(self._keypoint_points) >= 1:
                self._finish_keypoint()
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
