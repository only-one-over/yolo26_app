import copy
import math
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
    QGraphicsPixmapItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneWheelEvent,
)

# 工具类型常量(与现有字符串字面量保持兼容,作为别名使用)
TOOL_SELECT = "select"
TOOL_RECT = "rect"
TOOL_POLYGON = "polygon"
TOOL_KEYPOINT = "keypoint"
TOOL_OBB = "obb"
TOOL_SAM = "sam"


@dataclass
class AnnotationItem:
    class_index: int
    rect: QRectF = field(default_factory=QRectF)
    polygon: QPolygonF = field(default_factory=QPolygonF)
    item_type: str = "rect"
    keypoints: List[QPointF] = field(default_factory=list)
    # OBB 旋转角度(弧度),仅 item_type == "obb" 时使用,默认 0.0 向后兼容 rect
    angle: float = 0.0


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

        # OBB(旋转框)相关状态:第一次拖拽确定外接矩形,第二次拖拽围绕中心旋转
        self._obb_pending_index: int = -1  # 第一次拖拽完成后的待旋转 OBB index
        self._obb_rotating: bool = False
        self._obb_rotation_start_angle: float = 0.0  # 旋转开始时鼠标相对中心的角度
        self._obb_rotation_orig_angle: float = 0.0  # 旋转开始时 OBB 原始角度

        # 多边形顶点编辑相关状态
        self._vertex_handles: list = []  # list[QGraphicsEllipseItem] 顶点句柄
        self._vertex_dragging: bool = False
        self._dragging_vertex_index: int = -1
        self._dragging_ann_index: int = -1
        self._pre_drag_ann: Optional[AnnotationItem] = None  # 拖拽前副本,用于 undo

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
        # 清理 OBB 待旋转状态(切换工具或取消绘制时,待旋转的 OBB 仍保留角度 0)
        self._obb_pending_index = -1
        self._obb_rotating = False
        # 清理顶点句柄
        self._hide_vertex_handles()
        self._vertex_dragging = False
        self._dragging_vertex_index = -1
        self._dragging_ann_index = -1
        self._pre_drag_ann = None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.scenePos()

        # select 工具下:优先检测顶点句柄命中(左键拖拽 / 右键删除)
        if self._current_tool == TOOL_SELECT and self._selected_index >= 0:
            # 右键命中顶点句柄 → 删除该顶点
            if event.button() == Qt.MouseButton.RightButton:
                for h in self._vertex_handles:
                    if h.contains(pos):
                        ann_index = h.data(0)
                        vi = h.data(1)
                        ann = self._annotations[ann_index]
                        if ann.polygon.count() <= 3:
                            # 顶点数不足,不允许删除
                            return
                        old_ann = copy.deepcopy(ann)
                        new_poly = QPolygonF()
                        for i in range(ann.polygon.count()):
                            if i != vi:
                                new_poly.append(ann.polygon.at(i))
                        ann.polygon = new_poly
                        self._undo_stack.append(("modify", ann_index, old_ann, copy.deepcopy(ann)))
                        self._redo_stack.clear()
                        if len(self._undo_stack) > 50:
                            self._undo_stack.pop(0)
                        self._redraw_at(ann_index)
                        self._hide_vertex_handles()
                        self._show_vertex_handles(ann_index)
                        self.annotations_changed.emit()
                        return
            # 左键命中顶点句柄 → 进入顶点拖拽
            elif event.button() == Qt.MouseButton.LeftButton:
                for h in self._vertex_handles:
                    if h.contains(pos):
                        self._vertex_dragging = True
                        self._dragging_ann_index = h.data(0)
                        self._dragging_vertex_index = h.data(1)
                        self._pre_drag_ann = copy.deepcopy(self._annotations[self._dragging_ann_index])
                        return

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

        elif self._current_tool == TOOL_OBB:
            # OBB 两次拖拽:第一次拖外接矩形,第二次在 OBB 内拖拽旋转
            if self._obb_pending_index >= 0 and 0 <= self._obb_pending_index < len(self._annotations):
                ann = self._annotations[self._obb_pending_index]
                # 检查点击是否落在待旋转 OBB 内
                if self._obb_polygon(ann).containsPoint(pos, Qt.FillRule.OddEvenFill):
                    if event.button() == Qt.MouseButton.LeftButton:
                        # 进入旋转模式,记录起点角度与原始角度
                        cx = ann.rect.center().x()
                        cy = ann.rect.center().y()
                        self._obb_rotation_start_angle = math.atan2(pos.y() - cy, pos.x() - cx)
                        self._obb_rotation_orig_angle = ann.angle
                        self._obb_rotating = True
                        return
                # 点击在 OBB 外 → 取消待旋转状态,开始绘制新外接矩形
                self._obb_pending_index = -1
                self._obb_rotating = False

            if event.button() == Qt.MouseButton.LeftButton:
                # 开始绘制外接矩形(同 rect 流程)
                self._drawing = True
                self._start_point = pos
                color = QColor(self._get_color(self._current_class_index))
                pen = QPen(color, 2)
                self._temp_rect_item = QGraphicsRectItem(QRectF(pos, pos))
                self._temp_rect_item.setPen(pen)
                self.addItem(self._temp_rect_item)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.scenePos()
        # 顶点拖拽中:更新被拖拽顶点位置并增量重绘
        if self._vertex_dragging and 0 <= self._dragging_ann_index < len(self._annotations):
            ann = self._annotations[self._dragging_ann_index]
            poly = ann.polygon
            new_poly = QPolygonF()
            for i in range(poly.count()):
                if i == self._dragging_vertex_index:
                    new_poly.append(pos)
                else:
                    new_poly.append(poly.at(i))
            ann.polygon = new_poly
            self._redraw_at(self._dragging_ann_index)
            self._hide_vertex_handles()
            self._show_vertex_handles(self._dragging_ann_index)
            return

        # OBB 旋转中:基于鼠标相对中心的偏移角度更新 ann.angle
        if self._obb_rotating and self._obb_pending_index >= 0 and 0 <= self._obb_pending_index < len(self._annotations):
            ann = self._annotations[self._obb_pending_index]
            cx = ann.rect.center().x()
            cy = ann.rect.center().y()
            cur_angle = math.atan2(pos.y() - cy, pos.x() - cx)
            ann.angle = self._obb_rotation_orig_angle + (cur_angle - self._obb_rotation_start_angle)
            self._redraw_at(self._obb_pending_index)
            return

        if self._current_tool == "rect" and self._drawing and self._temp_rect_item is not None:
            rect = QRectF(self._start_point, event.scenePos()).normalized()
            self._temp_rect_item.setRect(rect)
        elif self._current_tool == TOOL_OBB and self._drawing and self._temp_rect_item is not None:
            rect = QRectF(self._start_point, event.scenePos()).normalized()
            self._temp_rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # 顶点拖拽结束:压入 modify undo 栈
        if self._vertex_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._vertex_dragging = False
            if 0 <= self._dragging_ann_index < len(self._annotations) and self._pre_drag_ann is not None:
                new_ann = self._annotations[self._dragging_ann_index]
                self._undo_stack.append(("modify", self._dragging_ann_index, self._pre_drag_ann, copy.deepcopy(new_ann)))
                self._redo_stack.clear()
                if len(self._undo_stack) > 50:
                    self._undo_stack.pop(0)
            self._pre_drag_ann = None
            self._dragging_ann_index = -1
            self._dragging_vertex_index = -1
            self.annotations_changed.emit()
            return

        # OBB 旋转结束:压入 modify undo 栈,清除待旋转状态
        if self._obb_rotating and event.button() == Qt.MouseButton.LeftButton and self._obb_pending_index >= 0:
            self._obb_rotating = False
            if 0 <= self._obb_pending_index < len(self._annotations):
                ann = self._annotations[self._obb_pending_index]
                # 用 angle=0 的副本作为 old_ann(创建时的初始状态)
                old_ann = copy.deepcopy(ann)
                old_ann.angle = 0.0
                self._undo_stack.append(("modify", self._obb_pending_index, old_ann, copy.deepcopy(ann)))
                self._redo_stack.clear()
                if len(self._undo_stack) > 50:
                    self._undo_stack.pop(0)
                self._redraw_at(self._obb_pending_index)
                self.annotations_changed.emit()
            self._obb_pending_index = -1
            return

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
        elif self._current_tool == TOOL_OBB and self._drawing and event.button() == Qt.MouseButton.LeftButton:
            # 第一次拖拽:完成外接矩形,创建 OBB 标注,进入待旋转状态
            self._drawing = False
            if self._temp_rect_item is not None:
                rect = self._temp_rect_item.rect()
                self.removeItem(self._temp_rect_item)
                self._temp_rect_item = None
                if rect.width() > 2 and rect.height() > 2:
                    ann = AnnotationItem(
                        class_index=self._current_class_index,
                        rect=rect,
                        item_type="obb",
                        angle=0.0,
                    )
                    self._annotations.append(ann)
                    new_index = len(self._annotations) - 1
                    self._draw_annotation(ann, new_index)
                    self._undo_stack.append(("add", new_index, ann))
                    self._redo_stack.clear()
                    if len(self._undo_stack) > 50:
                        self._undo_stack.pop(0)
                    # 进入待旋转状态,等待第二次拖拽
                    self._obb_pending_index = new_index
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
            return
        # select 工具下双击多边形边 → 在最近的边上插入新顶点
        if self._current_tool == TOOL_SELECT and self._selected_index >= 0:
            ann = self._annotations[self._selected_index]
            if ann.item_type == "polygon":
                pos = event.scenePos()
                best_edge_i = -1
                best_dist = float("inf")
                n = ann.polygon.count()
                for i in range(n):
                    p1 = ann.polygon.at(i)
                    p2 = ann.polygon.at((i + 1) % n)
                    dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
                    if dx == 0 and dy == 0:
                        continue
                    t = ((pos.x() - p1.x()) * dx + (pos.y() - p1.y()) * dy) / (dx * dx + dy * dy)
                    t = max(0.0, min(1.0, t))
                    proj_x = p1.x() + t * dx
                    proj_y = p1.y() + t * dy
                    dist = ((pos.x() - proj_x) ** 2 + (pos.y() - proj_y) ** 2) ** 0.5
                    if dist < best_dist and dist < 10:  # 10px 容差
                        best_dist = dist
                        best_edge_i = i
                if best_edge_i >= 0:
                    old_ann = copy.deepcopy(ann)
                    new_poly = QPolygonF()
                    for i in range(n):
                        new_poly.append(ann.polygon.at(i))
                        if i == best_edge_i:
                            new_poly.append(pos)
                    ann.polygon = new_poly
                    self._undo_stack.append(("modify", self._selected_index, old_ann, copy.deepcopy(ann)))
                    self._redo_stack.clear()
                    if len(self._undo_stack) > 50:
                        self._undo_stack.pop(0)
                    self._redraw_at(self._selected_index)
                    self._hide_vertex_handles()
                    self._show_vertex_handles(self._selected_index)
                    self.annotations_changed.emit()
                    event.accept()
                    return
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
            elif ann.item_type == "obb" and self._obb_polygon(ann).containsPoint(pos, Qt.FillRule.OddEvenFill):
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

        # 选中状态变化后,根据新选中是否为 polygon 显示/隐藏顶点句柄
        if self._selected_index != -1:
            ann = self._annotations[self._selected_index]
            if ann.item_type == "polygon":
                self._show_vertex_handles(self._selected_index)
            else:
                self._hide_vertex_handles()
        else:
            self._hide_vertex_handles()

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
        elif ann.item_type == "obb":
            # OBB:用 QPolygonF 表示旋转后的四个角点
            poly = self._obb_polygon(ann)
            item = QGraphicsPolygonItem(poly)
            item.setPen(pen)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
            item.setBrush(brush)
            item.setData(0, index)
            self.addItem(item)
            items_added.append(item)

        name = self._class_names[ann.class_index] if 0 <= ann.class_index < len(self._class_names) else str(ann.class_index)
        label_text = name
        label = QGraphicsTextItem(label_text)
        label.setDefaultTextColor(color)
        if ann.item_type == "rect" or ann.item_type == "obb":
            pos_x = ann.rect.left()
            pos_y = ann.rect.top() - 20
        else:
            pos_x = ann.polygon.boundingRect().left()
            pos_y = ann.polygon.boundingRect().top() - 20
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

    def _obb_polygon(self, ann: AnnotationItem) -> QPolygonF:
        """计算 OBB 旋转后的四个角点构成的多边形(与 _draw_annotation 中绘制逻辑一致)"""
        cx, cy = ann.rect.center().x(), ann.rect.center().y()
        w, h = ann.rect.width(), ann.rect.height()
        angle = ann.angle
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hw, hh = w / 2, h / 2
        # 四个角点(相对中心)经旋转矩阵后加回中心
        corners = [
            QPointF(cx + (-hw * cos_a - -hh * sin_a), cy + (-hw * sin_a + -hh * cos_a)),
            QPointF(cx + (hw * cos_a - -hh * sin_a), cy + (hw * sin_a + -hh * cos_a)),
            QPointF(cx + (hw * cos_a - hh * sin_a), cy + (hw * sin_a + hh * cos_a)),
            QPointF(cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)),
        ]
        return QPolygonF(corners)

    def _show_vertex_handles(self, ann_index: int) -> None:
        """选中多边形时显示其所有顶点句柄"""
        self._hide_vertex_handles()
        if not (0 <= ann_index < len(self._annotations)):
            return
        ann = self._annotations[ann_index]
        if ann.item_type != "polygon":
            return
        color = QColor(self._get_color(ann.class_index))
        for vi in range(ann.polygon.count()):
            pt = ann.polygon.at(vi)
            radius = 5
            ellipse = QGraphicsEllipseItem(pt.x() - radius, pt.y() - radius, radius * 2, radius * 2)
            ellipse.setBrush(QBrush(QColor(255, 255, 255)))
            ellipse.setPen(QPen(color, 2))
            ellipse.setData(0, ann_index)
            ellipse.setData(1, vi)  # 顶点索引
            self.addItem(ellipse)
            self._vertex_handles.append(ellipse)

    def _hide_vertex_handles(self) -> None:
        """移除所有顶点句柄"""
        for h in self._vertex_handles:
            self.removeItem(h)
        self._vertex_handles.clear()

    def _redraw_at(self, index: int) -> None:
        """仅重绘单个 index 的标注(增量优化,避免全场景重绘)"""
        self._remove_annotation_graphics(index)
        if 0 <= index < len(self._annotations):
            self._draw_annotation(self._annotations[index], index)

    def _redraw_all(self) -> None:
        # 遍历 _graphics_items 精确移除标注图元
        for items in self._graphics_items:
            if items is None:
                continue
            for item in items:
                self.removeItem(item)
        self._graphics_items.clear()
        # 隐藏顶点句柄(全场景重绘会重建)
        self._hide_vertex_handles()
        for i, ann in enumerate(self._annotations):
            self._draw_annotation(ann, i)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        action_type = entry[0]
        if action_type == "modify":
            # modify 用 4 元组:("modify", index, old_ann, new_ann)
            index, old_ann, new_ann = entry[1], entry[2], entry[3]
            if 0 <= index < len(self._annotations):
                self._annotations[index] = old_ann
                self._redraw_at(index)
                self._hide_vertex_handles()
                if 0 <= self._selected_index == index and old_ann.item_type == "polygon":
                    self._show_vertex_handles(index)
                self.annotations_changed.emit()
                self._redo_stack.append(("modify", index, old_ann, new_ann))
        else:
            # add/delete 用 3 元组:("add"/"delete", index, ann)
            index, ann = entry[1], entry[2]
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
        entry = self._redo_stack.pop()
        action_type = entry[0]
        if action_type == "modify":
            # modify 用 4 元组:("modify", index, old_ann, new_ann)
            index, old_ann, new_ann = entry[1], entry[2], entry[3]
            if 0 <= index < len(self._annotations):
                self._annotations[index] = new_ann
                self._redraw_at(index)
                self._hide_vertex_handles()
                if 0 <= self._selected_index == index and new_ann.item_type == "polygon":
                    self._show_vertex_handles(index)
                self.annotations_changed.emit()
                self._undo_stack.append(("modify", index, old_ann, new_ann))
        else:
            # add/delete 用 3 元组:("add"/"delete", index, ann)
            index, ann = entry[1], entry[2]
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
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsLineItem)):
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
