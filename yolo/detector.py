"""
YOLO-based manga panel detector using panel.pt model.
Detects panel bounding boxes for script synchronization.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Panel:
    """A detected panel bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    area: int = 0
    center_y: int = 0

    def __post_init__(self):
        self.area = (self.x2 - self.x1) * (self.y2 - self.y1)
        self.center_y = (self.y1 + self.y2) // 2


class YoloDetector:
    """YOLO-powered manga panel detector"""

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self._available = False

        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = str(project_root / "panel.pt")

        if not os.path.exists(model_path):
            logger.warning(f"YOLO model not found at {model_path}")
            return

        try:
            from ultralytics import YOLO
            self._patch_nms()
            self.model = YOLO(model_path)
            self._available = True
            logger.info(f"YOLO model loaded from {model_path} (conf={confidence_threshold})")
        except ImportError:
            logger.warning("ultralytics not installed - YOLO detection unavailable")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    @staticmethod
    def _patch_nms():
        """Patch torchvision NMS to move tensors to CPU before calling."""
        try:
            import torchvision
            _original_nms = torchvision.ops.nms

            def _cpu_nms(boxes, scores, iou_threshold):
                return _original_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)

            torchvision.ops.nms = _cpu_nms
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._available

    def detect_panels(self, image: np.ndarray) -> List[Panel]:
        """Run YOLO detection on a single image (BGR numpy array)"""
        if not self._available:
            return []

        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        panels = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                panels.append(Panel(int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf)))
        return panels

    def detect_and_filter(
        self,
        image: np.ndarray,
        iou_threshold: float = 0.3,
        containment_threshold: float = 0.75,
        min_height: int = 400,
    ) -> List[Panel]:
        """Detect panels, drop slivers, remove overlaps, sort top-to-bottom.

        Filters applied (in order):
          1. min_height — drop boxes shorter than this (sliver false positives)
          2. IoU NMS — drop boxes whose IoU with a higher-conf box exceeds iou_threshold
          3. Containment — drop boxes where ≥containment_threshold of the box's area
             is inside a higher-conf box (catches duplicate detections where one box
             is nested inside a much larger box, which IoU misses)
        """
        panels = self.detect_panels(image)
        if not panels:
            return []

        before = len(panels)
        panels = [p for p in panels if (p.y2 - p.y1) >= min_height]
        dropped_sliver = before - len(panels)

        before = len(panels)
        panels = self._remove_overlapping(panels, iou_threshold, containment_threshold)
        dropped_overlap = before - len(panels)

        if dropped_sliver or dropped_overlap:
            logger.info(
                f"YOLO filter: dropped {dropped_sliver} slivers (<{min_height}px tall), "
                f"{dropped_overlap} overlaps (iou>{iou_threshold} or contained>{containment_threshold})"
            )

        panels = sorted(panels, key=lambda p: p.y1)
        return panels

    def _calculate_iou(self, p1: Panel, p2: Panel) -> float:
        x1 = max(p1.x1, p2.x1)
        y1 = max(p1.y1, p2.y1)
        x2 = min(p1.x2, p2.x2)
        y2 = min(p1.y2, p2.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = p1.area + p2.area - intersection
        return intersection / union if union > 0 else 0.0

    def _containment(self, inner: Panel, outer: Panel) -> float:
        """Fraction of `inner` that lies inside `outer` (intersection / inner.area)."""
        x1 = max(inner.x1, outer.x1)
        y1 = max(inner.y1, outer.y1)
        x2 = min(inner.x2, outer.x2)
        y2 = min(inner.y2, outer.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        return intersection / inner.area if inner.area > 0 else 0.0

    def _remove_overlapping(
        self,
        panels: List[Panel],
        iou_threshold: float,
        containment_threshold: float,
    ) -> List[Panel]:
        panels.sort(key=lambda p: p.confidence, reverse=True)
        keep: List[Panel] = []
        while panels:
            current = panels.pop(0)
            keep.append(current)
            panels = [
                p for p in panels
                if self._calculate_iou(current, p) <= iou_threshold
                and self._containment(p, current) < containment_threshold
            ]
        return keep
