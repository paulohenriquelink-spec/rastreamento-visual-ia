"""Detecção da garrafa azul: YOLO acha 'bottle' e o filtro de cor HSV escolhe a azul.

Quando o YOLO perde a garrafa (deitada, cortada pela borda, borrada), a cor assume:
procura a mancha azul perto da última posição conhecida por até max_color_frames quadros.
"""
import cv2
import numpy as np
from ultralytics import YOLO

BOTTLE_CLASS = 39  # índice de 'bottle' no COCO
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def blue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (90, 80, 50), (135, 255, 255))


def blue_fraction(frame, box):
    """Fração de pixels azuis dentro da caixa (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    roi = frame[max(y1, 0):y2, max(x1, 0):x2]
    if roi.size == 0:
        return 0.0
    mask = blue_mask(roi)
    return float(np.count_nonzero(mask)) / mask.shape[0] / mask.shape[1]


class BottleDetector:
    # yolov8s em 480 px: no teste com a garrafa térmica azul o nano errava metade dos
    # quadros; o s acerta quase todos a ~10 fps na CPU. Conf baixa porque o filtro
    # azul já descarta as garrafas erradas.
    def __init__(self, weights="yolov8s.pt", conf=0.15, min_blue=0.08, imgsz=480,
                 max_color_frames=30):
        self.model = YOLO(weights)
        self.conf = conf
        self.min_blue = min_blue
        self.imgsz = imgsz
        self.max_color_frames = max_color_frames
        self.candidates = []  # (box, conf, fração azul) de todas as garrafas do último quadro
        self.source = None    # "yolo", "cor" ou None
        self._last_box = None
        self._color_frames = 0

    def detect(self, frame):
        """Retorna (cx, cy, conf, box) da garrafa azul mais provável, ou None."""
        best = self._detect_yolo(frame)
        if best is not None:
            self.source = "yolo"
            self._color_frames = 0
        elif self._last_box is not None and self._color_frames < self.max_color_frames:
            best = self._detect_color(frame, self._last_box)
            self._color_frames += 1
            self.source = "cor" if best is not None else None
        else:
            self.source = None
        self._last_box = best[3] if best is not None else None
        return best

    def _detect_yolo(self, frame):
        result = self.model.predict(
            frame, classes=[BOTTLE_CLASS], conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        self.candidates = []
        best, best_score = None, 0.0
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            blue = blue_fraction(frame, box)
            self.candidates.append((box, float(conf), blue))
            if blue < self.min_blue:
                continue
            score = float(conf) * blue
            if score > best_score:
                x1, y1, x2, y2 = box
                best = ((x1 + x2) / 2, (y1 + y2) / 2, float(conf), (x1, y1, x2, y2))
                best_score = score
        return best

    def _detect_color(self, frame, box):
        """Maior mancha azul numa região ao redor da última caixa (1 caixa de folga de cada lado)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        rx1, ry1 = int(max(x1 - bw, 0)), int(max(y1 - bh, 0))
        rx2, ry2 = int(min(x2 + bw, w)), int(min(y2 + bh, h))
        mask = cv2.morphologyEx(blue_mask(frame[ry1:ry2, rx1:rx2]), cv2.MORPH_OPEN, _KERNEL)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw2, bh2, area = stats[i]
        if area < max(1500, 0.15 * bw * bh):  # mancha pequena demais: não é a garrafa
            return None
        nx1, ny1 = rx1 + bx, ry1 + by
        nx2, ny2 = nx1 + bw2, ny1 + bh2
        return ((nx1 + nx2) / 2, (ny1 + ny2) / 2, 0.0, (nx1, ny1, nx2, ny2))
