"""Painel da demo: quadro completo, câmera virtual, indicadores e gráfico do erro numa tela só.

O que não muda a cada quadro (títulos, cartões, eixos, legendas, teclas) é desenhado uma vez
numa camada fixa, refeita só quando o controlador muda. Por quadro entram só imagens, linhas
e os poucos textos que variam.
"""
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
TOL_PX = 40        # raio em que o alvo conta como "no centro"
HISTORY_S = 10.0   # janela de tempo do gráfico e da métrica de centralização
GRAPH_MAX = 300    # limite do eixo do gráfico (px)

# cores em BGR (o texto também, ver _draw_texts)
BG = (30, 25, 22)
PANEL = (48, 41, 36)
GRID = (78, 69, 62)
BAND = (70, 85, 70)
TEXT = (240, 240, 240)
MUTED = (165, 158, 152)
DARK = (25, 20, 18)
WINDOW = (0, 215, 255)   # janela da câmera virtual
YOLO = (255, 150, 30)    # detecção pelo YOLO
COLOR = (220, 60, 230)   # detecção pelo plano B (cor)
LOST = (70, 70, 235)
GOOD = (110, 205, 110)
ERR_X = (255, 190, 80)
ERR_Y = (80, 170, 255)

# regiões (x, y, largura, altura)
FULL = (16, 72, 1040, 585)
VCAM = (1072, 72, 512, 288)
GRAPH = (16, 673, 1040, 211)
SIDE_X, SIDE_W = 1072, 512
X0 = SIDE_X + 16  # margem interna do painel lateral

# área de plotagem do gráfico
PX0, PX1 = GRAPH[0] + 60, GRAPH[0] + GRAPH[2] - 16
PY0, PY1 = GRAPH[1] + 40, GRAPH[1] + GRAPH[3] - 28


@lru_cache(maxsize=None)
def _font(size, bold=False):
    try:
        return ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def _graph_y(err):
    mid = (PY0 + PY1) / 2
    return int(mid - np.clip(err, -GRAPH_MAX, GRAPH_MAX) / GRAPH_MAX * (PY1 - PY0) / 2)


class Dashboard:
    def __init__(self):
        self.history = deque()  # (t, err_x, err_y); erros None quando o alvo está perdido
        self._texts = []
        self._static_idx = None
        self._static = None

    def update(self, t, err_x, err_y):
        self.history.append((t, err_x, err_y))
        while self.history and self.history[0][0] < t - HISTORY_S:
            self.history.popleft()

    def centered_ratio(self):
        if not self.history:
            return 0.0
        ok = sum(1 for _, ex, ey in self.history if ex is not None and np.hypot(ex, ey) <= TOL_PX)
        return ok / len(self.history)

    def render(self, frame, cam, det, source, controllers, idx, fps, recording):
        if self._static_idx != idx:
            self._static = self._build_static(controllers, idx)
            self._static_idx = idx
        canvas = self._static.copy()
        self._texts = []
        _, err_x, err_y = self.history[-1] if self.history else (0, None, None)
        centered = err_x is not None and np.hypot(err_x, err_y) <= TOL_PX

        self._text((W - 16, 30), f"{fps:.0f} fps", 18, color=MUTED, anchor="rm")
        if recording:
            cv2.circle(canvas, (W - 190, 30), 8, LOST, -1, cv2.LINE_AA)
            self._text((W - 176, 30), "GRAVANDO", 16, bold=True, color=LOST, anchor="lm")
        self._full_view(canvas, frame, cam, det, source)
        self._virtual_view(canvas, frame, cam, det, centered)
        self._indicators(canvas, det, source, err_x, err_y)
        self._graph_lines(canvas)
        return self._draw_texts(canvas)

    # --- camada fixa ---

    def _build_static(self, controllers, idx):
        canvas = np.full((H, W, 3), BG, np.uint8)
        self._texts = []

        title = "Rastreamento visual da garrafa azul"
        self._text((16, 30), title, 26, bold=True, anchor="lm")
        self._text((16 + _font(26, True).getlength(title) + 18, 32),
                   "YOLOv8s + filtro de cor  →  controle em malha fechada  →  câmera virtual",
                   16, color=MUTED, anchor="lm")

        # controlador
        self._card(canvas, 376, 496, "CONTROLADOR")
        chip_w = (SIDE_W - 32 - 3 * 8) // 4
        for i, c in enumerate(controllers):
            cx = X0 + i * (chip_w + 8)
            active = i == idx
            cv2.rectangle(canvas, (cx, 414), (cx + chip_w, 452), WINDOW if active else GRID, -1)
            self._text((cx + chip_w // 2, 433), f"{i + 1} {c.name}", 14, bold=active,
                       color=DARK if active else MUTED, anchor="mm")
        self._text((X0, 476), controllers[idx].describe(), 16, anchor="lm")

        self._card(canvas, 508, 588, "DETECÇÃO")

        self._card(canvas, 600, 720, "ERRO ATUAL (px)")
        for i, name in enumerate(("X", "Y", "Distância")):
            self._text((X0 + i * 160, 646), name, 14, color=MUTED, anchor="lm")

        self._card(canvas, 732, 830, f"NO CENTRO · raio {TOL_PX} px · últimos {HISTORY_S:.0f} s")
        cv2.rectangle(canvas, (X0 + 110, 782), (SIDE_X + SIDE_W - 16, 798), GRID, -1)

        cv2.rectangle(canvas, (SIDE_X, 842), (SIDE_X + SIDE_W, 884), PANEL, -1)
        self._text((SIDE_X + SIDE_W // 2, 863),
                   "1–4 controlador   ·   R gravar   ·   F tela cheia   ·   Q sair",
                   15, color=MUTED, anchor="mm")

        # gráfico: fundo, legenda, faixa de tolerância e eixos
        x, y, w, h = GRAPH
        cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
        self._text((x + 16, y + 18), f"Erro de rastreamento (px) · últimos {HISTORY_S:.0f} s",
                   16, bold=True, anchor="lm")
        lx = x + w - 330
        for label, color in (("X", ERR_X), ("Y", ERR_Y)):
            cv2.line(canvas, (lx, y + 18), (lx + 22, y + 18), color, 3, cv2.LINE_AA)
            self._text((lx + 30, y + 18), label, 15, color=MUTED, anchor="lm")
            lx += 70
        cv2.rectangle(canvas, (lx, y + 11), (lx + 22, y + 25), BAND, -1)
        self._text((lx + 30, y + 18), f"tolerância ±{TOL_PX} px", 15, color=MUTED, anchor="lm")

        cv2.rectangle(canvas, (PX0, _graph_y(TOL_PX)), (PX1, _graph_y(-TOL_PX)), BAND, -1)
        for v in (-GRAPH_MAX, -GRAPH_MAX // 2, 0, GRAPH_MAX // 2, GRAPH_MAX):
            gy = _graph_y(v)
            cv2.line(canvas, (PX0, gy), (PX1, gy), MUTED if v == 0 else GRID, 1)
            self._text((PX0 - 8, gy), f"{v:+d}" if v else "0", 13, color=MUTED, anchor="rm")
        for frac, label in ((0, f"-{HISTORY_S:.0f} s"), (0.5, f"-{HISTORY_S / 2:.0f} s"), (1, "agora")):
            self._text((int(PX0 + frac * (PX1 - PX0)), PY1 + 14), label, 13, color=MUTED,
                       anchor="lm" if frac == 0 else "rm" if frac == 1 else "mm")

        return self._draw_texts(canvas)

    # --- partes que mudam a cada quadro ---

    def _full_view(self, canvas, frame, cam, det, source):
        x, y, w, h = FULL
        fh, fw = frame.shape[:2]
        view = frame.copy()
        x1, y1, x2, y2 = cam.window()
        center = (int(cam.x), int(cam.y))
        cv2.rectangle(view, (x1, y1), (x2, y2), WINDOW, 3)
        cv2.drawMarker(view, center, WINDOW, cv2.MARKER_CROSS, 28, 3)
        if det is not None:
            ox, oy, _, box = det
            bx1, by1, bx2, by2 = (int(v) for v in box)
            cv2.rectangle(view, (bx1, by1), (bx2, by2), YOLO if source == "yolo" else COLOR, 3)
            cv2.arrowedLine(view, center, (int(ox), int(oy)), TEXT, 2, cv2.LINE_AA, tipLength=0.06)
            cv2.circle(view, (int(ox), int(oy)), 8, LOST, -1, cv2.LINE_AA)
        canvas[y:y + h, x:x + w] = cv2.resize(view, (w, h), interpolation=cv2.INTER_LINEAR)
        self._label(canvas, x, y, f"Quadro completo · {fw}×{fh}")

    def _virtual_view(self, canvas, frame, cam, det, centered):
        x, y, w, h = VCAM
        x1, y1, x2, y2 = cam.window()
        crop = frame[y1:y2, x1:x2]
        s = w / crop.shape[1]
        view = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        c = (w // 2, h // 2)
        cv2.circle(view, c, int(TOL_PX * s), WINDOW, 1, cv2.LINE_AA)
        cv2.drawMarker(view, c, WINDOW, cv2.MARKER_CROSS, 24, 2)
        if det is not None and x1 <= det[0] < x2 and y1 <= det[1] < y2:
            cv2.circle(view, (int((det[0] - x1) * s), int((det[1] - y1) * s)), 7, LOST, -1, cv2.LINE_AA)
        canvas[y:y + h, x:x + w] = view
        cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 1, y + h + 1), GOOD if centered else WINDOW, 2)
        self._label(canvas, x, y, f"Câmera virtual · {x2 - x1}×{y2 - y1}")

    def _indicators(self, canvas, det, source, err_x, err_y):
        if det is None:
            chip, label, fg = LOST, "Alvo perdido", TEXT
        elif source == "yolo":
            chip, label, fg = YOLO, f"YOLO · confiança {det[2]:.2f}", DARK
        else:
            chip, label, fg = COLOR, "Plano B · cor azul", DARK
        cv2.rectangle(canvas, (X0, 540), (X0 + 280, 576), chip, -1)
        self._text((X0 + 140, 558), label, 16, bold=True, color=fg, anchor="mm")

        dist = None if err_x is None else np.hypot(err_x, err_y)
        values = [(err_x, ERR_X, "{:+.0f}"), (err_y, ERR_Y, "{:+.0f}"),
                  (dist, GOOD if dist is not None and dist <= TOL_PX else TEXT, "{:.0f}")]
        for i, (val, color, fmt) in enumerate(values):
            self._text((X0 + i * 160, 682), "—" if val is None else fmt.format(val), 34,
                       bold=True, color=color, anchor="lm")

        ratio = self.centered_ratio()
        self._text((X0, 790), f"{ratio * 100:.0f}%", 34, bold=True, color=GOOD, anchor="lm")
        bx0, bx1 = X0 + 110, SIDE_X + SIDE_W - 16
        cv2.rectangle(canvas, (bx0, 782), (bx0 + int((bx1 - bx0) * ratio), 798), GOOD, -1)

    def _graph_lines(self, canvas):
        if not self.history:
            return
        t_end = self.history[-1][0]
        to_x = lambda t: int(PX0 + (t - (t_end - HISTORY_S)) / HISTORY_S * (PX1 - PX0))
        for axis, color in ((1, ERR_X), (2, ERR_Y)):
            segment = []
            for item in self.history:
                if item[axis] is None:  # alvo perdido: interrompe a linha
                    self._polyline(canvas, segment, color)
                    segment = []
                else:
                    segment.append((to_x(item[0]), _graph_y(item[axis])))
            self._polyline(canvas, segment, color)

    # --- auxiliares ---

    def _card(self, canvas, y1, y2, title):
        cv2.rectangle(canvas, (SIDE_X, y1), (SIDE_X + SIDE_W, y2), PANEL, -1)
        self._text((X0, y1 + 16), title, 13, bold=True, color=MUTED, anchor="lm")

    def _label(self, canvas, x, y, text):
        """Rótulo semitransparente no canto superior esquerdo de uma imagem."""
        w = int(_font(15).getlength(text)) + 20
        roi = canvas[y:y + 28, x:x + w]
        roi[:] = (roi * 0.35 + np.array(DARK) * 0.65).astype(np.uint8)
        self._text((x + 10, y + 14), text, 15, anchor="lm")

    @staticmethod
    def _polyline(canvas, pts, color):
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)

    def _text(self, xy, text, size, bold=False, color=TEXT, anchor="la"):
        self._texts.append((xy, text, size, bold, color, anchor))

    def _draw_texts(self, canvas):
        # cv2.putText não desenha acentos, então o texto sai pelo Pillow com a fonte do Windows.
        # Cada texto é desenhado só no recorte que ocupa, para não converter o painel inteiro.
        # O recorte continua em BGR e as cores também.
        for (x, y), text, size, bold, color, anchor in self._texts:
            font = _font(size, bold)
            left, top, right, bottom = font.getbbox(text, anchor=anchor)
            x0, y0 = max(int(x + left) - 2, 0), max(int(y + top) - 2, 0)
            x1, y1 = min(int(x + right) + 2, W), min(int(y + bottom) + 2, H)
            roi = Image.fromarray(canvas[y0:y1, x0:x1])
            ImageDraw.Draw(roi).text((x - x0, y - y0), text, font=font, fill=color, anchor=anchor)
            canvas[y0:y1, x0:x1] = np.asarray(roi)
        self._texts = []
        return canvas
