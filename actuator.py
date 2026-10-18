"""Câmera virtual: janela de recorte movida por um atuador com limites físicos simulados."""
from collections import deque


class VirtualCamera:
    """Janela (win_w x win_h) que se move dentro do quadro (frame_w x frame_h).

    Modelo do atuador, por eixo: comando -> atraso (latency_s) -> lag de 1ª ordem
    (constante de tempo tau_s) -> saturação em max_speed -> integra a posição.
    """

    def __init__(self, frame_w, frame_h, win_w, win_h,
                 max_speed=400.0, latency_s=0.10, tau_s=0.08):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.win_w, self.win_h = win_w, win_h
        self.max_speed, self.latency_s, self.tau_s = max_speed, latency_s, tau_s
        self.reset()

    def reset(self):
        self.x, self.y = self.frame_w / 2, self.frame_h / 2  # centro da janela
        self.vx = self.vy = 0.0
        self._queue = deque()  # (instante, cmd_x, cmd_y)
        self._applied = [0.0, 0.0]  # último comando que já "chegou" ao atuador
        self._t = 0.0

    def step(self, cmd_x, cmd_y, dt):
        self._t += dt
        self._queue.append((self._t, cmd_x, cmd_y))
        while self._queue and self._queue[0][0] <= self._t - self.latency_s:
            _, *self._applied = self._queue.popleft()
        tx, ty = self._applied

        sat = lambda v: max(-self.max_speed, min(self.max_speed, v))
        alpha = dt / (self.tau_s + dt)
        self.vx += alpha * (sat(tx) - self.vx)
        self.vy += alpha * (sat(ty) - self.vy)

        half_w, half_h = self.win_w / 2, self.win_h / 2
        self.x = min(max(self.x + self.vx * dt, half_w), self.frame_w - half_w)
        self.y = min(max(self.y + self.vy * dt, half_h), self.frame_h - half_h)

    def window(self):
        """Retângulo (x1, y1, x2, y2) da janela, em inteiros."""
        x1 = int(round(self.x - self.win_w / 2))
        y1 = int(round(self.y - self.win_h / 2))
        return x1, y1, x1 + self.win_w, y1 + self.win_h
