"""Demo ao vivo: a câmera virtual segue a garrafa azul.

Teclas: 1 liga/desliga | 2 P | 3 PD | 4 PID | r grava/para CSV | f tela cheia | q sai
"""
import argparse
import csv
import ctypes
import sys
import time
from pathlib import Path

import cv2

from actuator import VirtualCamera
from controllers import make_all
from dashboard import H as DASH_H, W as DASH_W, Dashboard
from detector import BottleDetector

HERE = Path(__file__).parent
WIN = "Rastreamento da garrafa azul"


def fit_window(w, h, margin=0.9):
    """Tamanho inicial da janela: o painel inteiro, reduzido se não couber na tela."""
    try:
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return w, h
    scale = min(1.0, margin * sw / w, margin * sh / h)
    return int(w * scale), int(h * scale)


def open_camera(index, width, height):
    # DirectShow abre a C920 bem mais rápido no Windows; nos outros sistemas, backend padrão
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {index}. Feche outros programas que "
                         "possam estar usando a webcam (Discord, Teams, navegador, outra demo).")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("A câmera abriu mas não entregou imagem. Ela pode estar em uso por "
                         "outro programa (Discord, Teams, navegador, outra demo).")
    h, w = frame.shape[:2]

    detector = BottleDetector()
    cam = VirtualCamera(w, h, win_w=w // 2, win_h=h // 2)
    ctrl_x, ctrl_y = make_all(), make_all()  # um conjunto de controladores por eixo
    idx = 1  # começa no P

    dash = Dashboard()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN, *fit_window(DASH_W, DASH_H))
    fullscreen = False

    print("Demo rodando. Clique na janela do painel antes de usar as teclas:")
    print("  1 liga/desliga | 2 P | 3 PD | 4 PID | r grava CSV | f tela cheia | q ou Esc sai")

    recording, writer, out_file = False, None, None
    prev = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        dt = max(now - prev, 1e-3)
        prev = now
        fps = 0.9 * fps + 0.1 / dt

        det = detector.detect(frame)
        cx, cy = ctrl_x[idx], ctrl_y[idx]
        if det is not None:
            ox, oy, conf, _ = det
            err_x, err_y = ox - cam.x, oy - cam.y
            cmd_x, cmd_y = cx.step(err_x, dt), cy.step(err_y, dt)
        else:
            ox = oy = conf = None
            err_x = err_y = None
            cx.reset()
            cy.reset()
            cmd_x = cmd_y = 0.0  # perdeu o alvo: mantém a posição
        cam.step(cmd_x, cmd_y, dt)

        if recording:
            writer.writerow([f"{now:.4f}", ox, oy, conf, detector.source,
                             f"{cam.x:.2f}", f"{cam.y:.2f}", cx.name])

        dash.update(now, err_x, err_y)
        cv2.imshow(WIN, dash.render(frame, cam, det, detector.source, ctrl_x, idx, fps, recording))

        key = cv2.waitKey(1) & 0xFF
        ch = chr(key).lower() if key != 255 else ""
        if ch == "q" or key == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break  # q, Esc ou janela fechada no X
        if ch == "f":
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if ch in ("1", "2", "3", "4"):
            idx = int(ch) - 1
            ctrl_x[idx].reset()
            ctrl_y[idx].reset()
            print(f"Controlador: {ctrl_x[idx].name}")
        if ch == "r":
            recording = not recording
            if recording:
                path = HERE / f"trajetoria_{int(time.time())}.csv"
                out_file = open(path, "w", newline="")
                writer = csv.writer(out_file)
                writer.writerow(["t", "obj_x", "obj_y", "conf", "fonte", "cam_x", "cam_y", "controlador"])
                print(f"Gravando em {path.name}")
            elif out_file:
                out_file.close()
                print("Gravação encerrada")

    if out_file and not out_file.closed:
        out_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
