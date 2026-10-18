# Rastreamento visual com IA: câmera que segue a garrafa azul

MVP da disciplina de Inteligência Artificial, por **Paulo Link**.

Uma IA de visão (YOLOv8) encontra uma garrafa azul na imagem da webcam, e um controlador em malha fechada move uma **câmera virtual** para manter a garrafa no centro. O atuador é simulado com limites físicos (atraso, inércia e velocidade máxima), para o mesmo código controlar depois um gimbal ou o drone de bancada do TCC.

## Como rodar (Windows)

1. Instale o **Python 3.12** ([python.org](https://www.python.org/downloads/) ou `winget install Python.Python.3.12`) e o **Git**.
2. Baixe o projeto:
   ```
   git clone https://github.com/paulohenriquelink-spec/rastreamento-visual-ia.git
   cd rastreamento-visual-ia
   ```
3. Crie o ambiente virtual: `python -m venv .venv`
4. Instale as dependências (leva alguns minutos): `.\.venv\Scripts\python -m pip install -r requirements.txt`
5. Conecte a webcam e rode: `.\rodar_demo.bat` (ou dê dois cliques no arquivo).

Os comandos funcionam no PowerShell e no Prompt de Comando. Na primeira execução, os pesos do YOLOv8s (~22 MB) são baixados automaticamente. Em Linux ou macOS, troque `.\.venv\Scripts\python` por `.venv/bin/python` e, no passo 5, rode `.venv/bin/python demo.py`.

**Teclas** (com a janela do painel em foco): `1` liga/desliga · `2` P · `3` PD · `4` PID · `R` grava a trajetória em CSV · `F` tela cheia · `Q` sai.

Opções: `--camera 1` usa outra webcam; `--width` e `--height` mudam a resolução (padrão 1280×720).

## Como funciona

```
Webcam ──► Detecção (IA) ──► Erro (px) ──► Controlador ──► Atuador simulado ──► Câmera virtual
 1280×720   YOLOv8s + cor    alvo − centro  P / PD / PID     atraso, inércia,      recorte 640×360
                                            liga/desliga     vel. máxima              │
              ▲                                                                       │
              └──────────────────────────── malha fechada ◄───────────────────────────┘
```

| Arquivo | Função |
|---|---|
| [detector.py](detector.py) | YOLOv8s detecta garrafas (classe COCO `bottle`), e um filtro HSV escolhe a azul. Se o YOLO perder a garrafa, o filtro de cor procura perto da última posição por até 30 quadros. |
| [controllers.py](controllers.py) | Liga/desliga, P, PD e PID (com anti-windup), um por eixo. |
| [actuator.py](actuator.py) | Câmera virtual: atraso de 100 ms, inércia de 1ª ordem (τ = 80 ms) e velocidade máxima de 400 px/s. |
| [dashboard.py](dashboard.py) | Painel ao vivo: quadro completo, câmera virtual, erro, % de tempo centralizado e gráfico. |
| [demo.py](demo.py) | Laço principal: câmera → detecção → controle → atuador → painel. |

## Resultados dos testes

Os testes foram feitos com uma garrafa térmica azul, na webcam Logitech C920 e em uma CPU sem GPU.

| Versão do detector | Quadros com a garrafa detectada | Velocidade |
|---|---|---|
| YOLOv8n | 51% | ~13 fps |
| YOLOv8s | 80% | ~10 fps |
| YOLOv8s + recuperação por cor | recuperou os casos de garrafa deitada e cortada na borda | ~9 fps com o painel |

Nos testes, o filtro de cor nunca confundiu a garrafa com outro objeto. Com o controlador P, o erro mediano ficou em 47 px, descontando as posições perto da borda, que a janela não alcança.

## Próximos passos

- Comparar os controladores sobre a mesma trajetória gravada (tecla `R`).
- Fazer fine-tuning do YOLO com fotos do objeto.
- Trocar o atuador simulado por servos (pan/tilt) e, no TCC, pelo drone de bancada.
