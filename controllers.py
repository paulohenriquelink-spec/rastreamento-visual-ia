"""Controladores de um eixo: recebem o erro (px) e devolvem a velocidade comandada (px/s)."""


class Controller:
    name = "base"

    def reset(self):
        pass

    def step(self, error, dt):
        raise NotImplementedError

    def describe(self):
        """Parâmetros em texto curto, para mostrar no painel."""
        return ""


class BangBang(Controller):
    name = "Liga/desliga"

    def __init__(self, speed=250.0, deadband=15.0):
        self.speed = speed
        self.deadband = deadband

    def step(self, error, dt):
        if abs(error) <= self.deadband:
            return 0.0
        return self.speed if error > 0 else -self.speed

    def describe(self):
        return f"velocidade {self.speed:g} px/s · zona morta {self.deadband:g} px"


class P(Controller):
    name = "P"

    def __init__(self, kp=4.0):
        self.kp = kp

    def step(self, error, dt):
        return self.kp * error

    def describe(self):
        return f"Kp {self.kp:g}"


class PD(Controller):
    name = "PD"

    def __init__(self, kp=4.0, kd=0.4):
        self.kp, self.kd = kp, kd
        self.reset()

    def reset(self):
        self.prev = None

    def step(self, error, dt):
        d = 0.0 if self.prev is None or dt <= 0 else (error - self.prev) / dt
        self.prev = error
        return self.kp * error + self.kd * d

    def describe(self):
        return f"Kp {self.kp:g} · Kd {self.kd:g}"


class PID(Controller):
    name = "PID"

    def __init__(self, kp=4.0, ki=1.0, kd=0.4, i_limit=150.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit  # anti-windup: limita a integral acumulada
        self.reset()

    def reset(self):
        self.prev = None
        self.integral = 0.0

    def step(self, error, dt):
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral + error * dt))
        d = 0.0 if self.prev is None or dt <= 0 else (error - self.prev) / dt
        self.prev = error
        return self.kp * error + self.ki * self.integral + self.kd * d

    def describe(self):
        return f"Kp {self.kp:g} · Ki {self.ki:g} · Kd {self.kd:g} · anti-windup ±{self.i_limit:g}"


def make_all():
    """Uma instância nova de cada controlador, na ordem das teclas 1-4."""
    return [BangBang(), P(), PD(), PID()]
