import numpy as np
import math


def euler_step(f, x0, t0, h):
    x1 = x0 + h * np.array(f(t0, x0))
    t1 = t0 + h
    return x1, t1


def rk4_step(f, x0, t0, h):
    k1 = np.array(f(t0, x0))
    k2 = np.array(f(t0 + h / 2, x0 + (h / 2) * k1))
    k3 = np.array(f(t0 + h / 2, x0 + (h / 2) * k2))
    k4 = np.array(f(t0 + h, x0 + h * k3))
    x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t1 = t0 + h
    return x1, t1


def solve_to(method, f, x1, t1, t2, dt_max):
    if method == "euler":
        func = euler_step
    elif method == "rk4":
        func = rk4_step

    steps = math.floor((t2 - t1) / dt_max)
    for i in range(steps):
        x1, t1 = func(f, x1, t1, dt_max)
    if t1 != t2:
        h = t2 - t1
        x1, t1 = func(f, x1, t1, h)
    return x1


def solve_ode(f, x0, t, method, dt_max):
    # x = np.zeros(len(t))
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(1, len(t)):
        x[i] = solve_to(method, f, x[i - 1], t[i - 1], t[i], dt_max)
    return x
