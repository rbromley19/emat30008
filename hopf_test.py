import math
import numpy as np


def hopf_bf(u, t, b):
    s = -1
    u1, u2 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


def hopf_exp(t, theta, b):
    u1 = math.sqrt(b) * math.cos(t + theta)
    u2 = math.sqrt(b) * math.sin(t + theta)
    return np.array([u1, u2])
