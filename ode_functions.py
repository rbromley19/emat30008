import numpy as np


def ode_1(t, x):
    return x


def f_s(t, X):
    x = X[0]
    y = X[1]
    xdot = y
    ydot = -x
    return np.array((xdot, ydot))


def X_analytic(t, X0):
    c2, c1 = X0

    return [c1 * np.sin(t) + c2 * np.cos(t), c1 * np.cos(t) - c2 * np.sin(t)]
