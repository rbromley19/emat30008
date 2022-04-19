import numpy as np


# Function to simulate ODE x. = x
def ode_1(t, x):
    return x


# Function to simulate ODE x.. = -x
def f_s(t, X):
    x = X[0]
    y = X[1]
    xdot = y
    ydot = -x
    return np.array((xdot, ydot))


# Function to simulate analytic solution to f_s
def X_analytic(t, X0):
    c2, c1 = X0
    return [c1 * np.sin(t) + c2 * np.cos(t), c1 * np.cos(t) - c2 * np.sin(t)]


# Function to simulate predator-prey equations
def predator_prey(t, u, b):
    a = 1
    d = 0.1
    b = 0.2
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array((dx, dy))


# Function to simulate Hopf bifurcation normal form
def hopf_bf(t, u, b):
    s = -1
    u1, u2 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


# Function to simulate Hopf bifurcation explicit solution
def hopf_exp(t, theta, b):
    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    return np.array([u1, u2])


# Function to simulate third order system of ODEs (eq. 3 in code testing)
def ode_3(t, u, b, s):
    u1, u2, u3 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    du3 = -u3
    return np.array([du1, du2, du3])


# Function to simulate explicit solution to third order system of ODEs (eq. 3 in code testing)
def ode_3_exp(t, theta, b):
    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    u3 = np.exp(-(t + theta))
    return np.array([u1, u2, u3])


def cubic(x, c):
    return x ** 3 - x + c


def hopf_normal(t, u, b):
    u1, u2 = u
    du1 = b * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 - u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


def hopf_mod(t, u, b):
    u1, u2 = u
    du1 = b * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
    du2 = u1 + b * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2
