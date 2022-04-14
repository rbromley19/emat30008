import solveODE
import numpy as np
from scipy.optimize import fsolve


# T = [t0, T]
def pred_prey(t, u):
    a = 1
    d = 0.1
    b = 0.2
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array([dx, dy])


def G(f, u0, t0, T):
    sol = solveODE.solve_ode(f, u0, [t0, T], 'rk4', 0.01)
    return sol[-1]


def conditions(f, u0):
    return np.array([f(0, u0)[0]])


def num_shoot(U, f):
    u0 = U[:-1]
    T = U[-1]
    G_sol = u0 - G(f, u0, 0, T)
    phase = conditions(f, u0)
    g = np.concatenate((G_sol, phase))
    return g


# First guess
# u0 = np.array([0.25, 0.35])
# T = [0, 20]
# b = 0.2
# sol = solveODE.solve_ode(pred_prey, u0, T, 'rk4', 0.01, args)
# print(sol[0])
# print(sol[1])

orbit = fsolve(num_shoot, [0.35, 0.35, 21], pred_prey)
u0 = orbit[:-1]
T = orbit[-1]
print(u0)
print(T)

t = np.linspace(0, T, 101)
orbit_sol = solveODE.solve_ode(pred_prey, u0, t, 'rk4', 0.001)
x = orbit_sol[:, 0]
y = orbit_sol[:, 1]
print(x)
print(y)
