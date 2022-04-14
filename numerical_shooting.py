import solveODE
import numpy as np
from predator_prey import pred_prey
from scipy.optimize import fsolve


def G(f, u0, t0, T):
    sol = solveODE.solve_ode(f, u0, [t0, T], 'rk4', 0.01)
    return sol[:, -1]


def conditions(f, u0):
    return np.array([f(0, u0)[0]])


def num_shoot(f, U):
    u0 = U[:-1]
    T = U[-1]
    G_sol = G(f, u0, T) - u0
    phase = conditions(f, u0)
    g = np.concatenate((G_sol, phase))
    return g


orbit = fsolve(num_shoot, [0.35, 0.35, 21], pred_prey)
u0 = orbit[:-1]
T = orbit[-1]
print('U0: ', u0)
print('Period: ', T)
