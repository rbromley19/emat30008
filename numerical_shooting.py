from solveODE import solve_ode
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# Predator-prey equations function
def pred_prey(t, u):
    a = 1
    d = 0.1
    b = 0.2
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array([dx, dy])


# Function to solve the system of odes f
def G(f, u0, t0, T):
    sol = solve_ode(f, u0, [t0, T], 'rk4', 0.01)
    return sol[-1]


# Function to calculate phase condition using d[var]/dt(0) of a system of odes f
def conditions(f, u0, var):
    return np.array([f(0, u0)[var]])


# Shooting function, implements the equation u0 - F(u0, T)
def num_shoot(U, f, var):
    u0 = U[:-1]
    T = U[-1]
    G_sol = u0 - G(f, u0, 0, T)
    phase = conditions(f, u0, var)
    g = np.concatenate((G_sol, phase))
    return g


# First guess
# u0 = np.array([0.25, 0.35])
# T = [0, 20]
# b = 0.2
# sol = solveODE.solve_ode(pred_prey, u0, T, 'rk4', 0.01, args)
# print(sol[0])
# print(sol[1])

# Function to solve root finding problem to return a limit cycle
def orbit_calc(f, u0, T, var):
    # orbit = fsolve(num_shoot(), [0.25, 0.35, 20], pred_prey)
    orbit = fsolve(lambda U, f: num_shoot(U, f, var), [u0[0], u0[1], T], f)
    u0 = orbit[:-1]
    T = orbit[-1]
    print(u0)
    print(T)
    # return u0, T
    plot_shoot(pred_prey, u0, T)


# Function to plot the phase portrait of the orbit
def plot_shoot(f, u0, T):
    t = np.linspace(0, T, 101)
    sol = solve_ode(f, u0, t, 'rk4', 0.01)
    print(sol)
    x = sol[:, 0]
    y = sol[:, 1]
    plt.plot(x, y)
    plt.show()


orbit_calc(pred_prey, [0.25, 0.35], 21, 1)
