from integrate_ode import solve_ode
import numpy as np
from scipy.optimize import fsolve
from ode_functions import predator_prey
import matplotlib.pyplot as plt


def orbit_calc(f, u0, var, *args):
    """Calculates the periodic orbit of ODEs using numerical shooting using fsolve

Parameters
----------
f   : function
     Function of given ODE(s) that returns the derivative at f(t, x)
u0  : array
     ODE initial conditions estimate
T   : float
     Time-period initial estimate
var : integer
     User specified choice of variable for phase condition

Returns
-------
u0 : array
    Initial conditions of orbis
T  : float
    Time period of orbit
"""

    # U0 = np.append(u0, T)
    orbit = fsolve(lambda U, f: num_shoot(U, f, var, *args), u0, f)
    return orbit


# Shooting function, implements the equation u0 - F(u0, T)
def num_shoot(U, f, var, *args):
    """Implements numerical shooting by calculating u0 - G(u0, T) and the phase conditions

Parameters
----------
U   : array
     Estimates for initial position and time period
f   : function
     Function of given ODE(s) that returns the derivative at f(t, x)
var : integer
     User specified choice of variable for phase condition

Returns
-------
g : array
    System of equations of the boundary value problem to be solved
"""
    u0 = U[:-1]
    T = U[-1]
    phase = conditions(f, u0, var, *args)
    integ_sol = solve_ode(f, u0, [0, T], 'rk4', 0.01, *args)
    diff = abs(u0 - integ_sol[-1])
    g = np.append((diff), phase)
    return g


# Function to calculate phase condition using d[var]/dt(0) of a system of odes f
def conditions(f, u0, var, *args):
    """Calculates the phase condition of the ODEs using f(0, u0)[var]

Parameters
----------
f   : function
     Function of given ODE(s) that returns the derivative at f(t, x)
u0  : array
     ODE initial conditions
var : integer
     User specified choice of variable for phase condition

Returns
-------
phase_cond : array
            Phase condition for shooting
"""
    phase_cond = np.array([f(0, u0, *args)[var]])
    return phase_cond


if __name__ == '__main__':
    u0 = [0.2, 0.3]
    T = 22
    var = 0
    # u0, T = orbit_calc(lambda t, u: predator_prey(t, u, b=0.2), u0, T, var)
    sol = orbit_calc(predator_prey, u0, T, var)
    print(sol)
    u0 = sol[:-1]
    T = sol[-1]
    # print(u0, T)
    # u0 = [0.2, 0.3]
    # T = 22
    # var = 0
    # sol = orbit_calc(lambda t, u: predator_prey(t, u, b=0.25), u0, T, var)
    t = np.linspace(0, T, 101)
    sol = solve_ode(predator_prey, u0, t, 'rk4', 0.01)
    t = np.linspace(0, T, 101)
    x = sol[:, 0]
    y = sol[:, 1]
    plt.plot(x, y)
    plt.show()
