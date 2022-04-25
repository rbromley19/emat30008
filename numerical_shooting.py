from integrate_ode import solve_ode
import numpy as np
from scipy.optimize import fsolve
from ode_functions import predator_prey
import matplotlib.pyplot as plt


def orbit_calc(f, u0, T, var):
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

    U0 = np.append(u0, T)
    orbit = fsolve(lambda U, f: num_shoot(U, f, var), U0, f)
    T = orbit[-1]
    u0 = orbit[:-1]
    return u0, T


# Shooting function, implements the equation u0 - F(u0, T)
def num_shoot(U, f, var):
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
    G_sol = abs(u0 - G(f, u0, 0, T))
    phase = conditions(f, u0, var)
    g = np.append((G_sol), phase)
    return g


# Function to solve the system of odes f
def G(f, u0, t0, T):
    """Solves the ODE using the 4th order Runge-kutta method

Parameters
----------
f   : function
     Function of given ODE(s) that returns the derivative at f(t, x)
u0  : array
     ODE initial conditions
t0  : float
     ODE initial time variabale
T   : integer
     Time period

Returns
-------
sol[-1] : array
         Solution of ODE
"""

    sol = solve_ode(f, u0, [t0, T], 'rk4', 0.01)
    return np.array(sol[-1])


# Function to calculate phase condition using d[var]/dt(0) of a system of odes f
def conditions(f, u0, var):
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

    phase_cond = np.array([f(0, u0)[var]])
    return phase_cond


if __name__ == '__main__':
    u0 = [0.2, 0.3]
    T = 22
    var = 0
    u0, T = orbit_calc(lambda t, u: predator_prey(t, u, b=0.2), u0, T, var)
    # print(u0, T)
    # u0 = [0.2, 0.3]
    # T = 22
    # var = 0
    # sol = orbit_calc(lambda t, u: predator_prey(t, u, b=0.25), u0, T, var)
    print(u0, T)
    t = np.linspace(0, T, 101)
    sol = solve_ode(lambda t, u: predator_prey(t, u, b=0.25), u0, t, 'rk4', 0.01)
    t = np.linspace(0, T, 101)
    print(sol)
    x = sol[:, 0]
    y = sol[:, 1]
    plt.plot(x, y)
    plt.show()
