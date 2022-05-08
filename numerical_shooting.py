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
    if callable(f):
        orbit = fsolve(lambda U, f: num_shoot(U, f, var, *args), u0, f)
    else:
        raise Exception("Input function %f is not a function" % f)
    return orbit


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
    if isinstance(U, np.ndarray):
        u0 = U[:-1]
        T = U[-1]
    else:
        raise Exception("Initial conditions %U must be a list" % U)
    if isinstance(var, int):
        phase = conditions(f, u0, var, *args)
    else:
        raise Exception("Phase variable %var must be an integer" % var)
    integ_sol = solve_ode(f, u0, [0, T], 'rk4', 0.01, *args)
    # print(integ_sol)
    diff = (u0 - integ_sol[:, -1])
    g = np.append((diff), phase)
    # print(g)
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
