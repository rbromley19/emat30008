from integrate_ode import solve_ode
import numpy as np
from scipy.optimize import fsolve
from ode_functions import predator_prey


def orbit_calc(f, u0, T, var):
    """Calculates the periodic orbit of ODEs using numerical shooting

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
    Initial conditions of orbit
T  : float
    Time period of orbit
"""

    shoot = lambda U, f: num_shoot(U, f, var)
    orbit = fsolve(shoot, [u0[0], u0[1], T], f)
    print(orbit)
    u0 = orbit[:-1]
    T = orbit[-1]
    print(u0)
    print(T)
    return u0, T


# Shooting function, implements the equation u0 - F(u0, T)
def num_shoot(U, f, var):
    u0 = U[:-1]
    T = U[-1]
    G_sol = u0 - G(f, u0, 0, T)
    phase = conditions(f, u0, var)
    g = np.concatenate((G_sol, phase))
    return g


# Function to solve the system of odes f
def G(f, u0, t0, T):
    sol = solve_ode(f, u0, [t0, T], 'rk4', 0.01)
    return sol[-1]


# Function to calculate phase condition using d[var]/dt(0) of a system of odes f
def conditions(f, u0, var):
    return np.array([f(0, u0)[var]])


if __name__ == '__main__':
    u0 = [0.2, 0.3]
    T = 20
    var = 0
    orbit_calc(lambda t, u: predator_prey(t, u, b=0.2), u0, T, var)
