import numpy as np
import math
from ode_functions import ode_1, f_s, X_analytic


# Function to calculate each step for euler integration method
def euler_step(f, x0, t0, h, *args):
    """Calculates one Euler step for a function f given an initial x0 and t0 and step h

Parameters
----------
f  : function
    Function of given ODE(s) that returns the derivative at f(t, x)
x0 : float
    Initial x-value
t0 : float
    Initial t-value
h  : float
    Step size between the initial time and final time

Returns
-------
x0 : float
    Final x-value after a single Euler step
t0 : float
    Final t-value after a single Euler step
"""

    x1 = x0 + h * np.array(f(t0, x0, *args))
    t1 = t0 + h
    return x1, t1


# Function to calculate each step for runge-kutta integration method
def rk4_step(f, x0, t0, h, *args):
    """Calculates one Runge-Kutta (4th order) step for a function f given an initial x0 and t0 and step h

Parameters
----------
f  : function
    Function of given ODE(s) that returns the derivative at f(t, x)
x0 : float
    Initial x-value
t0 : float
    Initial t-value
h  : float
    Step size between the initial time and final time

Returns
-------
x0 : float
    Final x-value after a single Runge-Kutta step
t0 : float
    Final t-value after a single Runge-Kutta step
"""
    k1 = np.array(f(t0, x0, *args))
    k2 = np.array(f(t0 + h / 2, x0 + (h / 2) * k1, *args))
    k3 = np.array(f(t0 + h / 2, x0 + (h / 2) * k2, *args))
    k4 = np.array(f(t0 + h, x0 + h * k3, *args))
    x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t1 = t0 + h
    return x1, t1


# Function to solve ode(s) between x1t1 to x2t2 in steps less than dt_max
def solve_to(method, f, x1, t1, t2, dt_max, *args):
    """Solves ODE from x1, t1 to x2, t2 while less than dt_max

Parameters
----------
method : string
         User's choice of integration method ('euler' or 'rk4')
f      : function
         Function of given ODE(s) that returns the derivative at f(t, x)
x1     : float
         Initial x-value
t1     : float
         Initial t-value
t2     : float
         Final t-value
dt_max : float
         Maximum size of each step

Returns
-------
x1 : float
    Final x-value after integration
"""

    if method == "euler":
        func = euler_step
    elif method == "rk4":
        func = rk4_step

    steps = math.floor((t2 - t1) / dt_max)
    for i in range(steps):
        x1, t1 = func(f, x1, t1, dt_max, *args)
    if t1 != t2:
        h = t2 - t1
        x1, t1 = func(f, x1, t1, h, *args)
    return x1


# Function to generate results from integration
def solve_ode(f, x0, t, method, dt_max, *args):
    """Generates and returns the solution estimates from integration

Parameters
----------
f       : function
         Function of given ODE(s) that returns the derivative at f(t, x)
x0      : float
         Initial x-value
t       : list
        List of t-values to be solved for
method : string
         User's choice of integration method ('euler' or 'rk4')
dt_max : float
         Maximum size of each step

Returns
-------
x : array
    Array of x-values calculated from each t-value
"""
    methods = {'euler', 'rk4'}
    if method in methods:
        x = np.zeros((len(t), len(x0)))
        x[0] = x0
        for i in range(1, len(t)):
                    x[i] = solve_to(method, f, x[i - 1], t[i - 1], t[i], dt_max, *args)
    else:
        raise Exception("Method %s not implemented" % method)
    return x


def rk4_run(f, t):
    method = 'rk4'
    x0 = [1]
    dt_max = 0.01
    rk4 = solve_ode(f, x0, t, method, dt_max)
    print('Runge-kutta approximation = ' + str(rk4))


if __name__ == '__main__':
    f = ode_1
    true_sol = math.e
    t = [0, 1]
    rk4_run(f, t)
