import numpy as np
import math


# Function to calculate each step for euler integration method
def euler_step(f, x0, t0, h):
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

    x1 = x0 + h * np.array(f(t0, x0))
    t1 = t0 + h
    return x1, t1


# Function to calculate each step for runge-kutta integration method
def rk4_step(f, x0, t0, h):
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

    k1 = np.array(f(t0, x0))
    k2 = np.array(f(t0 + h / 2, x0 + (h / 2) * k1))
    k3 = np.array(f(t0 + h / 2, x0 + (h / 2) * k2))
    k4 = np.array(f(t0 + h, x0 + h * k3))
    x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t1 = t0 + h
    return x1, t1


# Function to solve ode(s) between x1t1 to x2t2 in steps less than dt_max
def solve_to(method, f, x1, t1, t2, dt_max):
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
        x1, t1 = func(f, x1, t1, dt_max)
    if t1 != t2:
        h = t2 - t1
        x1, t1 = func(f, x1, t1, h)
    return x1


# Function to generate results from integration
def solve_ode(f, x0, t, method, dt_max):
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

    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(1, len(t)):
        x[i] = solve_to(method, f, x[i - 1], t[i - 1], t[i], dt_max)
    return x
