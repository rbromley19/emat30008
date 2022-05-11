import numpy as np


# Function to simulate ODE x. = x
def ode_1(t, x):
    """Function to simulate the ODE x' = x

Parameters
----------
t  : float
     Time
x  : array
     x-value at time t

Returns
-------
x : ODE simulation
"""

    return x


# Function to simulate ODE x.. = -x
def f_s(t, X):
    """Function to simulate the ODE x.. = -x

Parameters
----------
t  : float
     Time
x  : array
     x-value at time t

Returns
-------
Array of dx/dt and dy/dt
"""

    x = X[0]
    y = X[1]
    xdot = y
    ydot = -x
    return np.array((xdot, ydot))


# Function to simulate analytic solution to f_s
def x_analytic(t, X0):
    """Function to simulate the analytic solution to f_s

Parameters
----------
t  : float
     Time
X0  : array
     Initial x conditions

Returns
-------
Array of the analytic solution to f_s
"""

    a, b = X0
    return [a * np.sin(t) + b * np.cos(t), a * np.cos(t) - b * np.sin(t)]


# Function to simulate predator-prey equations
def predator_prey(t, u, b=0.26):
    """Function to simulate the predator-prey (Lokterra-Volterra Equations)

Parameters
----------
t  : float
     Time
u  : array
     x-value at time t
b  : float
     value of constant 'b'
Returns
-------
Array of dx/dt and dy/dt
"""
    a = 1
    d = 0.1
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array((dx, dy))


# Function to simulate Hopf bifurcation normal form
def hopf_bf(t, u, b=1):
    """Function to simulate the Hopf bifurcation normal form

Parameters
----------
t  : float
     Time
u  : array
     x-value at time t
b  : float
     value of constant 'b'
Returns
-------
Array of du1/dt and du2/dt
"""

    s = -1
    u1, u2 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


# Function to simulate Hopf bifurcation explicit solution
def hopf_exp(t, theta, b):
    """Function to simulate the Hopf bifurcation explicit solution

Parameters
----------
t      : float
         Time
theta  : float
         phase
b      : float
         value of constant 'b'
Returns
-------
Array of u1 and u2
"""

    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    return np.array([u1, u2])


# Function to simulate third order system of ODEs (eq. 3 in code testing)
def ode_3(t, u, b=1):
    """Function to simulate the third order system of ODEs (eq. 3 in code testing)

Parameters
----------
t  : float
     Time
u  : array
     x-value at time t
b  : float
     value of constant 'b'
s  : float
     value of constant 'b'

Returns
-------
Array of du1/dt, du2/dt and du3/dt
"""
    s = -1
    u1, u2, u3 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    du3 = -u3
    return np.array([du1, du2, du3])


# Function to simulate explicit solution to third order system of ODEs (eq. 3 in code testing)
def ode_3_exp(t, theta, b):
    """Function to simulate the explicit solution to ode_3

Parameters
----------
t      : float
         Time
theta  : float
         phase
b      : float
        value of constant 'b'

Returns
-------
Array of du1/dt, du2/dt and du3/dt
"""

    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    u3 = np.exp(-(t + theta))
    return np.array([u1, u2, u3])


def cubic(x, c):
    """Function to simulate the cubic equation

Parameters
----------
x  : arrat
     value of variable 'x'
c  : float
     value of constant 'c'

Returns
-------
Cubic equation
"""

    return x ** 3 - x + c


def hopf_normal(t, u, b):
    """Function to simulate the Hopf normal form

Parameters
----------
t  : float
     Time
u  : array
     x-value at time t
b  : float
     value of constant 'b'

Returns
-------
Array of du1/dt and du2/dt
"""

    u1, u2 = u
    du1 = b * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 - u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


def hopf_mod(t, u, b):
    """Function to simulate the modified Hopf

Parameters
----------
t  : float
     Time
u  : array
     x-value at time t
b  : float
     value of constant 'b'

Returns
-------
"""

    u1, u2 = u
    du1 = b * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
    du2 = u1 + b * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2
    return np.array([du1, du2])






