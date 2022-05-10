import numpy as np
import matplotlib.pyplot as plt
from integrate_ode import solve_ode


def predator_prey_below(t, u, b=0.22):
    """Function to simulate the predator-prey (Lokterra-Volterra Equations) with b<0.26

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


def predator_prey_above(t, u, b=0.3):
    """Function to simulate the predator-prey (Lokterra-Volterra Equations) with b>0.26

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


def simulate_plot(f):
    t = np.linspace(0, 200, 1000)
    x0 = [0.24, 0.24]
    sol = solve_ode(f, x0, t, 'rk4', 0.01, array=True)
    plt.plot(t, sol[0], label='x(t)')
    plt.plot(t, sol[1], label='y(t)')
    plt.xlabel('t')
    plt.ylabel('x(t) and y(t)')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # Simulate Predator-Prey for b<0.26
    simulate_plot(lambda t, u: predator_prey_below(t, u))
    # Simulate Predator-Prey for b>0.26
    simulate_plot(lambda t, u: predator_prey_above(t, u))
