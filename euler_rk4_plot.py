from integrate_ode import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from ode_functions import ode_1, f_s, X_analytic
import math


def error_plot(f, x0, t, true_sol, methods):
    """Generates error plots for 1-step ODE integration functions

f        : function
          Function of given ODE(s) that returns the derivative at f(t, x)
x0       : float
          Initial x-value
t        : list
          List of t-values to be solved for
true_sol : float
          Analytic solution to the function f
methods  : list
          List of methods to be plotted

Returns
-------
Plot of 1-step ODE integrator errors
"""


    h_val = np.logspace(-4, -1, 50)
    for method in methods:
        method_list = np.zeros(int(len(h_val)))
        for i in range(len(h_val)):
            method_sol = solve_ode(f, x0, t, method, h_val[i])
            error = abs(method_sol[-1] - true_sol)
            method_list[i] = error
        ax = plt.gca()
        ax.scatter(h_val, method_list)
    ax.legend(methods, loc='best')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Error')
    ax.set_ylabel('Timestep')
    plt.show()
    plt.savefig('eulererror.png')

# This needs to be moved to future euler test
def euler_run(f, t):
    method = 'euler'
    x0 = [1]
    dt_max = 0.01
    euler = solve_ode(f, x0, t, method, dt_max)[-1]
    print('Euler approximation = ' + str(euler))


# This needs to be moved to future rk4 test
def rk4_run(f, t):
    method = 'rk4'
    x0 = [1]
    dt_max = 0.01
    rk4 = solve_ode(f, x0, t, method, dt_max)[-1]
    print('Runge-kutta approximation = ' + str(rk4))


if __name__ == '__main__':
    f = ode_1
    true_sol = math.e
    t = [0, 1]
    euler_run(f, t)
    rk4_run(f, t)
    error_plot(f, [1], t, true_sol, ['euler', 'rk4'])
