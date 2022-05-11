from integrate_ode import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from ode_functions import ode_1, f_s, x_analytic
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
            method_sol = np.ndarray.tolist(method_sol)
            method_sol_flat = []
            for sublist in method_sol:
                for item in sublist:
                    method_sol_flat.append(item)
            method_sol = method_sol_flat
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


def sol_plot(f, methods):
    x0 = [1]
    t = np.linspace(0, 1, 10)
    dt_max = 0.01
    for method in methods:
        if method == 'analytic':
            sol = np.exp(t)
        else:
            sol = list(solve_ode(f, x0, t, method, dt_max).flat)
        plt.plot(t, sol)
    plt.legend(methods, loc='best')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.show()


def system_sol_plot(f, methods):
    x0 = [1, 0]
    t = np.linspace(0, 50, 1000)
    dt_max = 0.01
    sol = list(solve_ode(f, x0, t, 'rk4', dt_max).flat)
    print(sol)
    print(sol[0])
    print('--------------------------------------------------')
    print(sol[1])
    # xdot = sol[:,1]
    # plt.plot(xdot, x)
    # plt.show()


# This needs to be moved to future euler test
def euler_run(f, t):
    method = 'euler'
    x0 = [1]
    dt_max = 0.01
    euler = solve_ode(f, x0, t, method, dt_max)[-1]


# This needs to be moved to future rk4 test
def rk4_run(f, t):
    method = 'rk4'
    x0 = [1]
    dt_max = 0.01
    rk4 = solve_ode(f, x0, t, method, dt_max)[-1]


if __name__ == '__main__':
    f = ode_1
    true_sol = math.e
    t = [0, 1]
    error_plot(f, [1], t, true_sol, ['euler', 'rk4'])
    sol_plot(f, ['euler', 'rk4', 'analytic'])
    system_sol_plot(f_s, ['euler', 'rk4'])
