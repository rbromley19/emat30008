from PDE_solver import calc_PDE
from math import pi
import numpy as np
import pylab as pl


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y


def p_func(t):
    return t


def q_func(t):
    return t


def finite_diff_run():
    xfw, u_jfw = calc_PDE(u_I, kappa, L, T, 'fw', None, None)
    xbw, u_jbw = calc_PDE(u_I, kappa, L, T, 'bw', None, None)
    xck, u_jck = calc_PDE(u_I, kappa, L, T, 'ck', None, None)
    # Plot the final result and exact solution
    pl.plot(xfw, u_jfw[1], 'r-o', label='Forward-Euler')
    pl.plot(xbw, u_jbw[1], 'g-o', label='Backward-Euler')
    pl.plot(xck, u_jck[1], 'y-o', label='Crank-Nicholson')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='Exact')
    pl.title('Plot of Finite Difference Methods')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


def dirichlet_run():
    x, u_j = calc_PDE(u_I, kappa, L, T, 'fw', (p_func, q_func), 'dirichlet')
    pl.plot(x, u_j[1], 'r-o', label='Forward Euler with Dirichlet boundary conditions')
    pl.title('Plot of Forward Euler with Dirichlet Boundary Conditions')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


if __name__ == '__main__':
    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

    finite_diff_run()
    dirichlet_run()
