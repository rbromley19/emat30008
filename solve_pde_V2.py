# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi
import scipy
from scipy.sparse.linalg import spsolve


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y


def calc_PDE(u_I, kappa, L, T, method, bound_funcs, bc):
    x, lmbda, mx, mt = init_params(L, T, kappa)
    u_j, u_jp1 = sol_vars(x)

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i])
    u_j[0] = 0
    u_j[mx] = 0
    u_j = solve_PDE(lmbda, mx, u_j, mt, method, bound_funcs, x, bc)
    print(u_j)
    plot(x, u_j, L, T)


def init_params(L, T, kappa):
    # Set numerical parameters
    mx = 10  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    print("deltax=", deltax)
    print("deltat=", deltat)
    print("lambda=", lmbda)
    return x, lmbda, mx, mt


def sol_vars(x):
    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step
    return u_j, u_jp1


def solve_PDE(lmbda, mx, u_j, mt, method, bound_funcs, x, bc):
    # Solve the PDE: loop over all time points
    methods = {'fw': fw, 'bw': bw, 'ck': ck}
    method_name = str(method)
    if method_name in methods:
        u_j = methods[method_name](lmbda, mx, u_j, mt, bound_funcs, bc)
    else:
        raise Exception("Method %s not implemented" % method_name)

    print(x, u_j)
    return x, u_j


def fw(lmbda, mx, u_j, mt, bound_funcs, bc):
    # diagonals = [[lmbda] * (mx - 1), [1 - 2 * lmbda] * mx, [lmbda] * (mx - 1)]
    # A_FW = scipy.sparse.diags(diagonals, [-1, 0, 1]).toarray()
    n = np.around(mx - 1)
    diagonals = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    A_FW = scipy.sparse.diags(diagonals, [-1, 0, 1]).toarray()
    for i in range(mt):
        if bc == 'zero':
            u_j = np.dot(A_FW, u_j[1:])
            # u_jp1[1:] = np.dot(A_FW, u_j[1:])
        u_j0 = u_j[1:mx]
        if bc == 'dirichlet':
            u_j = calc_diri(u_j0, A_FW, lmbda, mt, bound_funcs)

    return u_j


def calc_diri(u_j0, A_FW, lmbda, mt, bound_funcs):
    sol_vec = np.zeros(len(u_j0))
    p, q = bound_funcs
    sol_vec[0] = p(mt)
    sol_vec[-1] = q(mt)
    sol = A_FW.dot(u_j0) + lmbda * np.array(sol_vec)
    # Add boundary values
    u_j = [p(mt)]
    for i in sol:
        u_j.append(i)
    u_j.append(q(mt))
    return u_j


def bw(lmbda, mx, u_j, u_jp1):
    diagonals = [[- lmbda] * (mx - 1), [1 + 2 * lmbda] * mx, [- lmbda] * (mx - 1)]
    A_BW = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csc')
    u_jp1[1:] = spsolve(A_BW, u_j[1:])
    return u_jp1[1:]


def ck(lmbda, mx, u_j, u_jp1):
    diagonals = [[-lmbda / 2] * (mx - 1), [1 + lmbda] * mx, [-lmbda / 2] * (mx - 1)]
    A_CK = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csc')
    diagonals = [[lmbda / 2] * (mx - 1), [1 - lmbda] * mx, [lmbda / 2] * (mx - 1)]
    B_CK = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csc')
    u_jp1[1:] = spsolve(A_CK, B_CK * u_j[1:])
    return u_jp1[1:]


def plot(x, u_j, L, T):
    # Plot the final result and exact solution
    print(u_j)
    pl.plot(x, u_j[1], 'ro', label='num')
    xx = np.linspace(0, L, 250)
    # pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


def p_func(t):
    return t


def q_func(t):
    return t


# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for

calc_PDE(u_I, kappa, L, T, 'fw', (p_func, q_func), 'dirichlet')
