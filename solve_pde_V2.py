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


def calc_PDE(u_I, kappa, L, T, method):
    x, lmbda, mx, mt = environment(L, T, kappa)
    u_j, u_jp1 = sol_vars(x)
    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i])
    u_j = solve_PDE(lmbda, mx, mt, u_j, u_jp1, method)
    plot(x, u_j, L, T)


def environment(L, T, kappa):
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

def solve_PDE(lmbda, mx, mt, u_j, u_jp1, method):
    # Solve the PDE: loop over all time points
    for j in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        # Check input method is valid, execute if so
        methods = {'fw': fw, 'bw': bw, 'ck': ck}
        method_name = str(method)
        if method_name in methods:
            methods[method_name](lmbda, mx, u_j, u_jp1)
        else:
            raise Exception("Method %s not implemented" % method_name)

        # Boundary conditions
        u_jp1[0] = 0;
        u_jp1[mx] = 0

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]
    return u_j



def fw(lmbda, mx, u_j, u_jp1):
    diagonals = [[lmbda] * (mx - 1), [1 - 2 * lmbda] * mx, [lmbda] * (mx - 1)]
    A_FW = scipy.sparse.diags(diagonals, [-1, 0, 1]).toarray()
    u_jp1[1:] = np.dot(A_FW, u_j[1:])
    return u_jp1[1:]


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
    pl.plot(x, u_j, 'ro', label='num')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for

# solve_PDE(u_I, 1, 1, 0.5, 10, 1000, 'ck')
calc_PDE(u_I, kappa, L, T, 'fw')
