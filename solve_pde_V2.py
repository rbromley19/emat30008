import numpy as np
import scipy
from scipy.sparse.linalg import spsolve


def calc_PDE(u_I, kappa, L, T, method, bound, bc):
    x, lmbda, mx, mt = init_params(L, T, kappa)
    u_j, u_jp1 = sol_vars(x)

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i])
    u_j[0] = 0
    u_j[mx] = 0
    u_j = solve_PDE(lmbda, mx, u_j, mt, method, bound, x, bc)
    return x, u_j


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


def solve_PDE(lmbda, mx, u_j, mt, method, bound, x, bc):
    # Solve the PDE: loop over all time points
    methods = {'fw': fw, 'bw': bw, 'ck': ck}
    method_name = str(method)
    if method_name in methods:
        if method_name == 'fw':
            u_j = methods[method_name](lmbda, mx, u_j, mt, bound, bc)
        else:
            u_j = methods[method_name](lmbda, mx, mt, u_j)
    else:
        raise Exception("Method %s not implemented" % method_name)
    return x, u_j


def fw(lmbda, mx, u_j, mt, bound, bc):
    n = np.around(mx - 1)
    diagonals = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)],
                         dtype=np.dtype(object))
    A_FW = scipy.sparse.diags(diagonals, [-1, 0, 1]).toarray()
    for i in range(mt):
        bc_list = {'dirichlet': dirichlet, 'neumann': neumann}
        bc_name = str(bc)
        if bc_name in bc_list:
            u_j0 = u_j[1:mx]
            u_j = bc_list[bc_name](lmbda, u_j0, mt, bound, A_FW)
        elif bc is None:
            u_j0 = u_j[1:mx]
            u_j = none(u_j0, A_FW)
        else:
            return Exception('Boundary condition %s not implemented. Select a valid boundary condition.' % bc)
    return u_j


def dirichlet(lmbda, u_j0, mt, bound, A_FW):
    sol_vec = np.array(np.zeros(len(u_j0)))
    p, q = bound
    sol_vec[0] = p(mt)
    sol_vec[-1] = q(mt)
    sol_arr = A_FW.dot(u_j0) + lmbda * sol_vec
    u_j = [p(mt)]
    for i in sol_arr:
        u_j.append(i)
    u_j.append(q(mt))
    return u_j


def none(u_j0, A_FW):
    u_j = [0]
    sol_arr = A_FW.dot(u_j0)
    for i in sol_arr:
        u_j.append(i)
    u_j.append(0)
    return u_j


def neumann():
    pass


def bw(lmbda, mx, mt, u_j):
    n = np.around(mx - 1)
    diagonals = np.array([-lmbda * np.ones(n - 1), 2 * lmbda + np.ones(n), -lmbda * np.ones(n - 1)],
                         dtype=np.dtype(object))
    A_BW = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csr')
    for i in range(mt):
        u_j0 = u_j[1:mx]
        sol_arr = spsolve(A_BW, u_j0)
        u_j = [0]
        for i in sol_arr:
            u_j.append(i)
        u_j.append(0)
    return u_j


def ck(lmbda, mx, mt, u_j):
    n = round(mx - 1)
    diagonals = np.array([(-lmbda / 2) * np.ones(n - 1), lmbda + np.ones(n), (-lmbda / 2) * np.ones(n - 1)],
                         dtype=np.dtype(object))
    A_CK = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csr')
    diagonals = np.array([(lmbda / 2) * np.ones(n - 1), np.ones(n) - lmbda, (lmbda / 2) * np.ones(n - 1)],
                         dtype=np.dtype(object))
    B_CK = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csr')
    for i in range(mt):
        u_j0 = u_j[1:mx]
        sol_arr = spsolve(A_CK, B_CK.dot(u_j0))
        u_j = [0]
        for i in sol_arr:
            u_j.append(i)
        u_j.append(0)
    return u_j
