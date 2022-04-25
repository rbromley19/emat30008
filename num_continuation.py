from ode_functions import cubic, hopf_normal, hopf_mod
import numpy as np
from scipy.optimize import fsolve


def natural_continuation(f, x0, param0, a, b, space, discretisation, solver):

    params = np.linspace(a, b, space)
    sol_array = []
    param0 = [param0]

    for i in params:
        param0[0] = i
        x0 = np.array(solver(discretisation(f), x0, args=param0))
        sol_array.append(x0)
    sol_array = np.array(sol_array)
    return params, sol_array


def cubic_natural():
    params, sol_array = natural_continuation(cubic, 1.5, 2, -2, 2, 101, lambda x: x, fsolve)
    print(params, sol_array)

def hopf_natural():
    # params, sol_array = param_continuation(hopf_normal, )

if __name__ == '__main__':
    cubic_natural()
