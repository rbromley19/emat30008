from ode_functions import cubic, hopf_normal, hopf_mod
import numpy as np
from scipy.optimize import fsolve
from numerical_shooting import orbit_calc, num_shoot


# def hopf_2(t, u, b):
#     u1, u2 = u
#     du1 = b * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
#     du2 = u1 + b * u2 - u2 * (u1 ** 2 + u2 ** 2)
#     return np.array([du1, du2])


def natural_continuation(f, u0, par, vary_par, range, space, discretisation):
    params = np.linspace(range[0], range[1], space)
    result = []
    counter = 0
    for i in params:
        counter = counter + 1
        print(counter)
        par[vary_par] = i
        if discretisation == 'num_shoot':
            sol = fsolve(lambda U, f: discretisation(U, f, var=0), u0, f)
            result.append(sol)
        else:
            sol = fsolve(lambda U, f: discretisation(f), u0, f)
        result.append(sol)
    result = np.array(result)
    return result


def cubic_natural():
    # params, sol_array = natural_continuation(cubic, [1], -2, 0, -2, 2, 30, lambda x: x, fsolve)
    result = natural_continuation(cubic, 1.5, [2], 0, (-2, 2), 30, lambda x: x)
    print(result)


def hopf_natural():
    pass


if __name__ == '__main__':
    cubic_natural()
