from ode_functions import cubic, hopf_normal, hopf_mod
import numpy as np
from scipy.optimize import fsolve
from numerical_shooting import orbit_calc, num_shoot


def natural_continuation(f, u0, par, vary_par, range, space, discretisation):
    params = np.linspace(range[0], range[1], space)
    result = []
    counter = 0
    for i in params:
        counter = counter + 1
        print(counter)
        par[vary_par] = i
        sol = fsolve(lambda U, f: discretisation(U, f, i), u0, f)
        result.append(sol)
    result = np.array(result)
    print(result)
    return result


def cubic_natural():
    result = natural_continuation(cubic, 1.5, [2], 0, (-2, 2), 30, lambda U, f, par: f(U, par))
    print(result)


def hopf_natural():
    natural_continuation(hopf_normal, [0.1, 0.5, 6], [2], 0, (2, 0), 30, lambda U, f, par: num_shoot(U, f, 0, par))


if __name__ == '__main__':
    cubic_natural()
    hopf_natural()
