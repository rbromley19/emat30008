from ode_functions import cubic, hopf_normal, hopf_mod
import numpy as np
from scipy.optimize import fsolve
from numerical_shooting import orbit_calc, num_shoot
import matplotlib.pyplot as plt


def natural_continuation(f, u0, range, space, discretisation):
    params = np.linspace(range[0], range[1], space)
    result = []
    for i in params:
        sol = fsolve(lambda U, f: discretisation(U, f, i), u0, f)
        u0 = sol
        result.append(sol)
        u0 = np.round(u0, 5)
    results = np.array(result)
    plt.plot(params, [i[0] for i in results])
    plt.show()
    result = np.array(result)
    return result


def cubic_natural():
    result = natural_continuation(cubic, 1.5, [2], 0, (-2, 2), 30, lambda U, f, par: f(U, par))


def hopf_natural():
    natural_continuation(hopf_normal, [0.1, 0.5, 6], (2, 0), 30, lambda U, f, par: num_shoot(U, f, 0, par))


def hopf_modif():
    natural_continuation(hopf_mod, [0.1, 0.5, 6], (2, -1), 30, lambda U, f, par: num_shoot(U, f, 0, par))


if __name__ == '__main__':
    cubic_natural()
    hopf_natural()
    hopf_modif()
