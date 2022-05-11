import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def natural_continuation(f, u0, range, space, discretisation, system, parameter):
    if callable(f):
        param_range = np.linspace(range[0], range[1], space)
        result = []
        for i in param_range:
            sol = fsolve(lambda U, f: discretisation(U, f, i), u0, f)
            u0 = sol
            result.append(sol)
            u0 = np.round(u0, 5)
        results = np.array(result)
        plot_bifurcation(results, param_range, system, parameter)
        return results
    else:
        raise Exception('Input equation(s) must be a function f')


def plot_bifurcation(results, param_range, system, parameter):
    plt.plot(param_range, [i[0] for i in results])
    plt.ylabel('u(t)')
    plt.xlabel(str(parameter))
    plt.title('Bifurcation plot of ' + str(system) + ' equation with varying parameter ' + str(parameter))
    plt.show()
