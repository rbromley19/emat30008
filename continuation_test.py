from ode_functions import cubic, hopf_normal, hopf_mod
import numpy as np
from numerical_shooting import num_shoot
from num_continuation import natural_continuation


def cubic_natural():
    natural_continuation(cubic, 1.5, (-2, 2), 30, lambda U, f, par: f(U, par), 'Cubic', 'x')


def hopf_natural():
    natural_continuation(hopf_normal, [0.1, 0.5, 6], (2, 0), 30, lambda U, f, par: num_shoot(U, f, 0, par), 'Hopf Normal', 'Beta')


def hopf_modif():
    natural_continuation(hopf_mod, [0.1, 0.5, 6], (2, -1), 30, lambda U, f, par: num_shoot(U, f, 0, par), 'Modified Hopf', 'Beta')


if __name__ == '__main__':
    cubic_natural()
    hopf_natural()
    hopf_modif()

