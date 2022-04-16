import numerical_shooting
import numpy as np
from ode_functions import hopf_bf, hopf_exp, ode_3, ode_3_exp


# Function to test Hopf bifurcation shooting result against explicit solution
def hopf_test(b, u0, T0, phase0, t):
    shoot_result = numerical_shooting.orbit_calc(lambda t, u: hopf_bf(t, u, b=b), u0, T0, phase0)
    shoot_points = shoot_result[:-1]
    T = shoot_result[-1]
    exp_result = hopf_exp(t, theta=T, b=b)
    print(exp_result)
    if np.allclose(exp_result, shoot_points):
        print("pass")
    else:
        print("fail")


def ode_3(t, u, b, s):
    u1, u2, u3 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    du3 = -u3
    return np.array([du1, du2, du3])


def du3_test(b, s, u0, T0, phase0, t):
    shoot_result = numerical_shooting.orbit_calc(lambda t, u: ode_3(t, u, b=b, s=s), u0, T0, phase0)
    shoot_points = shoot_result[:-1]
    T = shoot_result[-1]
    exp_result = hopf_exp(t, theta=T, b=b, s=s)
    print(exp_result)
    if np.allclose(exp_result, shoot_points):
        print("pass")
    else:
        print("fail")


if __name__ == '__main__':
    hopf_test(1, [1, 1.5], 5, 0, 0)
    du3_test(1, -1, [1.5, 0.5, 1], 5, 0, 0)
