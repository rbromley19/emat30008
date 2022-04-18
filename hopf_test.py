import numerical_shooting
import numpy as np
from ode_functions import hopf_bf, hopf_exp, ode_3, ode_3_exp


# Function to test Hopf bifurcation shooting result against explicit solution
def hopf_test(b, u0, T0, phase0, t):
    shoot_result = numerical_shooting.orbit_calc(lambda t, u: hopf_bf(t, u, b=b), u0, T0, phase0)
    shoot_points = shoot_result[:-1]
    T = shoot_result[-1]
    exp_result = hopf_exp(t, theta=T, b=b)
    if np.allclose(exp_result, shoot_points):
        print("pass")
    else:
        print("fail")


# Function to test 3-d system shooting result against explicit solution
def du3_test(b, s, u0, T0, phase0, t):
    shoot_result = numerical_shooting.orbit_calc(lambda t, u: ode_3(t, u, b=b, s=s), u0, T0, phase0)
    shoot_points = shoot_result[:-1]
    T = shoot_result[-1]
    exp_result = ode_3_exp(t, theta=T, b=b)
    if np.allclose(exp_result, shoot_points):
        print("pass")
    else:
        print("fail")


if __name__ == '__main__':
    hopf_test(1, [1, 1.5], 5, 0, 0)
    b = 1
    s = -1
    u0 = [1.5, 0.5, 1]
    T0 = 5
    phase0 = 0
    # shoot_result = numerical_shooting.orbit_calc(lambda t, u: ode_3(t, u, b=b, s=s), [1.5, 0.5, 1], 5, phase0)
    du3_test(1, -1, [1.5, 0.5, 1], 5, 0, 0)
    # print(shoot_result)
