from numerical_shooting import orbit_calc
import numpy as np
from ode_functions import hopf_bf, hopf_exp, ode_3, ode_3_exp


# fix this function line 12
def shooting_test(function, u0, phase, function_exp):
    test_result = orbit_calc(function, u0, phase)[:-1]
    T = test_result[-1]
    print(T)
    exp_result = function_exp(t, theta=T, b=b)
    if np.allclose(exp_result, test_result):
        print("Test" + str(function) + " has successfully passed")
    else:
         print("Test" + str(function) + " has failed")


# Function to test Hopf bifurcation shooting result against explicit solution
# def hopf_test(b, u0, phase0, t):
#     shoot_result = orbit_calc(lambda t, u: hopf_bf(t, u, b=1), u0, phase0)
#     shoot_points = shoot_result[:-1]
#     T = shoot_result[-1]
#     exp_result = hopf_exp(t, theta=T, b=b)
#     if np.allclose(exp_result, shoot_points):
#         print("pass")
#     else:
#         print("fail")


# Function to test 3-d system shooting result against explicit solution
# def du3_test(b, s, u0, phase0, t):
#     shoot_result = orbit_calc(lambda t, u: ode_3(t, u, b=b, s=s), u0, phase0)
#     shoot_points = shoot_result[:-1]
#     T = shoot_result[-1]
#     exp_result = ode_3_exp(t, theta=T, b=b)
#     if np.allclose(exp_result, shoot_points):
#         print("pass")
#     else:
#         print("fail")


if __name__ == '__main__':
    shooting_test(lambda t, u: hopf_bf(t, u, b=1), [1, 1.5, 5], 0, 'hopf_exp')
    # hopf_test(1, [1, 1.5, 5], 0, 0)
    # b = 1
    # s = -1
    # u0 = [1.5, 0.5, 1]
    # T0 = 5
    # phase0 = 0
    # du3_test(1, -1, [1.5, 0.5, 1, 5], 0, 0)
    # du3_test(1, -1, [-1, 0, 1, 6], 0, 0)
