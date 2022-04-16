import numerical_shooting
import numpy as np


# Function to simulate Hopf bifurcation normal form
def hopf_bf(t, u, b):
    s = -1
    u1, u2 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


# Function to simulate Hopf bifurcation explicit solution
def hopf_exp(t, theta, b):
    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    return np.array([u1, u2])


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


if __name__ == '__main__':
    b = 1
    u0 = [1, 1.5]
    T0 = 5
    phase0 = 0
    t = 0
    hopf_test(b, u0, T0, phase0, t)
