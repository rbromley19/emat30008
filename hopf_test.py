import numerical_shooting
import numpy as np


def hopf_bf(t, u, b):
    s = -1
    u1, u2 = u
    du1 = b * u1 - u2 + s * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + b * u2 + s * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])


def hopf_exp(t, theta, b):
    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    return np.array([u1, u2])


# sol = solve_ode(lambda t, u: hopf_bf(t, u, b=0.5), (-1, 0), t, 'rk4', 0.001)
# x = sol[:, 0]
# y = sol[:, 1]
# pyplot.plot(x, y)
# pyplot.show()

result = numerical_shooting.orbit_calc(lambda t, u: hopf_bf(t, u, b=1), [1, 1.5], 5, 0)
point = result[:-1]
T = result[-1]

# sol = solve_ode(lambda t, u: hopf_bf(t, u, b=0.5), (-1, 0), t, 'rk4', 0.001)
result_exp = hopf_exp(0, theta=T, b=1)
print(result_exp)

if np.allclose(result_exp, point):
    print("pass")
else:
    print("fail")

