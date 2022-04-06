import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def pred_prey(t, u, a, b, d):
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array((dx, dy))


sol = solve_ivp(pred_prey, [0, 15], [10, 5], args=(1, 0.26, 0.1),
                dense_output=True)
t = np.linspace(0, 15, 300)
z = sol.sol(t)

plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=True)
plt.title('Predator-Prey System')
plt.show()
