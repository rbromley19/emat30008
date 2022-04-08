import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import pyplot
from solveODE import solve_ode
from scipy.optimize import fsolve


def pred_prey(t, u):
    a = 1
    b = 0.28
    d = 0.1
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array((dx, dy))


int = [0.5, 0.5]
t = np.linspace(0, 200, 2001)
bspace = np.linspace(0.1, 0.5, 5)

sol = solve_ode(pred_prey, int, t, 'rk4', 0.001, 0)
print(sol)
x = sol[:, 0]
y = sol[:, 1]
# phase portrait: x, time-series: t
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



# sol = solve_ivp(pred_prey, [0, 15], [10, 5], dense_output=True)
#
# z = sol.sol(t)
#
# pyplot.plot(sol.t, sol.y[0, :])
# pyplot.show()
