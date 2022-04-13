import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import pyplot
from solveODE import solve_ode
from scipy.optimize import root
from math import nan


def pred_prey(t, u, b):
    a = 1
    d = 0.1
    b = 0.2
    x, y = u
    dx = x * (1 - x) - (a * x * y) / (d + x)
    dy = b * y * (1 - (y / x))
    return np.array((dx, dy))

b = 0.2
int = [0.25, 0.25]
t = np.linspace(0, 150, 1501)
bspace = np.linspace(0.1, 0.5, 5)

sol = solve_ode(lambda t, u: pred_prey(t, u, b), int, t, 'rk4', 0.001)
print(sol)
x = sol[:, 0]
y = sol[:, 1]
# phase portrait: x, time-series: t
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# X-nullcline
Xval = np.linspace(0.05,0.55, 51)
Yval = np.zeros(np.size(Xval))
for (i, X) in enumerate(Xval):
    result = root(lambda N: pred_prey(nan, (X, N), b)[0], 0.25)
    if result.success:
        Yval[i] = result.x
    else:
        Yval[i] = nan
pyplot.plot(Xval, Yval)

# Y-nullcline
Xval = np.linspace(0.05, 0.55, 51)
Yval = np.zeros(np.size(Xval))
for (i, X) in enumerate(Xval):
    result = root(lambda N: pred_prey(nan, (X, N), b)[1], 0.25)
    if result.success:
        Yval[i] = result.x
    else:
        Yval[i] = nan
pyplot.plot(Xval, Yval)
