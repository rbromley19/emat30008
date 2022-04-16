import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import pyplot
from integrate_ode import solve_ode
from scipy.optimize import root
from math import nan
from ode_functions import pred_prey

b = 0.2
int = [0.25, 0.25]
t = np.linspace(0, 150, 1501)
bspace = np.linspace(0.1, 0.5, 5)

sol = solve_ode(lambda t, u: pred_prey(t, u, b), int, t, 'rk4', 0.001)
x = sol[:, 0]
y = sol[:, 1]
pyplot.plot(x, y)
pyplot.show()

# X-nullcline
Xval = np.linspace(0.1, 0.6, 100)
Yval = np.zeros(np.size(Xval))
for (i, X) in enumerate(Xval):
    result = root(lambda N: pred_prey(nan, (X, N), b)[0], 0.25)
    if result.success:
        Yval[i] = result.x
    else:
        Yval[i] = nan
pyplot.plot(Xval, Yval)

# Y-nullcline
Xval = np.linspace(0.1, 0.6, 100)
Yval = np.zeros(np.size(Xval))
for (i, X) in enumerate(Xval):
    result = root(lambda N: pred_prey(nan, (X, N), b)[1], 0.25)
    if result.success:
        Yval[i] = result.x
    else:
        Yval[i] = nan
pyplot.plot(Xval, Yval)
pyplot.show()

# %%
result = root(lambda u: pred_prey(nan, u, b), (0.25, 0.25))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
result = root(lambda u: pred_prey(nan, u, b), (0.5, 0.3))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
# result = root(lambda u: pred_prey(nan, u, b), np.array((0.4, 0.5)))
# if result.success:
#     print("Equilibrium at {}".format(result.x))
# else:
#     print("Failed to converge")
