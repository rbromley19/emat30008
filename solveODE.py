import matplotlib.pyplot as plt
import numpy as np
import math


def f(t, x):
    return x


# t = np.linspace(0, 1, 100)
t = [0, 1]
true_sol = math.e


def euler_step(f, x0, t0, h):
    x1 = x0 + h * np.array(f(t0, x0))
    t1 = t0 + h
    return x1, t1


def rk4_step(f, x0, t0, h):
    k1 = np.array(f(t0, x0))
    k2 = np.array(f(t0 + h / 2, x0 + (h / 2) * k1))
    k3 = np.array(f(t0 + h / 2, x0 + (h / 2) * k2))
    k4 = np.array(f(t0 + h, x0 + h * k3))
    x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t1 = t0 + h
    return x1, t1


def solve_to(method, f, x1, t1, t2, dt_max):
    if method == "euler":
        func = euler_step
    elif method == "rk4":
        func = rk4_step

    steps = math.floor((t2 - t1) / dt_max)
    for i in range(steps):
        x1, t1 = func(f, x1, t1, dt_max)
    if t1 != t2:
        h = t2 - t1
        x1, t1 = func(f, x1, t1, h)
    return x1


def solve_ode(f, x0, t, method, dt_max, true_sol):
    # x = np.zeros(len(t))
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(1, len(t)):
        x[i] = solve_to(method, f, x[i - 1], t[i - 1], t[i], dt_max)
    return np.array(x)


def error_plot(f, x0, t, true_sol):
    h_val = np.logspace(-4, -1, 50)
    euler_list = np.zeros(int(len(h_val)))
    rk4_list = np.zeros(int(len(h_val)))

    for i in range(len(h_val)):
        euler_sol = solve_ode(f, x0, t, 'euler', h_val[i], true_sol)
        error = abs(euler_sol[-1] - true_sol)
        euler_list[i] = error

    for i in range(len(h_val)):
        rk4_sol = solve_ode(f, x0, t, 'rk4', h_val[i], true_sol)
        error = abs(rk4_sol[-1] - true_sol)
        rk4_list[i] = error

    # print(rk4_list)
    # print(euler_list)
    ax = plt.gca()
    ax.scatter(h_val, euler_list)
    ax.scatter(h_val, rk4_list)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(['Euler', 'RK4'], loc='best')
    plt.show()


def euler_run(f, t, true_sol):
    method = 'euler'
    x0 = [1]
    dt_max = 0.01
    euler = solve_ode(f, x0, t, method, dt_max, true_sol)[-1]
    print('Euler approximation = ' + str(euler))


def rk4_run(f, t, true_sol):
    method = 'rk4'
    x0 = [1]
    dt_max = 0.01
    rk4 = solve_ode(f, x0, t, method, dt_max, true_sol)[-1]
    print('Runge-kutta approximation = ' + str(rk4))


euler_run(f, t, true_sol)
rk4_run(f, t, true_sol)

t_lim = [0, 1]
error_plot(f, [1], t_lim, true_sol)

t = np.linspace(0, 1, 100)


def f_s(t, X):
    x = X[0]
    y = X[1]
    xdot = y
    ydot = -x
    return [xdot, ydot]


x0 = [1, 0]
t = np.linspace(0, 20, 100)
eul_sol = solve_ode(f_s, x0, t, 'euler', 0.001, true_sol)
rk4_sol = solve_ode(f_s, x0, t, 'rk4', 0.001, true_sol)
xeul = eul_sol[:, 0]
xeuldot = eul_sol[:, 1]
xrk4 = rk4_sol[:, 0]
xrk4dot = rk4_sol[:, 1]


def X_analytic(t, X0):
    c2, c1 = X0

    return [c1 * np.sin(t) + c2 * np.cos(t), c1 * np.cos(t) - c2 * np.sin(t)]


x_analytic, xdot_analytic = X_analytic(t, x0)

plt.plot(t, xeul, t, xrk4, t, x_analytic)
plt.xlabel('time, t')
plt.ylabel('x')
plt.show()

plt.plot(xeuldot, xeul, xrk4dot, xrk4)
plt.xlabel('xdot')
plt.ylabel('x')
plt.show()
