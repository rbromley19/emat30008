from numerical_shooting import orbit_calc
import numpy as np
from ode_functions import hopf_bf, hopf_exp, ode_3, ode_3_exp, predator_prey, ode_1
from integrate_ode import solve_ode
import matplotlib.pyplot as plt


def predator_prey_shooting():
    shoot_result = orbit_calc(predator_prey, [0.2, 0.3, 22], var=0)
    x, y = shoot_result[:-1]
    print(x, y)
    T = shoot_result[-1]
    print('Calculated Predator-Prey coordinates using numerical shooting: ' + str(x) + ', ' + str(y))
    print('Calculated Predator-Prey time period using numerical shooting: ' + str(T))
    solve_t = np.linspace(0, T, 100)
    realsol = solve_ode(predator_prey, [x, y], solve_t, 'rk4', 0.01)
    orbit_x = realsol[0]
    orbit_y = realsol[1]
    plt.plot(orbit_x, orbit_y, 'g', label="Actual Orbit")
    plt.plot(x, y, 'rx', label='Shooting Result')
    plt.legend()
    plt.show()


# Function to test Hopf bifurcation shooting result against explicit solution
def hopf_test(function, u0, phase):
    print('--------- Now running the Hopf bifurcation normal-form shooting tests (suitable inputs) ---------')
    test_result = orbit_calc(function, u0, var=phase)
    T = test_result[-1]
    u0 = test_result[:-1]
    pass_count = 0
    fail_count = 0
    real_T = 2 * np.pi
    if np.allclose(T, real_T):
        pass_count = pass_count + 1
        print('Correct period found - PASS')
    else:
        print('Incorrect period found - FAIL')
        fail_count = fail_count + 1
    t = np.linspace(0, T, 100)
    sol = solve_ode(function, u0, t, 'rk4', 0.01)
    point0 = [sol[0][0], sol[1][0]]
    pointf = [sol[0][-1], sol[1][-1]]
    if np.allclose(point0, pointf):
        pass_count = pass_count + 1
        print('Initial points match end points - PASS')
    else:
        fail_count = fail_count + 1
        print('Initial and end points do not match - FAIL')
    exp_sol = hopf_exp(t, theta=np.pi, b=1)
    real_point0 = [exp_sol[0][0], exp_sol[1][0]]
    real_pointf = [exp_sol[0][-1], exp_sol[1][-1]]
    if np.allclose(point0, real_point0):
        pass_count = pass_count + 1
        print('Calculated and explicit solution initial points match - PASS')
    else:
        fail_count = fail_count + 1
        print('Calculated and explicit solution initial points do not match - FAIL')

    if np.allclose(pointf, real_pointf):
        pass_count = pass_count + 1
        print('Calculated and explicit solution end points match - PASS')
    else:
        fail_count = fail_count + 1
        print('Calculated and explicit solution end points do not match - FAIL')
    print('Tests passed: ' + str(pass_count))
    print('Tests failed: ' + str(fail_count))


def du3_shoot_test(function, u0, phase):
    print('--------- Now running the 3-D system shooting tests (suitable inputs) ---------')
    test_result = orbit_calc(ode_3, u0, var=phase)
    print(test_result)
    T = test_result[-1]
    u0 = test_result[:-1]
    pass_count = 0
    fail_count = 0
    real_T = 2 * np.pi
    if np.allclose(T, real_T):
        pass_count = pass_count + 1
        print('Correct period found - PASS')
    else:
        print('Incorrect period found - FAIL')
        fail_count = fail_count + 1
    t = np.linspace(0, T, 100)
    sol = solve_ode(function, u0, t, 'rk4', 0.01)
    point0 = [sol[0][0], sol[1][0]]
    pointf = [sol[0][-1], sol[1][-1]]
    if np.allclose(point0, pointf):
        pass_count = pass_count + 1
        print('Initial points match end points - PASS')
    else:
        fail_count = fail_count + 1
        print('Initial and end points do not match - FAIL')
    exp_sol = hopf_exp(t, theta=np.pi, b=1)
    real_point0 = [exp_sol[0][0], exp_sol[1][0]]
    real_pointf = [exp_sol[0][-1], exp_sol[1][-1]]
    if np.allclose(point0, real_point0):
        pass_count = pass_count + 1
        print('Calculated and explicit solution initial points match - PASS')
    else:
        fail_count = fail_count + 1
        print('Calculated and explicit solution initial points do not match - FAIL')

    if np.allclose(pointf, real_pointf):
        pass_count = pass_count + 1
        print('Calculated and explicit solution end points match - PASS')
    else:
        fail_count = fail_count + 1
        print('Calculated and explicit solution end points do not match - FAIL')
    print('Tests passed: ' + str(pass_count))
    print('Tests failed: ' + str(fail_count))


if __name__ == '__main__':
    hopf_test(hopf_bf, [1, 1, 6], 0)
    du3_shoot_test(ode_3, [-1, -1, 0, 6], 0)
