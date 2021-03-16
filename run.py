"""Determines the optimum load distribution on the wing supplied via the command line."""

import sys
import json

import matplotlib.pyplot as plt
from optimizer import TwistOptimizer


if __name__=="__main__":

    # Get MachUpX wing input from argv
    input_file = sys.argv[1]
    with open(input_file) as input_handle:
        wing_input = json.load(input_handle)

    # Initialize optimizer
    my_case = TwistOptimizer(wing_input, 100.0, 0.0023769)

    # Run
    s, twist = my_case.optimize(11, 0.5)
    print("Minimum induced drag coef: {0}".format(my_case.C_D))

    # Get distributions
    s, twist, lift, load = my_case.get_distributions()

    # Plot optimum result
    fig, ax = plt.subplots(figsize=(3*3.25, 3.0), nrows=1, ncols=3)
    ax[0].plot(s, twist, 'k-')
    ax[0].set_xlabel('$\\eta$')
    ax[0].set_ylabel('Twist [deg]')
    ax[1].plot(s, lift, 'k-')
    ax[1].set_xlabel('$\\eta$')
    ax[1].set_ylabel('$\\frac{c_L}{C_L}$')
    ax[2].plot(s, load, 'k-')
    ax[2].set_xlabel('$\\eta$')
    ax[2].set_ylabel('$\\frac{c_n c}{C_L c_m}$')
    plt.show()

    # Write results to file
    with open("results.csv", 'w') as results_handle:
        print("{0},{1},{2},{3}".format('2y/b', 'Twist', 'Lift', 'Load'), file=results_handle)
        for si, ti, li, loi in zip(s, twist, lift, load):
            print("{0},{1},{2},{3}".format(si, ti, li, loi), file=results_handle)
