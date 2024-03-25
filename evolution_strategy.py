import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import time
import random
from cec2017.functions import f3, f19, f10
from typing import Callable, Tuple, Optional, Dict, List
from matplotlib.axes import Axes
import sys
import os
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from Zadanie1.gradient_descent import gradient_descent


def visualize_functions(function, domain=(-100, 100), points=30, dimension=2, ax=None):
    # create points^2 tuples of (x,y) and populate z
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])

    if dimension > 2:
        # concatenate remaining zeros
        tail = np.zeros((xys.shape[0], dimension - 2))
        x = np.concatenate([xys, tail], axis=1)
        zs = function(x)
    else:
        zs = function(xys)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    X = xys[:, 0].reshape((points, points))
    Y = xys[:, 1].reshape((points, points))
    Z = zs.reshape((points, points))

    # plot surface
    surf = ax.plot_surface(X, Y, Z, cmap="plasma")  # , edgecolor='none')
    ax.contour(X, Y, Z, zdir="z", cmap="plasma", linestyles="solid", offset=40)
    ax.contour(X, Y, Z, zdir="z", colors="k", linestyles="solid")
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15, label="")
    ax.set_title(function.__name__, fontsize=20)  # Increase the font size of the title

    # plot contour
    ax.contour(X, Y, Z, levels=10, cmap="plasma", linestyles="solid", offset=40)

    ax.set_title(function.__name__, fontsize=20)  # Increase the font size of the title
    ax.set_xlabel("x", fontsize=14)  # Increase the font size of the x-label
    ax.set_ylabel("y", fontsize=14)  # Increase the font size of the y-label
    ax.set_zlabel("y", fontsize=14)  # Increase the font size of the z-label

    # Set the background color
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    plt.show()

    return fig


def ES_1_plus_1(
    f: Callable,
    optim_params: Dict[str, float],
):
    dimension = optim_params.get("dimension")
    generations = optim_params.get("generations")
    domain_lower_bound = optim_params.get("domain_lower_bound")
    domain_upper_bound = optim_params.get("domain_upper_bound")
    n = optim_params.get("n")
    parent = np.random.uniform(
        domain_lower_bound, domain_upper_bound, size=(1, dimension)
    )
    alpha = optim_params.get("alpha")
    recent_successes = np.zeros(n)
    sigma = optim_params.get("sigma")
    y_vals = []

    for i in range(generations):
        child = parent + np.random.normal(0, 1, dimension) * sigma

        if f(child) < f(parent):
            parent = child
            recent_successes[i % n] = 1
        else:
            recent_successes[i % n] = 0

        if i % n == 0:
            success_freq = np.sum(recent_successes) / n
            if success_freq < 0.2:
                sigma = sigma / alpha
            else:
                sigma = sigma * alpha
            recent_successes = np.zeros(n)
        y_vals.append(f(parent)[0])
    return parent, f(parent), y_vals


def run_evolution_strategy(optim_params):
    y_list = []
    sigmas = np.arange(0.0, 10.5, 0.5)
    for sigma in sigmas:
        y_temp = []
        optim_params["sigma"] = sigma
        for i in range(50):
            x, y = ES_1_plus_1(f=f19, optim_params=optim_params)
            print("hey")
            y_temp.append(y)
        print(sigma)
        print(np.mean(y_temp))
        y_list.append(np.mean(y_temp))

    df = pd.DataFrame({"sigma": sigmas, "average_result": y_list})

    df.to_csv(f"results_f19.csv", index=False)

    return y_list


def f19_wrapper(x):
    x_reshaped = x.reshape(1, -1)
    return f19(x_reshaped)[0]


def f3_wrapper(x):
    x_reshaped = x.reshape(1, -1)
    return f3(x_reshaped)[0]


def plot_convergence(es_values: pd.DataFrame, gd_values: pd.DataFrame, values: pd.DataFrame) -> None:
    print(len(es_values), len(gd_values))
    if len(es_values) != len(gd_values):
        diff = len(gd_values) - len(es_values)
        if len(es_values) < len(gd_values):
            es_values.extend([es_values[-1]] * diff)
        else:
            gd_values.extend([gd_values[-1]] * -diff)
        print(len(es_values), len(gd_values))

    data = pd.DataFrame({"3": es_values, "10": gd_values, "6": values})

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    palette = sns.color_palette(["#FF0000", "#0000FF", "#00FF00"])

    sns.lineplot(data=data, palette=palette, linewidth=2.5, dashes=True, alpha=0.7)

    plt.title("Convergence of ES for f19 function", fontsize=30)
    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("Function Value", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.yscale("log")
    plt.legend(title="sigma", title_fontsize="23", fontsize=22)
    plt.show()


def main() -> None:
    optim_params: Dict[str, float] = {
        "dimension": 10,
        "generations": 100_000,
        "domain_lower_bound": 0,
        "domain_upper_bound": 300,
        "n": 50,
        "alpha": 2,
        "sigma": 6,
    }
    # visualize_functions(f3, domain=(-100, 100), points=30, dimension=10, ax=None)
    # y = run_evolution_strategy(optim_params)
    # data = pd.read_csv('results_f19.csv')
    # plt.plot(data['sigma'], data['average_result'])
    # plt.yscale('log')
    # plt.show()

    # num_runs = 30
    # all_results = []

    # for i in range(num_runs):
    #     print(i)
    #     x, y, y_vals = ES_1_plus_1(f=f19, optim_params=optim_params)
    #     all_results.append(y_vals)

    # all_results = list(map(list, zip(*all_results)))

    # average_results = [sum(vals) / num_runs for vals in all_results]

    # data = pd.DataFrame({"average_result": average_results})
    # data.to_csv("results_f19_convergence_sigma_6.csv", index=False)

    # x = np.random.uniform(0, 300, size=10)
    # print(f19_wrapper(x))
    # x, y = gradient_descent(
    #     f19_wrapper,
    #     x0=x,
    #     learning_rate=0.01,
    #     max_iterations=10_0000,
    #     tolerance=1e-4,
    #     clip_value=5,
    # )

    # grad_df = pd.DataFrame({"gradient_descent": y})
    # grad_df.to_csv("results_f19_gradient_descent.csv", index=False)
    # print(y[-1])
    plot_convergence(
        pd.read_csv("results_f19_convergence_sigma_10.csv")["average_result"],
        pd.read_csv("results_f19_convergence.csv")["average_result"],
        pd.read_csv("results_f19_convergence_sigma_6.csv")["average_result"],
    )
    return


if __name__ == "__main__":
    main()
