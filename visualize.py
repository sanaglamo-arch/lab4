# visualize.py
import numpy as np
import matplotlib.pyplot as plt


def plot_interpolation(x, y, f, label, color="blue"):
    x_dense = np.linspace(np.min(x), np.max(x), 400)
    y_dense = f(x_dense)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", label="Исходные точки")
    plt.plot(x_dense, y_dense, color=color, label=label)
    plt.title(f"Метод интерполяции: {label}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_comparison(x, y, interps):
    x_dense = np.linspace(np.min(x), np.max(x), 400)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="black", label="Исходные точки")

    colors = ["red", "blue", "green", "purple"]
    for (name, f), color in zip(interps.items(), colors):
        plt.plot(x_dense, f(x_dense), color=color, label=name)

    plt.title("Сравнение всех методов интерполяции")
    plt.legend()
    plt.grid(True)
    plt.show()
