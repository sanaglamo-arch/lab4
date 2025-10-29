from regression import run_regression
from interpolation import build_function_data, create_interpolators, is_in_range
from visualize import plot_interpolation, plot_comparison
import numpy as np


def main():

    reg_data = run_regression()
    print("\nТеперь переходим к методам интерполяции.")
    formula = input("Введите функцию (например sin(x) + 0.5*x): ").strip() or "sin(x)"
    x_start = float(input("Начало диапазона x: ") or 0)
    x_end = float(input("Конец диапазона x: ") or 10)
    step = float(input("Шаг: ") or 1)

    x, y = build_function_data(formula, x_start, x_end, step)
    interps = create_interpolators(x, y)

    x_pred = float(input(f"Введите значение x для интерполяции (в диапазоне {x[0]}..{x[-1]}): "))

    if not is_in_range(x, x_pred):
        print("Ошибка: значение вне диапазона исходных данных.")
        return

    true_val = eval(formula, {"__builtins__": None, "np": np, "sin": np.sin, "cos": np.cos,
                              "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pi": np.pi, "e": np.e},
                    {"x": x_pred})
    print(f"\nИстинное значение функции f({x_pred}) = {true_val:.4f}")

    for name, f in interps.items():
        print(f"\nИнтерполяция методом: {name}")
        y_pred = f(x_pred)
        print(f"{name}: y({x_pred}) = {y_pred:.4f}")
        plot_interpolation(x, y, f, label=name)

    plot_comparison(x, y, interps)
    print("\nГотово! Все методы выполнены.")


if __name__ == "__main__":
    main()
