import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def run_regression():
    """
    Линейная регрессия с вводом CSV и выбором признаков пользователем.
    """

    csv_path = input("Введите путь к CSV-файлу: ").strip()
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print("Ошибка при чтении файла:", e)
        return None

    print(f"\nФайл успешно загружен! Строк: {len(data)}, столбцов: {len(data.columns)}")
    print("Доступные колонки:", ", ".join(data.columns))

    x_col = input("\nВведите имя признака X (независимая переменная): ").strip()
    y_col = input("Введите имя признака Y (зависимая переменная): ").strip()

    if x_col not in data.columns or y_col not in data.columns:
        print("Ошибка: колонки не найдены.")
        return None

    # --- 🔧 Очистка данных от NaN ---
    subset = data[[x_col, y_col]].dropna()
    if subset.empty:
        print("Ошибка: после очистки от NaN данных не осталось.")
        return None

    X = subset[[x_col]].values
    y = subset[y_col].values

    # Обучение модели
    model = LinearRegression()
    model.fit(X, y)

    a, b = model.coef_[0], model.intercept_
    print(f"\nУравнение: {y_col} = {a:.4f} * {x_col} + {b:.4f}")

    try:
        x_pred = float(input(f"\nВведите значение {x_col} для предсказания: "))
    except ValueError:
        print("Ошибка: ожидалось число.")
        return None

    y_pred = model.predict([[x_pred]])[0]
    print(f"Предсказанное значение {y_col}: {y_pred:.3f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Исходные данные", alpha=0.7)
    plt.plot(X, model.predict(X), color="red", label="Линия регрессии")
    plt.scatter(x_pred, y_pred, color="green", s=80, label=f"Прогноз ({x_col}={x_pred})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Метод линейной регрессии")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"model": model, "x": X, "y": y, "x_pred": x_pred, "y_pred": y_pred}
