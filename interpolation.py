import numpy as np
from scipy.interpolate import interp1d, CubicSpline


def build_function_data(formula: str, x_start: float, x_end: float, step: float):
    safe_env = {"__builtins__": None, "np": np, "sin": np.sin, "cos": np.cos,
                "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pi": np.pi, "e": np.e}
    x = np.arange(x_start, x_end + step / 2, step)
    y = np.array([eval(formula, safe_env, {"x": val}) for val in x], dtype=float)
    return x, y


def create_interpolators(x, y):
    interps = {
        "linear": interp1d(x, y, kind="linear", fill_value="extrapolate"),
        "quadratic": interp1d(x, y, kind="quadratic", fill_value="extrapolate"),
        "cubic": interp1d(x, y, kind="cubic", fill_value="extrapolate"),
        "spline": CubicSpline(x, y)
    }
    return interps


def is_in_range(x, val):
    return np.min(x) <= val <= np.max(x)
