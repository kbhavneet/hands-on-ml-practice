"""
chapter 1 - life satisfaction vs gdp

re-implementation of the example from 'hands-on machine learning'
to compare linear regression with k-nearest neighbors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "chapter01" / "lifesat.csv"
GDP_COLUMN = "GDP per capita (USD)"
SATISFACTION_COLUMN = "Life satisfaction"

X_MIN, X_MAX = 23_500, 62_500
Y_MIN, Y_MAX = 5.0, 8.0
N_POINTS = 100

CYPRUS_GDP_PER_CAPITA = 37_655.2


def load_data(path: Path = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """load the dataset and return features, targets and full dataframe."""
    lifesat = pd.read_csv(DATA_PATH)
    X = lifesat[[GDP_COLUMN]].to_numpy()
    y = lifesat[[SATISFACTION_COLUMN]].to_numpy()
    return X, y, lifesat


def make_plot(
    lifesat: pd.DataFrame,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    title: str,
) -> None:
    """plot scatter of data and model predictions."""
    fig, ax = plt.subplots()

    # scatter plot of the original data
    ax.scatter(lifesat[GDP_COLUMN], lifesat[SATISFACTION_COLUMN])

    # model prediction line
    ax.plot(x_fit, y_fit)

    # axis labels and limits
    ax.set_xlabel(GDP_COLUMN)
    ax.set_ylabel(SATISFACTION_COLUMN)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    # grid and title
    ax.grid(True)
    ax.set_title(title)

    plt.show()


def evaluate_model(
    model: RegressorMixin,
    X: np.ndarray,
    y: np.ndarray,
    lifesat: pd.DataFrame,
    label: str,
) -> None:
    """fit a model, plot predictions and print cyprus estimate."""
    # fit the model
    model.fit(X, y)

    # create x values and predictions for the line
    x_fit = np.linspace(X_MIN, X_MAX, N_POINTS).reshape(-1, 1)
    y_fit = model.predict(x_fit)

    # plot data and model predictions
    make_plot(lifesat, x_fit, y_fit, title=label)

    # prediction for cyprus
    X_new = np.array([[CYPRUS_GDP_PER_CAPITA]])
    cyprus_pred = model.predict(X_new)[0, 0]
    print(f"{label} prediction for cyprus: {cyprus_pred:.3f}")


def main() -> None:
    """run the comparison between linear regression and k-nearest neighbors."""
    X, y, lifesat = load_data()

    # linear regression
    evaluate_model(
        LinearRegression(),
        X,
        y,
        lifesat,
        label="linear regression",
    )

    # k-nearest neighbors
    evaluate_model(
        KNeighborsRegressor(n_neighbors=3),
        X,
        y,
        lifesat,
        label="k-nearest neighbors (k=3)",
    )


if __name__ == "__main__":
    main()
