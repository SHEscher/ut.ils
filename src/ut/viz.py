#!/usr/bin/python3
"""
Collection of visualization functions.

Author: Simon M. Hofmann | <[firstname].[lastname][ät]pm.me> | 2024
"""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

if TYPE_CHECKING:
    from pandas import DataFrame

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def model_summary_publication_ready(model: sm.regression.linear_model.RegressionResults) -> str:
    """
    Create the summary of a linear polynomial model that is publication ready (APA style).

    :param model: The polynomial model object.
    :return: The model summary message (APA style).
    """
    degree = len(model.params) - 1
    summary_message = f"\n({degree}-order polynomial fit: R2={model.rsquared:.3f}; "
    for bi, b in enumerate(model.params):
        summary_message += f"b{bi}={b:.3f}, SE{bi}={model.bse[bi]:.3f}, "
        summary_message += f"t{bi}={model.tvalues[bi]:.3f}, p{bi}<{model.pvalues[bi]:.3f}; "
    summary_message += f"AIC={model.aic:.0f}, BIC={model.bic:.0f}).\n"
    print(summary_message)
    return summary_message


def poly_model_fit(
    df: DataFrame,
    x_col: str,
    y_col: str,
    degree: int,
    return_x_eval: bool = False,
    n_x_eval: int = 2_000,
) -> tuple[sm.regression.linear_model.RegressionResults, np.ndarray] | sm.regression.linear_model.RegressionResults:
    """
    Fit a polynomial model of degree 'degree' to the data.

    :param df: The DataFrame that contains the data.
    :param x_col: The column name of the x-values.
    :param y_col: The column name of the y-values.
    :param degree: The degree (order) of the polynomial fit.
    :param return_x_eval: Whether to return the x-values for evaluation.
    :param n_x_eval: The number of x-values to evaluate the model on.
    :return: The polynomial model object and optionally the x-values for evaluation.
    """
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(df[[x_col]].to_numpy())
    model = sm.OLS(df[y_col], x_poly).fit()
    if return_x_eval:
        return model, polynomial_features.fit_transform(
            np.linspace(df[x_col].min(), df[x_col].max(), n_x_eval).reshape(-1, 1)
        )
    return model


def plot_poly_fit(
    df: DataFrame,
    x_col: str,
    y_col: str,
    degree: int,
    n_std: int = 2,
    scatter_kws: dict | None = None,
    line_kws: dict | None = None,
    ci_kws: dict | None = None,
    dpi: int = 150,
    verbose: bool = False,
) -> tuple[sm.regression.linear_model.RegressionResults, plt.Figure, plt.Axes]:
    """
    Plot a polynomial fit of degree 'degree' on the data.

    This function is similar to seaborn's regplot, but it will return the polynomial model object.

    :param df: The DataFrame that contains the data.
    :param x_col: The column name of the x-values.
    :param y_col: The column name of the y-values.
    :param degree: The degree (order) of the polynomial fit.
    :param n_std: The number of standard deviations for the confidence interval.
    :param scatter_kws: The keyword arguments for the scatterplot.
    :param line_kws: The keyword arguments for the line plot.
    :param ci_kws: The keyword arguments for the confidence interval.
    :param dpi: The resolution of the plot.
    :param verbose: Whether to print the model summary.
    :return: The polynomial model object, the figure, and the axes object.
    """
    # Fit the polynomial model
    model, x_eval = poly_model_fit(df=df, x_col=x_col, y_col=y_col, degree=degree, return_x_eval=True)
    y_pred = model.get_prediction(x_eval)
    # Note that: model.get_prediction(x_eval).predicted_mean == model.predict(x_eval)

    # Create the plot
    fs = 14  # fontsize (x|y labels) TODO: should scale with dpi.
    ds = 14  # dot size (scatter plot) TODO: should scale with dpi.

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 8),
        dpi=dpi,
        num=f"{y_col} ~ {x_col} | {degree}-order polynomial",
    )

    # Draw the polynomial fit: confidence interval
    ci_kws = {} if ci_kws is None else ci_kws
    ci_kws["color"] = ci_kws.pop("color", "red")
    ax.fill_between(
        x_eval[:, 1],
        y_pred.predicted_mean - n_std * y_pred.se_mean,
        y_pred.predicted_mean + n_std * y_pred.se_mean,
        alpha=0.3,
        **ci_kws,
    )
    # The confidence interval will lay behind the scatterplot, on the foreground, however, will be the fitted mean line

    # Draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws
    scatter_kws["s"] = scatter_kws.pop("s", ds)  # dot size
    # no filling of the dots, keep only the edge
    scatter_kws["facecolors"] = scatter_kws.pop("facecolors", "none")
    scatter_kws["edgecolors"] = scatter_kws.pop("edgecolors", "dodgerblue")
    alpha = scatter_kws.pop("alpha", 0.5)
    ax.scatter(df[x_col], df[y_col], alpha=alpha, **scatter_kws)

    # Draw the polynomial fit: mean line
    line_kws = {} if line_kws is None else line_kws
    line_label = line_kws.pop("label", f"{degree}-order poly")
    line_label = f"{line_label} | R^2={model.rsquared:.3f}"
    line_kws["color"] = line_kws.pop("color", "red")
    _ = ax.plot(x_eval[:, 1], y_pred.predicted_mean, label=line_label, **line_kws)  # h = _

    # Finalise the plot
    ax.set_xlabel(x_col, fontsize=fs)
    ax.set_ylabel(y_col, fontsize=fs)
    ax.legend()
    plt.tight_layout()

    if verbose:
        print(model.summary())

    return model, fig, ax


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
