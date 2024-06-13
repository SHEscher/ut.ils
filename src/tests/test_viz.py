# !/usr/bin/env python3
"""
Tests for viz.py in the `ut.ils` package.

Run me via the shell:

    pytest . --cov; coverage html; open src/tests/coverage_html_report/index.html

"""

# %% Import
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from matplotlib import pyplot as plt
from ut.viz import model_summary_publication_ready, plot_poly_fit, poly_model_fit

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.fixture(scope="session")
def temp_dataframe():
    """Create temp dir and file."""
    temp_df = pd.DataFrame(columns=["x", "y"])
    n = 500
    temp_df["x"] = np.linspace(-10, 10, n)
    temp_df["y"] = (
        1.7 + 2.14 * temp_df["x"] + 3.6 * temp_df["x"] ** 2 - 0.46 * temp_df["x"] ** 3 + np.random.normal(0, 64.5, n)  # noqa: NPY002
    )

    yield temp_df

    # Tear down
    ...
    # Remove temp dir
    ...


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_model_summary_publication_ready(capsys, temp_dataframe):
    """Test model_summary_publication_ready()."""
    for degree in [1, 2, 3]:
        poly_model = poly_model_fit(df=temp_dataframe, x_col="x", y_col="y", degree=degree)
        summary = model_summary_publication_ready(model=poly_model)
        out, _ = capsys.readouterr()
        assert f"({degree}-order polynomial fit" in out
        assert "AIC=" in out
        assert "BIC=" in out
        for i in range(degree + 1):
            assert f"b{i}=" in out
            assert f"SE{i}=" in out
            assert f"t{i}=" in out
            assert f"p{i}<" in out
        assert isinstance(summary, str)
        assert "R2=" in summary


def test_poly_model_fit(temp_dataframe):
    """Test poly_model_fit()."""
    for degree in [1, 2, 3]:
        poly_model = poly_model_fit(df=temp_dataframe, x_col="x", y_col="y", degree=degree, return_x_eval=False)
        assert isinstance(poly_model, sm.regression.linear_model.RegressionResultsWrapper)
        assert hasattr(poly_model, "rsquared")
        assert len(poly_model.params) == degree + 1

    # Test return_x_eval=True
    for degree in [1, 2, 3]:
        poly_model, x_eval = poly_model_fit(df=temp_dataframe, x_col="x", y_col="y", degree=degree, return_x_eval=True)
        assert len(poly_model.params) == degree + 1
        assert x_eval.shape[1] == len(poly_model.params)
        assert x_eval.shape[0] == 2_000
        assert temp_dataframe.x.min() == x_eval[0, 1]
        assert temp_dataframe.x.max() == x_eval[-1, 1]

    # Test n_x_eval
    for n_x_eval in [500, 1_250]:
        poly_model, x_eval = poly_model_fit(
            df=temp_dataframe, x_col="x", y_col="y", degree=degree, return_x_eval=True, n_x_eval=n_x_eval
        )
        assert x_eval.shape[0] == n_x_eval


def test_plot_poly_fit(capsys, temp_dataframe):
    """Test plot_poly_fit()."""
    poly_model, fig, ax = plot_poly_fit(
        df=temp_dataframe,
        x_col="x",
        y_col="y",
        degree=3,
        n_std=2,
        scatter_kws=None,
        line_kws=None,
        ci_kws=None,
        dpi=150,
        verbose=True,
    )
    out, _ = capsys.readouterr()
    assert " OLS Regression Results" in out
    assert isinstance(poly_model, sm.regression.linear_model.RegressionResultsWrapper)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
