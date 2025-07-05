import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def boxplots_plotter(
        df: pd.DataFrame,
        n_cols: int=2,
        figs_size: tuple[int, int]=(15, 5)):
    """
    Plot boxplot for each column in a DataFrame using a grid layout.

    This function creates a grid of subplots, with each subplot displaying a boxplot
    for one feature (column) from the input DataFrame. The grid layout is determined
    by the number of columns specified, and unused subplots are removed if the number
    of features is not a multiple of `n_cols`. The function is particularly useful for
    visualizing the distribution and variability of multiple features in a structured,
    comparative format.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the features to plot. Each column will be visualized
        as a separate boxplot.
    n_cols : int, optional
        The number of columns in the subplot grid (default is 2).
    figs_size : tuple of (float, float), optional
        The size of each row in the figure, specified as (width, height) in inches.
        The number of rows scales the total figure height (default is (15, 5)).

    Returns
    -------
    None
        - This function displays the generated boxplots and does not return any value.

    Example
    --------
    >> boxplots_plotter(df, n_cols=2, figs_size=(12, 4))
    """
    n_features = len(df.columns)
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figs_size[0], figs_size[1]*n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(df.columns):
        sns.boxplot(data=df, y=feature, ax=axes[i])
        axes[i].set_title(f"Boxplot of {feature}")
        axes[i].set_ylabel("Values")
        axes[i].set_xlabel("")

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boxplots", fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
