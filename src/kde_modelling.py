from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def fit_kde_model(data: pd.Series, bandwidth: float) -> KernelDensity:
    """
    Fit a KDE model for the given data.
    """
    reshaped_data = data.values.reshape(-1, 1)
    return KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reshaped_data)


def generate_kde_models(df: pd.DataFrame, assets: list, bandwidth: float, MIN_DATA_POINTS: int) -> dict:
    """
    Generate KDE models for volume and price differences of assets.
    """
    kde_models = {}
    
    for year, df_grouped in df.groupby("Year"):
        if len(df_grouped) < MIN_DATA_POINTS:
            continue

        asset_kdes = {}
        for asset in assets:
            if asset not in ["VIX", "IRX"]:
                volume = df_grouped[f"{asset}_Volume"]
                asset_kdes[asset] = fit_kde_model(volume, bandwidth)

            # Model the difference in asset open, close, and high
            deltas = {
                "delta_open": df_grouped[f"{asset}_Open"] - df_grouped[f"{asset}_Close"],
                "delta_high": df_grouped[f"{asset}_High"] - df_grouped[f"{asset}_Close"],
                "delta_low": df_grouped[f"{asset}_Low"] - df_grouped[f"{asset}_Close"]
            }

            for delta_name, delta_data in deltas.items():
                asset_kdes[f"{asset}_{delta_name}"] = fit_kde_model(delta_data, bandwidth)

        kde_models[year] = asset_kdes

    return kde_models


def plot_kde_subplot(ax, model: KernelDensity, data: pd.Series, title: str):
    """
    Plot the KDE on a given subplot axis against the actual data histogram.
    """
    # Generate sample data for KDE
    x = np.linspace(data.min(), data.max(), 1000)
    log_density = model.score_samples(x.reshape(-1, 1))
    
    # Plot on the given axis
    sns.histplot(data, kde=False, stat="density", bins=30, label='Actual Data', ax=ax)
    ax.plot(x, np.exp(log_density), '-r', label='KDE')
    ax.set_title(title)
    ax.legend()