
# dependencies

import os

from pathlib import Path

import re

import ast

import datetime as dt

from branca.colormap import LinearColormap

import math
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

import contextily as ctx

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import folium
from folium.plugins import HeatMapWithTime

import logging



# log
logger = logging.getLogger(__name__)



def plot_target_vs_total_ratio(df, target):
    """
    Plot target / TOTAL_RATIO point.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with target and TOTAL_RATIO columns.
    target : str
        Target variable.
    """

    # checks
    if df is None or df.empty:
        raise ValueError("Invalid df.")
    if "TOTAL_RATIO" not in df.columns:
        raise ValueError("DataFrame must contain 'TOTAL_RATIO'.")
    if not target or target.strip() == "" or target not in df.columns:
        raise ValueError("Invalid or null target.")

    df = df.copy()
    df["TOTAL_RATIO"] = df["TOTAL_RATIO"].astype(float)

    # Scatter plot
    plt.figure(figsize=(8,6))

    # Utiliser les couleurs différentes pour chaque TOTAL_RATIO
    unique_ratios = sorted(df["TOTAL_RATIO"].unique())
    colors = plt.cm.viridis(np.linspace(0,1,len(unique_ratios)))

    for ratio, color in zip(unique_ratios, colors):
        subset = df[df["TOTAL_RATIO"] == ratio]
        plt.scatter(subset["TOTAL_RATIO"], subset[target], color=color, label=f"TOTAL_RATIO={ratio}", s=50)

    plt.xlabel("TOTAL_RATIO")
    plt.ylabel(target)
    plt.title(f"{target} vs TOTAL_RATIO")
    plt.legend(
        title="TOTAL_RATIO",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_target_vs_ratio_by_cluster(df, target):
    """
    Plot TARGET vs TARGET_RATIO, colored by TARGET_CLUSTER.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with TARGET_RATIO, TARGET_CLUSTER and target columns.
    target : str
        Target variable.
    """

    # checks
    if df is None or df.empty:
        raise ValueError("Invalid df.")
    if "TARGET_CHANGE" not in df.columns:
        raise ValueError("DataFrame must contain 'TARGET_CHANGE'.")
    if "TARGET_CLUSTER" not in df.columns:
        raise ValueError("DataFrame must contain 'TARGET_CLUSTER'.")
    if not target or target.strip() == "" or target not in df.columns:
        raise ValueError("Invalid or null target.")

    df = df.copy()
    df["TARGET_CHANGE"] = df["TARGET_CHANGE"].astype(float)

    # Scatter plot
    plt.figure(figsize=(8, 6))

    # couleurs par cluster
    unique_clusters = sorted(df["TARGET_CLUSTER"].unique())
    logger.info(unique_clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    for cluster, color in zip(unique_clusters, colors):
        subset = df[df["TARGET_CLUSTER"] == cluster]
        plt.scatter(
            subset["TARGET_CHANGE"],
            subset[target],
            color=color,
            label=f"TARGET_CLUSTER={cluster}",
            s=50,
            alpha=0.6
        )

    plt.xlabel("TARGET_CHANGE")
    plt.ylabel(target)
    plt.title(f"{target} vs TARGET_CHANGE by TARGET_CLUSTER")
    plt.legend(
        title="TARGET_CLUSTER",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()












        