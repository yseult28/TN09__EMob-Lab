
# dependencies

import os

from pathlib import Path

import re

import datetime as dt

import math
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import geopandas as gpd

import seaborn as sns

from shapely.geometry import Point

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import logging

# log
logger = logging.getLogger(__name__)






def plot_ratio_vs(title, df, y_col):
    """
    Affiche un nuage de points où :
      - la couleur correspond à la combinaison (PARAMETERS, TYPE)
      - la forme du marqueur correspond à la combinaison (START, END)

    Paramètres
    ----------
    title : str
        Titre du graphique.
    df : pandas.DataFrame
        Doit contenir les colonnes : 'RATIO', y_col, 'PARAMETERS', 'TYPE', 'START', 'END'.
    y_col : str
        Nom de la colonne à afficher en ordonnée.
    """

    # Vérifications minimales
    required_cols = ["TYPE","LAW","TOTAL_RATIO"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le DataFrame.")

    if y_col not in df.columns:
        raise ValueError(f"La colonne '{y_col}' n'existe pas dans le DataFrame.")

    # 🧩 Création des combinaisons
    df = df.copy()
    df["COLOR_KEY"] = df[cols_from_11].astype(str).agg("_".join, axis=1)
    df["STYLE_KEY"] = df["TYPE"].astype(str) + "-" + df["LAW"].astype(str)

    # 🎨 Création du graphique
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="TOTAL_RATIO",
        y=y_col,
        hue="COLOR_KEY",     # couleur unique pour chaque combinaison PARAMETRE+TYPE
        style="STYLE_KEY",   # forme unique pour chaque combinaison START+END
        palette="tab20",     # palette large de couleurs
        s=80,
        alpha=0.85,
        edgecolor="black"
    )

    # 🧾 Mise en forme
    plt.title(f"{title} : Ratio vs {y_col}", fontsize=14)
    plt.xlabel("Ratio")
    plt.ylabel(y_col)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Combinaisons")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_ratio_vs_dual(title, df, TARGET1, TARGET2):
    """
    Affiche deux nuages de points côte à côte où :
      - la couleur correspond à la combinaison (PARAMETERS, TYPE)
      - la forme du marqueur correspond à la combinaison (START, END)

    Paramètres
    ----------
    title : str
        Titre du graphique.
    df : pandas.DataFrame
        Doit contenir les colonnes : 'RATIO', TARGET1, TARGET2, 'PARAMETERS', 'TYPE', 'START', 'END'.
    TARGET1 : str
        Nom de la première colonne à afficher en ordonnée.
    TARGET2 : str
        Nom de la deuxième colonne à afficher en ordonnée.
    """

    # Vérifications minimales
    required_cols = ["TYPE","LAW","TOTAL_RATIO"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le DataFrame.")

    for target in [TARGET1, TARGET2]:
        if target not in df.columns:
            raise ValueError(f"La colonne '{target}' n'existe pas dans le DataFrame.")

    # 🧩 Création des combinaisons
    df = df.copy()
    cols_from_11 = df.columns[10:]
    df["COLOR_KEY"] = df[cols_from_11].astype(str).agg("_".join, axis=1)
    df["STYLE_KEY"] = df["TYPE"].astype(str) + "-" + df["LAW"].astype(str)

    # 🎨 Création des graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    
    # Premier scatterplot
    sns.scatterplot(
        data=df,
        x="TOTAL_RATIO",
        y=TARGET1,
        hue="COLOR_KEY",
        style="STYLE_KEY",
        palette="tab20",
        s=80,
        alpha=0.85,
        edgecolor="black",
        ax=axes[0]
    )
    #axes[0].set_title(f"{title} : Ratio vs {TARGET1}", fontsize=14)
    axes[0].set_xlabel("Ratio")
    axes[0].set_ylabel(TARGET1)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Deuxième scatterplot
    sns.scatterplot(
        data=df,
        x="TOTAL_RATIO",
        y=TARGET2,
        hue="COLOR_KEY",
        style="STYLE_KEY",
        palette="tab20",
        s=80,
        alpha=0.85,
        edgecolor="black",
        ax=axes[1]
    )
    #axes[1].set_title(f"{title} : Ratio vs {TARGET2}", fontsize=14)
    axes[1].set_xlabel("Ratio")
    axes[1].set_ylabel(TARGET2)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Déplacer légende en dehors du plot droit
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.05, 1), loc="upper left", title="Combinaisons")

    plt.suptitle(f"{title} : ratio vs {TARGET1} and {TARGET2}")
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ratio_by_cluster(df, TARGET, n_couples_par_ligne=2):

    if TARGET not in df.columns:
        raise ValueError(f"La colonne '{TARGET}' n'existe pas dans le DataFrame.")
    if "TARGET_CLUSTER" not in df.columns:
        raise ValueError("Le DataFrame doit contenir la colonne 'TARGET_CLUSTER'.")

    clusters = sorted([c for c in df["TARGET_CLUSTER"].unique() if c != -1])
    n_clusters = len(clusters)

    # STYLE seulement
    df = df.copy()
    df["STYLE_KEY"] = df["TYPE"].astype(str) + "-" + df["LAW"].astype(str)

    y_min, y_max = df[TARGET].min(), df[TARGET].max()

    # Calcul du nombre de lignes nécessaires
    n_rows = math.ceil(n_clusters / n_couples_par_ligne)

    # Création de la figure
    fig = plt.figure(figsize=(5 * n_couples_par_ligne, 5 * n_rows))
    width_ratios = [1, 3] * n_couples_par_ligne
    gs = fig.add_gridspec(n_rows, n_couples_par_ligne * 2,
                          width_ratios=width_ratios,
                          wspace=1, hspace=0.6)

    for idx, cluster in enumerate(clusters):
        row = idx // n_couples_par_ligne
        col = (idx % n_couples_par_ligne) * 2  # colonne de la légende

        # Plot principal
        ax = fig.add_subplot(gs[row, col + 1])
        subset = df[(df["TARGET_CLUSTER"] == cluster) | (df["TARGET_CLUSTER"] == -1)]

        # Convertir le cluster en catégorie
        df["TARGET_RATIO"] = df["TARGET_RATIO"].astype(str)

        # Palette discrète
        palette = sns.color_palette("tab20", df["TARGET_RATIO"].nunique())


        sns.scatterplot(
            data=subset,
            x="TOTAL_RATIO",
            y=TARGET,
            hue="TARGET_RATIO",      # devient CATEGORIEL !
            style="STYLE_KEY",
            palette=palette,            # adaptée au catégoriel
            alpha=0.85,
            s=10,
            edgecolor="black",
            ax=ax
        )

        if ax.legend_:
            ax.legend_.remove()

        ax.set_title(f"Cluster {cluster}", fontsize=11, fontweight="bold")
        ax.set_ylim(y_min - 0.01, y_max + 0.01)
        ax.grid(True, linestyle="--", alpha=0.25)

        # Légende à gauche
        legend_ax = fig.add_subplot(gs[row, col])
        legend_ax.axis("off")

        # Légende filtrée pour TARGET (valeurs numériques)
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l.replace('.', '', 1).isdigit()]
        if filtered:
            handles, labels = zip(*filtered)
            legend_ax.legend(
                handles, labels,
                title="TARGET_RATIO",
                loc="center right",
                fontsize=7,
                frameon=True
            )

    plt.show()


def plot_ratio_by_ratio(df, TARGET, n_couples_par_ligne=2):

    if TARGET not in df.columns:
        raise ValueError(f"La colonne '{TARGET}' n'existe pas dans le DataFrame.")
    if "TARGET_RATIO" not in df.columns:
        raise ValueError("Le DataFrame doit contenir la colonne 'TARGET_CLUSTER'.")

    ratios = sorted([c for c in df["TARGET_RATIO"].unique() if c != -1])
    n_ratios = len(ratios)

    # STYLE seulement
    df = df.copy()
    df["STYLE_KEY"] = df["TYPE"].astype(str) + "-" + df["LAW"].astype(str)

    y_min, y_max = df[TARGET].min(), df[TARGET].max()

    # Calcul du nombre de lignes nécessaires
    n_rows = math.ceil(n_ratios / n_couples_par_ligne)

    # Création de la figure
    fig = plt.figure(figsize=(5 * n_couples_par_ligne, 5 * n_rows))
    width_ratios = [1, 3] * n_couples_par_ligne
    gs = fig.add_gridspec(n_rows, n_couples_par_ligne * 2,
                          width_ratios=width_ratios,
                          wspace=1, hspace=0.6)

    for idx, ratio in enumerate(ratios):
        row = idx // n_couples_par_ligne
        col = (idx % n_couples_par_ligne) * 2  # colonne de la légende

        # Plot principal
        ax = fig.add_subplot(gs[row, col + 1])
        subset = df[(df["TARGET_RATIO"] == ratio) | (df["TARGET_RATIO"] == -1)]


        # Convertir le cluster en catégorie
        df["TARGET_CLUSTER"] = df["TARGET_CLUSTER"].astype(str)

        # Palette discrète
        palette = sns.color_palette("tab20", df["TARGET_CLUSTER"].nunique())


        sns.scatterplot(
            data=subset,
            x="TOTAL_RATIO",
            y=TARGET,
            hue="TARGET_CLUSTER",      # devient CATEGORIEL !
            style="STYLE_KEY",
            palette=palette,            # adaptée au catégoriel
            alpha=0.85,
            s=10,
            edgecolor="black",
            ax=ax
        )


        if ax.legend_:
            ax.legend_.remove()

        ax.set_title(f"ratio {ratio}", fontsize=11, fontweight="bold")
        ax.set_ylim(y_min - 0.01, y_max + 0.01)
        ax.grid(True, linestyle="--", alpha=0.25)

        # Légende à gauche
        legend_ax = fig.add_subplot(gs[row, col])
        legend_ax.axis("off")

        # Légende filtrée pour TARGET (valeurs numériques)
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l.replace('.', '', 1).isdigit()]
        if filtered:
            handles, labels = zip(*filtered)
            legend_ax.legend(
                handles, labels,
                title="TARGET_CLUSTER",
                loc="center right",
                fontsize=7,
                frameon=True
            )

    plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_boxplot_dual(title, df, TARGET1, TARGET2):
    """
    Affiche deux boxplots côte à côte (alignés verticalement sur la même échelle Y)
    pour TARGET1 et TARGET2.
      - Couleurs arbitraires
      - Valeurs aberrantes = croix rouges (sans légende)
      - Moyenne = losange noir
      - Axes alignés pour comparaison directe
    """

    # Vérifications
    for target in [TARGET1, TARGET2]:
        if target not in df.columns:
            raise ValueError(f"La colonne '{target}' n'existe pas dans le DataFrame.")

    # Couleurs arbitraires
    palette = sns.color_palette("Set2", n_colors=2)

    # Figure avec partage de l’axe Y (pour alignement vertical)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # --- Premier boxplot ---
    sns.boxplot(
        data=df,
        y=TARGET1,
        color=palette[0],
        flierprops=dict(marker="x", color="red", markersize=6),
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=6),
        ax=axes[0]
    )
    axes[0].set_title(f"{TARGET1}", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Valeur", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_xlabel("")

    # --- Deuxième boxplot ---
    sns.boxplot(
        data=df,
        y=TARGET2,
        color=palette[1],
        flierprops=dict(marker="x", color="red", markersize=6),
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=6),
        ax=axes[1]
    )
    axes[1].set_title(f"{TARGET2}", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")  # pas besoin de répéter le label
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xlabel("")

    # Harmonisation manuelle du range vertical pour être sûr
    y_min = min(df[TARGET1].min(), df[TARGET2].min())
    y_max = max(df[TARGET1].max(), df[TARGET2].max())
    for ax in axes:
        ax.set_ylim(y_min-0.01, y_max+0.01)

    # Titre global
    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()



def plot_boxplot_by_cluster(title, df, TARGET):
    """
    Affiche un boxplot pour chaque valeur unique de la colonne 'TARGET_CLUSTER',
    en incluant toujours les lignes où TARGET_CLUSTER == -1.
    
    Chaque cluster a son sous-graph, automatiquement organisé en grille.
    """
    if TARGET not in df.columns:
        raise ValueError(f"La colonne '{TARGET}' n'existe pas dans le DataFrame.")
    if "TARGET_CLUSTER" not in df.columns:
        raise ValueError("Le DataFrame doit contenir la colonne 'TARGET_CLUSTER'.")

    clusters = sorted([c for c in df["TARGET_CLUSTER"].unique() if c != -1])
    n_clusters = len(clusters)

    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)  # pour itérer facilement même si axes 1D

    # Limites globales pour harmonisation des axes Y
    y_min = df[TARGET].min()
    y_max = df[TARGET].max()

    palette = sns.color_palette("Set2", n_colors=n_clusters)

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        subset = df[(df["TARGET_CLUSTER"] == cluster) | (df["TARGET_CLUSTER"] == -1)]
        sns.boxplot(
            data=subset,
            y=TARGET,
            color=palette[i],
            flierprops=dict(marker="x", color="red", markersize=6),
            showmeans=True,
            meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=6),
            ax=ax
        )
        ax.set_title(f"Cluster {cluster}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("")
        ax.set_ylabel(TARGET)
        ax.set_ylim(y_min-0.01, y_max+0.01)

    # Supprimer les axes vides si clusters < n_rows*n_cols
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()



def plot_boxplot_by_ratio(title, df, TARGET):
    """
    Affiche un boxplot pour chaque valeur unique de la colonne 'TARGET_CLUSTER',
    en incluant toujours les lignes où TARGET_CLUSTER == -1.
    
    Chaque cluster a son sous-graph, automatiquement organisé en grille.
    """
    if TARGET not in df.columns:
        raise ValueError(f"La colonne '{TARGET}' n'existe pas dans le DataFrame.")
    if "TARGET_CLUSTER" not in df.columns:
        raise ValueError("Le DataFrame doit contenir la colonne 'TARGET_CLUSTER'.")

    ratios = sorted([c for c in df["TARGET_RATIO"].unique() if c != -1])
    n_ratios = len(ratios)

    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)  # pour itérer facilement même si axes 1D

    # Limites globales pour harmonisation des axes Y
    y_min = df[TARGET].min()
    y_max = df[TARGET].max()

    palette = sns.color_palette("Set2", n_colors=n_clusters)

    for i, ratio in enumerate(ratios):
        ax = axes[i]
        subset = df[(df["TARGET_RATIO"] == ratio) | (df["TARGET_RATIO"] == -1)]
        sns.boxplot(
            data=subset,
            y=TARGET,
            color=palette[i],
            flierprops=dict(marker="x", color="red", markersize=6),
            showmeans=True,
            meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=6),
            ax=ax
        )
        ax.set_title(f"Ratio {ratio}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("")
        ax.set_ylabel(TARGET)
        ax.set_ylim(y_min-0.01, y_max+0.01)

    # Supprimer les axes vides si clusters < n_rows*n_cols
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
