
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



def display_naive_sensitivity_with_Kansey(title="", df=[], TARGET=""):
    """
    Display df in a a Sankey diagram with title.

    Parameters
    ----------
    title : string
        The figure's title.
    df : pandas dataframe
        The dataframe to display.
    TARGET : string
        The target variable to display, the weight of the diagram. The values must be positive.
    """

    #check
    if not title or title.strip()=="":
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.")
    if df.empty:
        logger.error("The dataframe is empty.")
        raise ValueError("The dataframe is empty.")
    if not TARGET in df.columns:
        logger.error("Invalid target variable.")
        raise ValueError("Invalid target variable.")

    labels = []
    sources, targets, values = [], [], []

    for _, row in df.iterrows():
        elements = [
            str(row["TYPE"]) if str(row["TYPE"]) not in ("None", "nan") else "None_TYPE",
            str(row["PARAMETERS"]) if str(row["PARAMETERS"]) not in ("None", "nan") else "None_PARAMETERS",
            str(row["RATIO"]) if str(row["RATIO"]) not in ("None", "nan") else "None_RATIO",
            str(row["START"]) if str(row["START"]) not in ("None", "nan") else "None_START",
            str(row["END"]) if str(row["END"]) not in ("None", "nan") else "None_END",
            str(row["EVENTS"]) if str(row["EVENTS"]) not in ("None", "nan") else "None_EVENTS"
        ]

        # relier tous les niveaux (même si None)
        for i in range(len(elements) - 1):
            src = elements[i]
            tgt = elements[i + 1]
            if src not in labels:
                labels.append(src)
            if tgt not in labels:
                labels.append(tgt)
            sources.append(labels.index(src))
            targets.append(labels.index(tgt))
            values.append(float(row[TARGET]))


    # If the differences are small, processing

    # log and factor
    values_log = np.log1p(values * 1000)  # log1p pour gérer valeurs proches de 0


    # normalization and exponential
    #values_norm = (values - values.min()) / (values.max() - values.min())
    #values_scaled = values_norm * 5  # ajuste l'épaisseur maximale
    #values = values_scaled

    norm = mcolors.Normalize(vmin=min(values_log), vmax=max(values))
    cmap = cm.get_cmap("coolwarm")
    link_colors = [mcolors.to_hex(cmap(norm(v))) for v in values]

    # Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20, color="lightblue"),
        link=dict(source=sources, target=targets, value=values, color=link_colors, hovertemplate='%{source.label} → %{target.label}<br>Valeur: %{value:.5f}<extra></extra>')
    )])

    #values_arr = np.array(values)
    legend_values = [min(values), (min(values) + max(values))/2, max(values)]
    legend_colors = [mcolors.to_hex(cmap(norm(v))) for v in legend_values]

    for val, col in zip(legend_values, legend_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # points invisibles
            mode='markers',
            marker=dict(size=10, color=col),
            showlegend=True,
            name=f"{val:.2f}"  # affichage de la valeur dans la légende
        ))
    
    fig.update_layout(title_text=title, font_size=10)
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    fig.show()


import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_naive_sensitivity_with_Kansey_dual(title="", df=None, TARGET1="", TARGET2=""):
    """
    Affiche deux diagrammes Sankey empilés verticalement :
      - Le premier basé sur TARGET1
      - Le second basé sur TARGET2
      Les deux partagent la même échelle de couleurs.
    """

    if not title or title.strip() == "":
        raise ValueError("Invalid or null title.")
    if df is None or df.empty:
        raise ValueError("The dataframe is empty.")
    for target in [TARGET1, TARGET2]:
        if target not in df.columns:
            raise ValueError(f"'{target}' n'existe pas dans le DataFrame.")

    # === Calcul de l'échelle globale ========================================
    all_values = np.concatenate([df[TARGET1].to_numpy(), df[TARGET2].to_numpy()])
    min_val, max_val = all_values.min(), all_values.max()

    cmap = cm.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=np.log1p(min_val * 1000), vmax=np.log1p(max_val * 1000))

    def build_sankey_data(df, target_col):
        """Construit les structures sources/targets/values/labels pour un target donné."""
        labels = []
        sources, targets, values = [], [], []

        for _, row in df.iterrows():
            elements = [
                str(row["TYPE"]) if str(row["TYPE"]) not in ("None", "nan") else "None_TYPE",
                str(row["PARAMETERS"]) if str(row["PARAMETERS"]) not in ("None", "nan") else "None_PARAMETERS",
                str(row["RATIO"]) if str(row["RATIO"]) not in ("None", "nan") else "None_RATIO",
                str(row["START"]) if str(row["START"]) not in ("None", "nan") else "None_START",
                str(row["END"]) if str(row["END"]) not in ("None", "nan") else "None_END",
                str(row["EVENTS"]) if str(row["EVENTS"]) not in ("None", "nan") else "None_EVENTS"
            ]

            for i in range(len(elements) - 1):
                src, tgt = elements[i], elements[i + 1]
                if src not in labels:
                    labels.append(src)
                if tgt not in labels:
                    labels.append(tgt)
                sources.append(labels.index(src))
                targets.append(labels.index(tgt))
                values.append(float(row[target_col]))

        link_colors = [mcolors.to_hex(cmap(norm(np.log1p(v * 1000)))) for v in values]

        sankey_dict = dict(
            node=dict(label=labels, pad=15, thickness=20, color="lightblue"),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate='%{source.label} → %{target.label}<br>Valeur: %{value:.5f}<extra></extra>'
            )
        )
        return sankey_dict

    # === Création des deux jeux de données ==================================
    sankey_1 = build_sankey_data(df, TARGET1)
    sankey_2 = build_sankey_data(df, TARGET2)

    # === Création de la figure plotly =======================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        subplot_titles=[f"{TARGET1}", f"{TARGET2}"],
        vertical_spacing=0.15,
        specs=[[{"type": "domain"}], [{"type": "domain"}]]
    )

    fig.add_trace(go.Sankey(**sankey_1), row=1, col=1)
    fig.add_trace(go.Sankey(**sankey_2), row=2, col=1)

    # === Légende unique basée sur l'échelle globale ==========================
    legend_values = [min_val, (min_val + max_val)/2, max_val]
    legend_colors = [mcolors.to_hex(cmap(norm(np.log1p(v * 1000)))) for v in legend_values]

    for val, col in zip(legend_values, legend_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=col),
            showlegend=True,
            name=f"{val:.2f}"
        ))

    # === Mise en page ========================================================
    fig.update_layout(
        title_text=title,
        font_size=10,
        height=900,
        legend=dict(
            title="Valeurs",
            orientation="h",
            yanchor="bottom", y=-0.1,
            xanchor="center", x=0.5
        )
    )
    fig.show()




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
    required_cols = ["RATIO", "PARAMETERS", "TYPE", "START", "END"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le DataFrame.")

    if y_col not in df.columns:
        raise ValueError(f"La colonne '{y_col}' n'existe pas dans le DataFrame.")

    # 🧩 Création des combinaisons
    df = df.copy()
    df["COLOR_KEY"] = df["TYPE"].astype(str)  + "_" + df["PARAMETERS"].astype(str)
    df["STYLE_KEY"] = df["START"].astype(str) + "_" + df["END"].astype(str)

    # 🎨 Création du graphique
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="RATIO",
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
    required_cols = ["RATIO", "PARAMETERS", "TYPE", "START", "END"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le DataFrame.")

    for target in [TARGET1, TARGET2]:
        if target not in df.columns:
            raise ValueError(f"La colonne '{target}' n'existe pas dans le DataFrame.")

    # 🧩 Création des combinaisons
    df = df.copy()
    df["COLOR_KEY"] = df["TYPE"].astype(str)  + "_" + df["PARAMETERS"].astype(str)
    df["STYLE_KEY"] = df["START"].astype(str) + "_" + df["END"].astype(str)

    # 🎨 Création des graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    
    # Premier scatterplot
    sns.scatterplot(
        data=df,
        x="RATIO",
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
        x="RATIO",
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
