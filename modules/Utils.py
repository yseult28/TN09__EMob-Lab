 
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

import mplcursors

import logging



# log
logger = logging.getLogger(__name__)


# inputs

# preprocessing

def process_time_variable(df, column="DEPARTURE"):
    """
    Extract the time (HH:MM:SS) from a string, converts in datetime.
    Round to the minute.


    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose column to convert.
    column : str
        Name of the column to comvert

    Returns
    -------
    pandas.DataFrame
        The transformed DataFrame
    """
    def replace_24_hour(t):
        if t.startswith("24:"):
            return "00:" + t.split(":", 1)[1]
        return t
    
    df[column] = df[column].apply(replace_24_hour)
    df[column] = df[column].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
    df[column] = pd.to_datetime(df[column], format='%H:%M:%S', errors='coerce')
    df[column] = df[column].dt.round("min")
    return df.copy()
    

def process_departure(df, column="DEPARTURE"):
    """
    Extrait l'heure (HH:MM:SS) d'une chaîne, convertit en datetime
    puis arrondit à la minute.

    Parameters
    ----------
    df : pandas.DataFrame
        Le dataframe contenant la colonne à transformer.
    column : str
        Nom de la colonne à nettoyer.

    Returns
    -------
    pandas.DataFrame
        Le dataframe avec la colonne transformée.
    """
    def replace_24_hour(t):
        if t.startswith("24:"):
            return "00:" + t.split(":", 1)[1]
        return t
    
    df[column] = df[column].apply(replace_24_hour)
    df[column] = df[column].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
    df[column] = pd.to_datetime(df[column], format='%H:%M:%S', errors='coerce')
    df[column] = df[column].dt.round("min")
    return df



def prepare_gdf(df, crs="EPSG:4326"):
    """
    Prepares GeoDataFrames for origins and destinations from a DataFrame containing WKT points.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'ORIGIN' and 'DESTINATION' columns as WKT strings.
    crs : str, optional
        Coordinate reference system for the output GeoDataFrames (default: "EPSG:4326").

    Returns
    -------
    ori_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the origins.
    dest_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the destinations.
    """

    # Convert WKT ORIGIN or Point geometry to X and Y
    if isinstance(df["ORIGIN"].iloc[0], str):  # cas WKT en string
        df[["ORIGIN_X", "ORIGIN_Y"]] = df["ORIGIN"].apply(
            lambda p: p.replace("POINT(", "").replace(")", "")
        ).str.split(' ', expand=True).astype(float)
    else:  # cas Shapely Point
        df["ORIGIN_X"] = df["ORIGIN"].apply(lambda p: p.x)
        df["ORIGIN_Y"] = df["ORIGIN"].apply(lambda p: p.y)


    if isinstance(df["DESTINATION"].iloc[0], str):  
        df[["DESTINATION_X", "DESTINATION_Y"]] = df["DESTINATION"].apply(
            lambda p: p.replace("POINT(", "").replace(")", "")
        ).str.split(' ', expand=True).astype(float)
    else:
        df["DESTINATION_X"] = df["DESTINATION"].apply(lambda p: p.x)
        df["DESTINATION_Y"] = df["DESTINATION"].apply(lambda p: p.y)


    # Convert all coordinates to float
    df[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]] = df[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]].astype(float)

    # Recreate Point objects for origin and destination
    df["ORIGIN"] = df.apply(lambda row: Point(row["ORIGIN_X"], row["ORIGIN_Y"]), axis=1)
    df["DESTINATION"] = df.apply(lambda row: Point(row["DESTINATION_X"], row["DESTINATION_Y"]), axis=1)

    # Drop temporary coordinate columns
    df.drop(columns=["ORIGIN_X", "ORIGIN_Y", "DESTINATION_X", "DESTINATION_Y"], inplace=True)

    # Create separate GeoDataFrames for origins and destinations
    ori_gdf = gpd.GeoDataFrame(df.drop(columns="DESTINATION"), geometry="ORIGIN", crs=crs)
    dest_gdf = gpd.GeoDataFrame(df.drop(columns="ORIGIN"), geometry="DESTINATION", crs=crs)

    return ori_gdf, dest_gdf


# distribution


def display_accumulation_time_distribution(csv_file, title, interval="1min"):
    """
    Displays the aggregated variation of demand from a CSV file with a bar plot.

    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file containing columns 'DEPARTURE' and 'ID'.
    title : str
        Title of the plot.
    interval : str
        Temporal aggregation interval, e.g. "1min", "5min", "15min", "30min", "1H".
    """

    # --- File checks ---
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise ValueError(f"File not found: {csv_file}")

    if not title or title.strip() == "":
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.")

    # --- Load CSV ---
    df = pd.read_csv(csv_file, sep=';')
    if "DEPARTURE" not in df.columns or "ID" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_file.name}")

    # --- Clean & transform 'DEPARTURE' ---
    df = process_departure(df, "DEPARTURE")

    # --- Apply temporal rounding ---
    try:
        df["DEPARTURE"] = df["DEPARTURE"].dt.floor(interval)
    except:
        raise ValueError(f"Invalid interval '{interval}'. Use a Pandas offset alias like '5min', '15min', '1H', etc.")

    # --- Aggregate ---
    df = df.groupby("DEPARTURE")["ID"].count().reset_index()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["DEPARTURE"], df["ID"],
           width=pd.Timedelta(interval), 
           align='center',
           color='steelblue',
           edgecolor='black')

    ax.set_xlabel("Time")
    ax.set_ylabel("Number of departures")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()



def display_origindestination_spatial_distribution__(csv_file, title, crs="EPSG:4326"):
    """
    Displays two maps side by side for a single CSV file:
    - Left: demand by origin
    - Right: demand by destination

    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file containing 'ORIGIN', 'DESTINATION', and 'ID' columns.
    crs : str
        Coordinate reference system of the input data (default EPSG:4326).
    """
    if not csv_file or not Path(csv_file).exists():
        raise ValueError("Invalid file path or file does not exist.")

    if not title or title.strip() == "":
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.")
    
    # Read data
    data = pd.read_csv(csv_file, sep=';')
    required_cols = {"ORIGIN", "DESTINATION", "ID"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Missing required columns in {csv_file}. Expected: {required_cols}")

    # Prepare GeoDataFrames for origin and destination
    ori_gdf, dest_gdf = prepare_gdf(data, crs=crs)

    # Aggregate counts
    ori_sum = ori_gdf.groupby("ORIGIN")["ID"].count().reset_index()
    dest_sum = dest_gdf.groupby("DESTINATION")["ID"].count().reset_index()

    # Replace geometries
    ori_sum_gdf = gpd.GeoDataFrame(ori_sum, geometry="ORIGIN", crs=crs)
    dest_sum_gdf = gpd.GeoDataFrame(dest_sum, geometry="DESTINATION", crs=crs)

    # Project to Web Mercator for plotting with basemap
    ori_sum_gdf = ori_sum_gdf.to_crs("EPSG:3857")
    dest_sum_gdf = dest_sum_gdf.to_crs("EPSG:3857")

    # Create side-by-side plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))

    # Origins map
    ori_sum_gdf.plot(ax=ax[0], color="red", alpha=1, markersize=ori_sum_gdf["ID"] / 5)
    ctx.add_basemap(ax[0], source=ctx.providers.OpenStreetMap.Mapnik)
    ax[0].axis("off")
    ax[0].set_title("Origin spatial distribution", fontsize=14)

    # Destinations map
    dest_sum_gdf.plot(ax=ax[1], color="blue", edgecolor="black", alpha=1, markersize=dest_sum_gdf["ID"] / 5)
    ctx.add_basemap(ax[1], source=ctx.providers.OpenStreetMap.Mapnik)
    ax[1].axis("off")
    ax[1].set_title("Destination spatial distribution", fontsize=14)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()






def display_origindestination_spatial_distribution_plotly(csv_file, title, crs="EPSG:4326"):
    """
    Displays two side-by-side interactive maps using Plotly with shared point-size scale.
    Hover shows aggregated demand.
    """

    # --- Load and validate file ---
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise ValueError("File not found.")

    data = pd.read_csv(csv_file, sep=';')

    if not {"ORIGIN", "DESTINATION", "ID"}.issubset(data.columns):
        raise ValueError("Missing columns ORIGIN, DESTINATION, ID")

    # --- Convert origin/destination WKT text to geometry ---
    ori_gdf, dest_gdf = prepare_gdf(data, crs=crs)

   # --- Aggregate origins ---
    ori_sum = ori_gdf.groupby("ORIGIN")["ID"].count().reset_index()
    ori_sum = gpd.GeoDataFrame(ori_sum, geometry="ORIGIN", crs=crs)  # <-- étape indispensable
    ori_sum = ori_sum.to_crs("EPSG:4326")
    ori_sum["lon"] = ori_sum.geometry.x
    ori_sum["lat"] = ori_sum.geometry.y

    # --- Aggregate destinations ---
    dest_sum = dest_gdf.groupby("DESTINATION")["ID"].count().reset_index()
    dest_sum = gpd.GeoDataFrame(dest_sum, geometry="DESTINATION", crs=crs)  # <-- idem
    dest_sum = dest_sum.to_crs("EPSG:4326")
    dest_sum["lon"] = dest_sum.geometry.x
    dest_sum["lat"] = dest_sum.geometry.y

    # --- Shared scale for marker size ---
    max_global = max(ori_sum["ID"].max(), dest_sum["ID"].max())

    def scale_size(x):
        return 5 + 25 * (x / max_global)

    ori_sizes = ori_sum["ID"].apply(scale_size)
    dest_sizes = dest_sum["ID"].apply(scale_size)

    # --- Map center ---
    center_lon = pd.concat([ori_sum["lon"], dest_sum["lon"]]).mean()
    center_lat = pd.concat([ori_sum["lat"], dest_sum["lat"]]).mean()

    # --- Create subplots ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Origin spatial distribution", "Destination spatial distribution"),
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]]
    )

    # Origins
    fig.add_trace(
        go.Scattermapbox(
            lon=ori_sum["lon"],
            lat=ori_sum["lat"],
            mode="markers",
            marker=dict(size=ori_sizes, color="red", opacity=0.8),
            customdata=ori_sum["ID"],
            hovertemplate="<b>Origin</b><br>Demand: %{customdata}<extra></extra>"
        ),
        row=1, col=1
    )

    # Destinations
    fig.add_trace(
        go.Scattermapbox(
            lon=dest_sum["lon"],
            lat=dest_sum["lat"],
            mode="markers",
            marker=dict(size=dest_sizes, color="blue", opacity=0.8),
            customdata=dest_sum["ID"],
            hovertemplate="<b>Destination</b><br>Demand: %{customdata}<extra></extra>"
        ),
        row=1, col=2
    )

    # Layout
    fig.update_layout(
        height=700,
        title_text=title,
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11),
        mapbox2=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11),
        showlegend=False
    )

    fig.show()


def display_origin_spatial_distribution(csv_file, title, crs="EPSG:32631", max_size_value=None):
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file, sep=';')
    ori_gdf, _ = prepare_gdf(df, crs=crs)

    # Agrégation par origine
    ori_sum = ori_gdf.groupby("ORIGIN")["ID"].count().reset_index()
    ori_sum = gpd.GeoDataFrame(ori_sum, geometry="ORIGIN", crs=crs)

    # Reprojection en WGS84 pour Plotly
    ori_sum = ori_sum.to_crs("EPSG:4326")
    ori_sum["lon"] = ori_sum.geometry.x
    ori_sum["lat"] = ori_sum.geometry.y

    # Taille des points
    max_val = max_size_value if max_size_value is not None else ori_sum["ID"].max()
    ori_sum["size"] = 5 + 25 * (ori_sum["ID"] / max_val)

    fig = go.Figure(go.Scattermapbox(
        lon=ori_sum["lon"],
        lat=ori_sum["lat"],
        mode="markers",
        marker=dict(size=ori_sum["size"], color="red", opacity=0.8,line=dict(width=1, color='black')),
        customdata=ori_sum["ID"],
        hovertemplate="<b>Origin</b><br>Demand: %{customdata}<extra></extra>"
    ))

    # Centrer la carte
    center_lon = ori_sum["lon"].mean()
    center_lat = ori_sum["lat"].mean()

    fig.update_layout(
        title=title,
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11, style="open-street-map"),
        height=700
    )
    fig.show()


def display_destination_spatial_distribution(csv_file, title, crs="EPSG:32631", max_size_value=None):
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file, sep=';')
    _, dest_gdf = prepare_gdf(df, crs=crs)

    # Agrégation par destination
    dest_sum = dest_gdf.groupby("DESTINATION")["ID"].count().reset_index()
    dest_sum = gpd.GeoDataFrame(dest_sum, geometry="DESTINATION", crs=crs)

    # Reprojection en WGS84
    dest_sum = dest_sum.to_crs("EPSG:4326")
    dest_sum["lon"] = dest_sum.geometry.x
    dest_sum["lat"] = dest_sum.geometry.y

    # Taille des points
    max_val = max_size_value if max_size_value is not None else dest_sum["ID"].max()
    dest_sum["size"] = 5 + 25 * (dest_sum["ID"] / max_val)

    fig = go.Figure(go.Scattermapbox(
        lon=dest_sum["lon"],
        lat=dest_sum["lat"],
        mode="markers",
        marker=dict(size=dest_sum["size"], color="blue", opacity=0.8, line=dict(width=1, color='black')),
        customdata=dest_sum["ID"],
        hovertemplate="<b>Destination</b><br>Demand: %{customdata}<extra></extra>"
    ))

    center_lon = dest_sum["lon"].mean()
    center_lat = dest_sum["lat"].mean()

    fig.update_layout(
        title=title,
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11, style="open-street-map"),
        height=700
    )
    fig.show()




def MFD_comparison(directory):
    """
    Load multiple flow.csv files located inside:
        directory / scenario_folder / subfolder / flow.csv

    Filter by VEHICLE_TYPE = CAR and plot:
        - ACCUMULATION vs TIME
        - SPEED vs TIME
        - TRIP_LENGTH vs TIME

    Legend name for each scenario is built from directory names:
        scenario_folder.split('__')[-1].split('_')[0:3]

    Interactive feature (mplcursors):
    Hover over a curve to display its label.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    import mplcursors

    # List top-level scenario directories
    scenario_dirs = [f.path for f in os.scandir(directory) if f.is_dir()]

    # === colormap large (256 couleurs distinctes) ===
    cmap = plt.cm.nipy_spectral

    # Create figure
    fig, (ax_acc, ax_speed) = plt.subplots(1, 2, figsize=(24, 8))

    handles = []
    labels = []

    for i, scenario_path in enumerate(scenario_dirs):

        # --- Label construction ---
        folder_name = os.path.basename(scenario_path)

        try:
            last_part = folder_name.split("__")[-1]
            parts = last_part.split("_")
            label = "_".join(parts[:3])  # keep exactly 3 components
        except:
            label = folder_name

        # --- Look for flow.csv in subfolders ---
        subfolders = [f.path for f in os.scandir(scenario_path) if f.is_dir()]

        flow_file = None
        for sub in subfolders:
            candidate = os.path.join(sub, "flow.csv")
            if os.path.isfile(candidate):
                flow_file = candidate
                break

        if flow_file is None:
            print(f"Warning: flow.csv not found in {scenario_path}")
            continue

        # Load CSV
        df = pd.read_csv(flow_file, sep=';')

        # Filter CAR
        df = df[df["VEHICLE_TYPE"] == "CAR"]
        if df.empty:
            print(f"Warning: no CAR data in {flow_file}")
            continue

        # Parse TIME
        df["TIME"] = df["TIME"].apply(
            lambda x: datetime.strptime(x, "%H:%M:%S.%f")
            if "." in x else datetime.strptime(x, "%H:%M:%S")
        )
        df = df.sort_values("TIME")

        # === couleur spécifique parmi 256 ===
        color = cmap(i / max(1, len(scenario_dirs)-1))

        # Plot metrics
        h = None
        if "ACCUMULATION" in df.columns:
            h, = ax_acc.plot(df["TIME"], df["ACCUMULATION"], color=color, label=label)

        if "SPEED" in df.columns:
            ax_speed.plot(df["TIME"], df["SPEED"], color=color, label=label)

        if h is not None:
            handles.append(h)
            labels.append(label)

    # Axis titles
    ax_acc.set_title("ACCUMULATION vs TIME")
    ax_acc.set_xlabel("Time")
    ax_acc.set_ylabel("Accumulation")

    ax_speed.set_title("SPEED vs TIME")
    ax_speed.set_xlabel("Time")
    ax_speed.set_ylabel("Speed")


    # === Légende sous les figures ===
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        fontsize=9
    )

    # === Interaction: survol = affichage automatique du label ===
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        line = sel.artist
        label = line.get_label()
        sel.annotation.set_text(label)
        sel.annotation.get_bbox_patch().set(alpha=0.9)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()





import pandas as pd
import plotly.graph_objs as go
from shapely.geometry import Point

def display_origin_time_distribution(csv_file, title, crs, interval="15min", max_size=20, zoom=11):
    df = pd.read_csv(csv_file, sep=';')

    # Extraction et conversion du temps
    ddf = process_departure(df, "DEPARTURE")
    df["DEPARTURE_ROUNDED"] = df["DEPARTURE"].dt.round(interval)

    origines_demande_originale, _ = prepare_gdf(df, "EPSG:32631")

    origines_demande_originale = origines_demande_originale.to_crs("epsg:4326")
    df["LON"] = origines_demande_originale["ORIGIN"].x
    df["LAT"] = origines_demande_originale["ORIGIN"].y

    # Agrégation
    agg_df = df.groupby(["DEPARTURE_ROUNDED", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
    agg_df["SIZE"] = (agg_df["ID"] / agg_df["ID"].max() * max_size).clip(3, max_size)

    # Trames temporelles
    time_values = sorted(agg_df["DEPARTURE_ROUNDED"].unique())
    frames = []

    for t in time_values:
        df_t = agg_df[agg_df["DEPARTURE_ROUNDED"] == t]
        frames.append(go.Frame(
            data=[go.Scattermap(
                lat=df_t["LAT"],
                lon=df_t["LON"],
                mode='markers',
                marker=dict(
                    size=df_t["SIZE"],
                    color="red",
                    opacity=0.7,
                    sizemode="area",
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=df_t.apply(lambda row: f"Demand: {row['ID']}", axis=1)
            )],
            name=str(t)
        ))

    # Figure initiale
    df_init = agg_df[agg_df["DEPARTURE_ROUNDED"] == time_values[0]]
    fig = go.Figure(
        data=[go.Scattermap(
            lat=df_init["LAT"],
            lon=df_init["LON"],
            mode='markers',
            marker=dict(
                size=df_init["SIZE"],
                color="red",
                opacity=0.7,
                sizemode="area"
            ),
            hoverinfo='text',
            hovertext=df_init.apply(lambda row: f"Demand: {row['ID']}", axis=1)
        )],
        layout=go.Layout(
            title=title,
            width=800,     
            height=800,       
            hovermode='closest',
            map=dict(
                center=dict(lat=agg_df["LAT"].mean(), lon=agg_df["LON"].mean()),
                zoom=zoom,
                style="streets"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.05,
                xanchor="left",
                yanchor="bottom",
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
            )],
            sliders=[dict(
                steps=[dict(method='animate', args=[[f.name], {"mode": "immediate", "frame": {"duration": 500, "redraw": True}, "transition": {"duration": 0}}], label=f.name) for f in frames],
                transition=dict(duration=0),
                x=0,
                y=0,
                currentvalue=dict(prefix="Heure: ", visible=True),
                len=1.0
            )]
        ),
        frames=frames
    )

    fig.show()


def display_destination_time_distribution(csv_file, title, crs, interval="15min", max_size=20, zoom=11):
    df = pd.read_csv(csv_file, sep=';')

    # Extraction et conversion du temps
    df = process_departure(df, "DEPARTURE")
    df["DEPARTURE_ROUNDED"] = df["DEPARTURE"].dt.round(interval)

    _, destinations_demande_originale = prepare_gdf(df, "EPSG:32631")

    destinations_demande_originale = destinations_demande_originale.to_crs("epsg:4326")
    df["LON"] = destinations_demande_originale["DESTINATION"].x
    df["LAT"] = destinations_demande_originale["DESTINATION"].y

    # Agrégation
    agg_df = df.groupby(["DEPARTURE_ROUNDED", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
    agg_df["SIZE"] = (agg_df["ID"] / agg_df["ID"].max() * max_size).clip(3, max_size)

    # Trames temporelles
    time_values = sorted(agg_df["DEPARTURE_ROUNDED"].unique())
    frames = []

    for t in time_values:
        df_t = agg_df[agg_df["DEPARTURE_ROUNDED"] == t]
        frames.append(go.Frame(
            data=[go.Scattermap(
                lat=df_t["LAT"],
                lon=df_t["LON"],
                mode='markers',
                marker=dict(
                    size=df_t["SIZE"],
                    color="blue",
                    opacity=0.7,
                    sizemode="area",
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=df_t.apply(lambda row: f"Demand: {row['ID']}", axis=1)
            )],
            name=str(t)
        ))

    # Figure initiale
    df_init = agg_df[agg_df["DEPARTURE_ROUNDED"] == time_values[0]]
    fig = go.Figure(
        data=[go.Scattermap(
            lat=df_init["LAT"],
            lon=df_init["LON"],
            mode='markers',
            marker=dict(
                size=df_init["SIZE"],
                color="blue",
                opacity=0.7,
                sizemode="area"
            ),
            hoverinfo='text',
            hovertext=df_init.apply(lambda row: f"Demand: {row['ID']}", axis=1)
        )],
        layout=go.Layout(
            title=title,
            width=800,     
            height=800,       
            hovermode='closest',
            map=dict(
                center=dict(lat=agg_df["LAT"].mean(), lon=agg_df["LON"].mean()),
                zoom=zoom,
                style="streets"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.05,
                xanchor="left",
                yanchor="bottom",
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
            )],
            sliders=[dict(
                steps=[dict(method='animate', args=[[f.name], {"mode": "immediate", "frame": {"duration": 500, "redraw": True}, "transition": {"duration": 0}}], label=f.name) for f in frames],
                transition=dict(duration=0),
                x=0,
                y=0,
                currentvalue=dict(prefix="Heure: ", visible=True),
                len=1.0
            )]
        ),
        frames=frames
    )

    fig.show()


def display_boxplot_departure(csv_file, title):
    """


    """

    if not title or title.strip() == "":
        logger.error("Invalid or null title.")
        raise ValueError("invalid or null title.")


    df = pd.read_csv(csv_file, sep=';')

    # Extraction et conversion du temps
    df = process_departure(df, "DEPARTURE")

    # Convertir en minutes depuis minuit
    df['MINUTES'] = df['DEPARTURE'].dt.hour * 60 + df['DEPARTURE'].dt.minute

    plt.figure(figsize=(8, 6))

    # Boxplot
    plt.boxplot(df['MINUTES'], vert=True)

    # Customiser l'axe y pour afficher HH:MM
    def minutes_to_hhmm(x, pos):
        h = int(x // 60)
        m = int(x % 60)
        return f"{h:02d}:{m:02d}"

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(minutes_to_hhmm))
    plt.ylabel("Heure")
    plt.title(title)
    plt.show()



# outputs

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def display_acc_speed_time_for_cars(outputs_dir):
    """
    Display the car accumulation and car speed by time on multiple plots for each unique reservoir value with flow.csv.
    Each floor corresponds to a value of the "RESERVOIR" column.
    
    Parameters
    ----------
    outputs_dir : str
        relative path to a given simulation outputs.
    """
    
    # Charger le fichier de flux
    flow_file = list(Path(outputs_dir).glob("flow.csv"))
    
    if len(flow_file) == 0: 
        logger.error("No flow.csv file for the given directory.")
        raise ValueError("No flow.csv file for the given directory.")
    
    flow_data = pd.read_csv(flow_file[0], sep=';')

    # Conversion du temps en format datetime
    flow_data = process_departure(flow_data, "TIME")

    # Trouver les valeurs uniques de RESERVOIR
    reservoirs = flow_data["RESERVOIR"].unique()

    # Créer une figure pour tous les sous-graphes
    fig, axes = plt.subplots(len(reservoirs), 2, figsize=(10, 6 * len(reservoirs)), sharex=True)
    if len(reservoirs) == 1:  # Si il y a un seul réservoir, axes sera un tableau 1D
        axes = np.expand_dims(axes, axis=0)

    # Boucler sur chaque réservoir
    for i, reservoir in enumerate(reservoirs):
        # Filtrer les données par réservoir
        flow_reservoir = flow_data[flow_data["RESERVOIR"] == reservoir]
        
        # Filtrer les voitures uniquement
        flow_car = flow_reservoir[flow_reservoir["VEHICLE_TYPE"] == "CAR"]
        
        # Générer une palette de couleurs unique pour chaque réservoir
        # Ici on génère 2 couleurs différentes pour chaque réservoir
        colors = plt.cm.get_cmap('tab20', 2)([0, 1])  # Choisir une palette de 2 couleurs distinctes
        
        # Graphique 1 : Accumulation des voitures
        axes[i, 0].plot(flow_car["TIME"], flow_car["ACCUMULATION"], color=colors[0])
        axes[i, 0].set_title(f"Accumulation des voitures - Réservoir {reservoir}")
        axes[i, 0].set_xlabel("Temps")
        axes[i, 0].set_ylabel("Accumulation")
        axes[i, 0].grid(True)
        axes[i, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Graphique 2 : Vitesse des voitures
        axes[i, 1].plot(flow_car["TIME"], flow_car["SPEED"], color=colors[1])
        axes[i, 1].set_title(f"Vitesse des voitures - Réservoir {reservoir}")
        axes[i, 1].set_xlabel("Temps")
        axes[i, 1].set_ylabel("Vitesse")
        axes[i, 1].grid(True)
        axes[i, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    
    # Ajuster l'affichage pour éviter le chevauchement
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


def display_cost_length_time(outputs_dir):
    """
    Display the mean travel cost and mean travel length by time on multiple plots with path.csv.

    
    Parameters
    ----------
    outputs_dir : str
        relative path to a given simulation outputs.
    """

     # Charger le fichier de flux
    path_file = list(Path(outputs_dir).glob("path.csv"))
    
    if len(path_file) == 0: 
        logger.error("No path.csv file for the given directory.")
        raise ValueError("No path.csv file for the given directory.")
    
    path_data = pd.read_csv(path_file[0], sep=';')

    # Conversion du temps en format datetime
    path_data = process_departure(path_data, "TIME")


    path_buffer = path_data.replace([np.inf, -np.inf], np.nan).dropna()
 
    path_cost = path_buffer.groupby("TIME").agg({"COST":"mean"}).reset_index()
    path_length = path_buffer.groupby("TIME").agg({"LENGTH":"mean"}).reset_index()

    colors = plt.cm.tab20(np.linspace(0,1,10))

    fig, ax = plt.subplots(2,2,figsize=(10,6),sharex=True)

    ax[0][0].plot(path_cost["TIME"],path_cost["COST"], color=colors[2])
    ax[0][0].set_title("cout des trajets en fonction du temps")
    ax[0][0].set_xlabel("temps")
    ax[0][0].set_ylabel("cout")
    ax[0][0].grid(True)
    ax[0][0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0][0].tick_params(axis='x', labelbottom=True)

    ax[0][1].plot(path_length["TIME"],path_length["LENGTH"], color=colors[9])
    ax[0][1].set_title("longueur des trajets en fonction du temps")
    ax[0][1].set_xlabel("temps")
    ax[0][1].set_ylabel("longueur")
    ax[0][1].grid(True)
    ax[0][1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0][1].tick_params(axis='x', labelbottom=True)

    ax[1][0].bar(path_cost["TIME"],path_cost["COST"], width=pd.Timedelta(minutes=10),align="edge",edgecolor='black', color="blue")
    ax[1][0].set_title("cout des trajets en fonction du temps")
    ax[1][0].set_xlabel("temps")
    ax[1][0].set_ylabel("cout")
    ax[1][0].grid(True)
    ax[1][0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1][0].tick_params(axis='x', labelbottom=True)

    ax[1][1].bar(path_length["TIME"],path_length["LENGTH"], width=pd.Timedelta(minutes=10),align="edge",edgecolor='black', color="red")
    ax[1][1].set_title("longueur des trajets en fonction du temps")
    ax[1][1].set_xlabel("temps")
    ax[1][1].set_ylabel("longueur")
    ax[1][1].grid(True)
    ax[1][1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1][1].tick_params(axis='x', labelbottom=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()



def display_costs_time(outputs_dir):
    """
    display costs by time on multiple subplots with travel_tile_linl.csv

    Parameters
    ----------
    outputs_dir : str
        relative path to a given simulation outputs.
    """


    # Charger le fichier de flux
    travel_time_link_file = list(Path(outputs_dir).glob("travel_time_link.csv"))
    
    if len(travel_time_link_file) == 0: 
        logger.error("No travel_time_link.csv file for the given directory.")
        raise ValueError("No travel_time_link.csv file for the given directory.")
    
    travel_time_link_data = pd.read_csv(travel_time_link_file[0], sep=';')

    # Conversion du temps en format datetime
    travel_time_link_data = process_departure(travel_time_link_data, "TIME")
    travel_time_link_data["TIME"] = travel_time_link_data["TIME"].dt.round("10min")


    travel_time_link_data["COSTS"] = travel_time_link_data["COSTS"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.strip() != "" else {}
    )

    buffer = travel_time_link_data.copy()

    buffer["TRAVEL_TIME"] = buffer["COSTS"].apply(lambda x: x.get("travel_time") if isinstance(x, dict) else None)
    buffer["SPEED"] = buffer["COSTS"].apply(lambda x: x.get("speed") if isinstance(x, dict) else None)
    buffer["LENGTH"] = buffer["COSTS"].apply(lambda x: x.get("length") if isinstance(x, dict) else None)

    car_buffer = buffer.loc[buffer["MOBILITY_SERVICE"]=="CAR"]

    car_buffer = car_buffer.groupby(["TIME"]).agg({"TRAVEL_TIME":"mean", "SPEED":"mean", "LENGTH":"mean"}).reset_index()

    colors = plt.cm.tab20(np.linspace(0,1,10))
    
    fix, ax = plt.subplots(2,3,figsize=(18,10),sharex=True)

    ax[0][0].plot(car_buffer["TIME"],car_buffer["TRAVEL_TIME"],color=colors[0])
    ax[0][0].set_title("temps de trajets moyens en fonction du temps")
    ax[0][0].set_xlabel("temps")
    ax[0][0].set_ylabel("temps de trajet")
    ax[0][0].grid(True)
    ax[0][0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[0][0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0][0].tick_params(axis='x', labelbottom=True)


    ax[0][1].plot(car_buffer["TIME"],car_buffer["SPEED"],color=colors[1])
    ax[0][1].set_title("vitesse moyenne en fonction du temps")
    ax[0][1].set_xlabel("temps")
    ax[0][1].set_ylabel("vitesse")
    ax[0][1].grid(True)
    ax[0][1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[0][1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0][1].tick_params(axis='x', labelbottom=True)

    ax[0][2].plot(car_buffer["TIME"],car_buffer["LENGTH"],color=colors[2])
    ax[0][2].set_title("longueur moyenne des trajets en fonction du temps")
    ax[0][2].set_xlabel("temps")
    ax[0][2].set_ylabel("longueur")
    ax[0][2].grid(True)
    ax[0][2].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[0][2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0][2].tick_params(axis='x', labelbottom=True)


    ax[1][0].bar(car_buffer["TIME"],car_buffer["TRAVEL_TIME"], width=pd.Timedelta(minutes=10),align="edge",edgecolor='black', color=colors[0])
    ax[1][0].set_title("temps de trajets moyens en fonction du temps")
    ax[1][0].set_xlabel("temps")
    ax[1][0].set_ylabel("temps de trajet")
    ax[1][0].grid(True)
    ax[1][0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[1][0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1][0].tick_params(axis='x', labelbottom=True)

    ax[1][1].bar(car_buffer["TIME"],car_buffer["SPEED"], width=pd.Timedelta(minutes=10),align="edge",edgecolor='black', color=colors[1])
    ax[1][1].set_title("vitesse moyenne en fonction du temps")
    ax[1][1].set_xlabel("temps")
    ax[1][1].set_ylabel("vitesse")
    ax[1][1].grid(True)
    ax[1][1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[1][1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1][1].tick_params(axis='x', labelbottom=True)

    ax[1][2].bar(car_buffer["TIME"],car_buffer["LENGTH"], width=pd.Timedelta(minutes=10),align="edge",edgecolor='black', color=colors[2])
    ax[1][2].set_title("longueur moyenne des trajets en fonction du temps en fonction du temps")
    ax[1][2].set_xlabel("temps")
    ax[1][2].set_ylabel("longueur")
    ax[1][2].grid(True)
    ax[1][2].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[1][2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1][2].tick_params(axis='x', labelbottom=True)

    plt.tight_layout()
    plt.show()


def display_acc_spatiotemporal(outputs_dir, title, interval="10min"):
    """
    Display the spatiotemporal car density distribution with user.csv

    Parameters
    ----------
    outputs_dir : str
        relative path to a given simulation outputs.
    interval : str
        time interval between each frame.
    """

    if not title or title.strip() == "":
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.")
        
    # Charger le fichier de flux
    user_file = list(Path(outputs_dir).glob("user.csv"))
    
    if len(user_file) == 0: 
        logger.error("No user.csv file for the given directory.")
        raise ValueError("No user.csv file for the given directory.")
    
    user_data = pd.read_csv(user_file[0], sep=';')

    # Conversion du temps en format datetime
    user_data = process_departure(user_data, "TIME")
    user_data["TIME"] = user_data["TIME"].dt.round(interval)

    buffer = user_data.copy()
    buffer[["POSITION_X","POSITION_Y"]] = buffer["POSITION"].str.split(' ', expand=True).astype(float)
    buffer["GEOMETRY"] = buffer.apply(lambda row: Point(row["POSITION_X"], row["POSITION_Y"]), axis=1)
    buffer.drop(columns=["POSITION_X","POSITION_Y"], inplace=True)
    crs = "EPSG:32631"
    buffer = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=crs)
    
    buffer = buffer.loc[buffer["STATE"]=="INSIDE_VEHICLE"]

    # changement du système de projection vers un système latitude/longitude
    buffer.to_crs("epsg:4326",inplace=True)

    buffer["LON"] = buffer["GEOMETRY"].x
    buffer["LAT"] = buffer["GEOMETRY"].y

    # Agrégation
    agg_df = buffer.groupby(["TIME", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
    agg_df["SIZE"] = agg_df["ID"]

    # Trames temporelles
    time_values = sorted(agg_df["TIME"].unique())
    frames = []

    for t in time_values:
        df_t = agg_df[agg_df["TIME"] == t]
        frames.append(go.Frame(
            data=[go.Scattermap(
                lat=df_t["LAT"],
                lon=df_t["LON"],
                mode='markers',
                marker=dict(
                    size=df_t["SIZE"],
                    color="blue",
                    opacity=0.7,
                    sizemode="area",
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=df_t.apply(lambda row: f"Demand: {row['ID']}", axis=1)
            )],
            name=str(t)
        ))

    # Figure initiale
    df_init = agg_df[agg_df["TIME"] == time_values[0]]
    fig = go.Figure(
        data=[go.Scattermap(
            lat=df_init["LAT"],
            lon=df_init["LON"],
            mode='markers',
            marker=dict(
                size=df_init["SIZE"],
                color="blue",
                opacity=0.7,
                sizemode="area"
            ),
            hoverinfo='text',
            hovertext=df_init.apply(lambda row: f"Demand: {row['ID']}", axis=1)
        )],
        layout=go.Layout(
            title=title,
            width=800,     
            height=800,       
            hovermode='closest',
            map=dict(
                center=dict(lat=agg_df["LAT"].mean(), lon=agg_df["LON"].mean()),
                zoom=11,
                style="streets"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.05,
                xanchor="left",
                yanchor="bottom",
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
            )],
            sliders=[dict(
                steps=[dict(method='animate', args=[[f.name], {"mode": "immediate", "frame": {"duration": 500, "redraw": True}, "transition": {"duration": 0}}], label=f.name) for f in frames],
                transition=dict(duration=0),
                x=0,
                y=0,
                currentvalue=dict(prefix="Heure: ", visible=True),
                len=1.0
            )]
        ),
        frames=frames
    )

    fig.show()


def display_density_spatiotemporal(outputs_dir, title, interval="10min", max_opacity=1, radius=15, blur=0.8):
    """
    Display the spatiotemporal car density distribution with user.csv

    Parameters
    ----------
    outputs_dir : str
        relative path to a given simulation outputs.
    interval : str
        time interval between each frame.
    """

    if not title or title.strip() == "":
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.")
        
    # Charger le fichier de flux
    user_file = list(Path(outputs_dir).glob("user.csv"))
    
    if len(user_file) == 0: 
        logger.error("No user.csv file for the given directory.")
        raise ValueError("No user.csv file for the given directory.")
    
    user_data = pd.read_csv(user_file[0], sep=';')

    # Conversion du temps en format datetime
    user_data = process_departure(user_data, "TIME")
    user_data["TIME"] = user_data["TIME"].dt.round(interval)

    buffer = user_data.copy()
    buffer[["POSITION_X","POSITION_Y"]] = buffer["POSITION"].str.split(' ', expand=True).astype(float)
    buffer["GEOMETRY"] = buffer.apply(lambda row: Point(row["POSITION_X"], row["POSITION_Y"]), axis=1)
    buffer.drop(columns=["POSITION_X","POSITION_Y"], inplace=True)
    crs = "EPSG:32631"
    buffer = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=crs)
    
    buffer = buffer.loc[buffer["STATE"]=="INSIDE_VEHICLE"]

    # changement du système de projection vers un système latitude/longitude
    buffer.to_crs("epsg:4326",inplace=True)


    buffer["TIME"] = buffer["TIME"].dt.round(interval)
    # regroupement des lignes par pas de temps et par coordonnées
    buffer = buffer.groupby(["TIME","GEOMETRY"])["ID"].count().reset_index()
    buffer = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs="epsg:4326")
    # formattage pour folium : heatmap et tooltip

    maxi = buffer["ID"].max()

    # formattage des données
    hm_point_data = []
    hm_time_data = []
    t_data = []

    #hm_point_data = buffer.apply(lambda row : [row["ORIGIN"].y,row["ORIGIN"].x ,row["ID"] ], axis=1)
    hm_time_data = buffer.apply(lambda row : row["TIME"].strftime("%H:%M"), axis=1)
    buffer["TIME"] = hm_time_data
    hm_time_data = list(set(hm_time_data))
    hm_time_data.sort()
    #hm_time_data = sorted(buffer["DEPARTURE"].unique())

    t_data = buffer.apply(lambda row : {"lat":row["GEOMETRY"].y, "lon":row["GEOMETRY"].x, "val":row["ID"]}, axis=1)

    x = 0


    for time in hm_time_data:
        points = []
        if(x > buffer.shape[0]-1) : break
        #print(time)
        while(buffer.loc[x,"TIME"]==time):
            #print(buffer.loc[x,"DEPARTURE"])
            val = float(buffer.loc[x,"ID"]/maxi)
            points.append([buffer.loc[x,"GEOMETRY"].y,buffer.loc[x,"GEOMETRY"].x,val])
            x+=1
            if(x > buffer.shape[0]-1) : break
        #print(points)
        hm_point_data.append(points)


    # construction de la carte
    mean_lat = buffer.geometry.y.mean()
    mean_lon = buffer.geometry.x.mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="cartodb positron")

    # Nouveau gradient : très détaillé dans les petites valeurs
    gradient = {
        0.000: 'white',
        0.001: 'azure',
        0.002: 'lightcyan',
        0.003: 'paleturquoise',
        0.004: 'powderblue',
        0.005: 'lightblue',
        0.006: 'skyblue',
        0.007: 'lightskyblue',
        0.008: 'greenyellow',
        0.009: 'yellowgreen',
        0.010: 'yellow',
        0.1: 'gold',
        0.2: 'orange',
        0.3: 'darkorange',
        0.4: 'orangered',
        0.5: 'red',
        0.6: 'firebrick',
        0.7: 'darkred',
        0.8: 'maroon',
        0.9: 'purple',
        1.0: 'black'
    }

    colormap = LinearColormap(
        colors=[
            'white', 'azure', 'lightcyan', 'paleturquoise', 'powderblue',
            'lightblue', 'skyblue', 'lightskyblue', 'greenyellow',
            'yellowgreen', 'yellow',  # jusqu'à 0.01
            'gold', 'orange', 'darkorange', 'orangered',
            'red', 'firebrick', 'darkred', 'maroon', 'purple', 'black'
        ],
        vmin=0,
        vmax=maxi,
        caption="Quantité réelle"
    )
    colormap.add_to(m)


    # construction de la heatmap
    HeatMapWithTime(
        hm_point_data,
        index=hm_time_data,
        gradient=gradient,
        max_opacity=1,
        radius=15,
        blur=0.8

    ).add_to(m)

    return m


def display_acc_spatiotemporal_with_veh(outputs_dir, title, interval="10min"):
    """
    Display the spatiotemporal car density distribution with veh.csv

    Parameters
    ----------
    outputs_dir : str
        relative path to a given simulation outputs.
    interval : str
        time interval between each frame.
    """

    if not title or title.strip() == "":
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.")
        
    # Charger le fichier de flux
    veh_file = list(Path(outputs_dir).glob("veh.csv"))
    
    if len(veh_file) == 0: 
        logger.error("No veh.csv file for the given directory.")
        raise ValueError("No veh.csv file for the given directory.")
    
    veh_data = pd.read_csv(veh_file[0], sep=';')

    # Conversion du temps en format datetime
    veh_data =  process_departure(veh_data, "TIME")
    veh_data["TIME"] = veh_data["TIME"].dt.round(interval)

    buffer = veh_data.copy()
    buffer = buffer.loc[buffer["TYPE"]=="Car"]
    buffer[["POSITION_X","POSITION_Y"]] = buffer["POSITION"].str.split(' ', expand=True).astype(float)
    buffer["GEOMETRY"] = buffer.apply(lambda row: Point(row["POSITION_X"], row["POSITION_Y"]), axis=1)
    buffer.drop(columns=["POSITION_X","POSITION_Y"], inplace=True)
    crs = "EPSG:32631"
    buffer = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=crs)
    

    # changement du système de projection vers un système latitude/longitude
    buffer.to_crs("epsg:4326",inplace=True)

    buffer["LON"] = buffer["GEOMETRY"].x
    buffer["LAT"] = buffer["GEOMETRY"].y

    # Agrégation
    agg_df = buffer.groupby(["TIME", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
    agg_df["SIZE"] = agg_df["ID"]

    # Trames temporelles
    time_values = sorted(agg_df["TIME"].unique())
    frames = []

    for t in time_values:
        df_t = agg_df[agg_df["TIME"] == t]
        frames.append(go.Frame(
            data=[go.Scattermap(
                lat=df_t["LAT"],
                lon=df_t["LON"],
                mode='markers',
                marker=dict(
                    size=df_t["SIZE"],
                    color="blue",
                    opacity=0.7,
                    sizemode="area",
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=df_t.apply(lambda row: f"Demand: {row['ID']}", axis=1)
            )],
            name=str(t)
        ))

    # Figure initiale
    df_init = agg_df[agg_df["TIME"] == time_values[0]]
    fig = go.Figure(
        data=[go.Scattermap(
            lat=df_init["LAT"],
            lon=df_init["LON"],
            mode='markers',
            marker=dict(
                size=df_init["SIZE"],
                color="blue",
                opacity=0.7,
                sizemode="area"
            ),
            hoverinfo='text',
            hovertext=df_init.apply(lambda row: f"Demand: {row['ID']}", axis=1)
        )],
        layout=go.Layout(
            title=title,
            width=800,     
            height=800,       
            hovermode='closest',
            map=dict(
                center=dict(lat=agg_df["LAT"].mean(), lon=agg_df["LON"].mean()),
                zoom=11,
                style="streets"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.05,
                xanchor="left",
                yanchor="bottom",
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
            )],
            sliders=[dict(
                steps=[dict(method='animate', args=[[f.name], {"mode": "immediate", "frame": {"duration": 500, "redraw": True}, "transition": {"duration": 0}}], label=f.name) for f in frames],
                transition=dict(duration=0),
                x=0,
                y=0,
                currentvalue=dict(prefix="Heure: ", visible=True),
                len=1.0
            )]
        ),
        frames=frames
    )

    fig.show()

def display_car_speed_distance_time(directory="", title="", interval="10min"):
    """
    Displays car speed and distanceversus time on two figures with an output file in the given directory and a time aggregation at the given
    interval.

    Parameters
    ----------
    directory : str
        Directory to find the output file.
    title : str
        Title of the figure.
    interval : str
        Time interval for aggregation.
    """

    # check
    if not directory or directory.strip() == "" : 
        logger.info("Null or invalid directory.")
        raise ValueError("Null or invalid directory.")
    if not directory or directory.strip() == "" : 
        logger.info("Null or invalid directory.")
        raise ValueError("Null or invalid directory.")

    # load veh.csv path
    veh_file = list(Path(directory).glob("veh.csv"))
    
    if len(veh_file) == 0: 
        logger.error("No veh.csv file for the given directory.")
        raise ValueError("No veh.csv file for the given directory.")
    
    veh_data = pd.read_csv(veh_file[0], sep=';')


    veh_data = process_departure(veh_data, "TIME")
    veh_data = veh_data.loc[veh_data["TYPE"]=="Car"]
    veh_data["TIME"] = veh_data["TIME"].dt.round(interval)

    veh_data = (
            veh_data.groupby(["TIME","ID"])  
              .agg(
                SPEED=("SPEED", "mean"),

                DISTANCE=("DISTANCE", lambda x: x.iloc[-1] - x.iloc[0]),
              ).reset_index()
            )

     # convert interval to seconds
    interval_seconds = pd.Timedelta(interval).total_seconds()
    logger.info(interval_seconds)

    # --- compute SPEED from DISTANCE / interval_seconds ---
    veh_data["SPEED_DISTANCE"] = veh_data["DISTANCE"] / interval_seconds
    veh_data = veh_data.groupby("TIME").agg({"SPEED":"mean", "DISTANCE":"mean", "SPEED_DISTANCE":"mean"}).reset_index()

   

    fig, ax = plt.subplots(1,2,figsize=(10,6))

    ax[0].bar(veh_data["TIME"], veh_data["SPEED"],
           width=pd.Timedelta(interval), 
           align='center',
           color='steelblue',
           edgecolor='black')

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Speed")
    ax[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].grid(True)

    ax[1].bar(veh_data["TIME"], veh_data["SPEED_DISTANCE"],
           width=pd.Timedelta(interval), 
           align='center',
           color='red',
           edgecolor='black')

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Speed-Distance")
    ax[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].grid(True)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    



    




def display_demand_variation(path="",title="",labels=[],cols=0):
    """
    Displays demand's variations on a subplots with barplot.
    
    Parameters
    ----------
    path : string
        Path to the demand's variations.
    title : string
        figure's title.
    labels : list
        subtitle of each barplot.
    cols : int, optional
        number of barplot per line.
    """
    if not path or str(path).strip() == "": 
        logger.error("Invalid or null path.")
        raise ValueError("Invalid or null path.")
    if not title or str(title).strip() == "": 
        logger.error("Invalid or null title.")
        raise ValueError("Invalid or null title.") 
    
    # looks for visible csv files
    files = [p for p in Path(path).glob("*.csv") if not p.name.startswith(".")]
    files.sort()

    n = len(files)

    if len(labels) != n: 
        logger.error("Invalid labels.")
        raise ValueError("Invalid labels.")

    if cols == 0 or cols > n : 
        logger.error("Invalid number of columns.")
        raise ValueError("Invalid number of columns.")
        
    rows = math.ceil(n / cols)

    # loads the df
    variations = []
    for file in files:
        buffer = pd.read_csv(file, sep=';')
        if "DEPARTURE" not in buffer.columns or "ID" not in buffer.columns:
            raise ValueError(f"Missing required columns in file {file.name}")
        buffer =  process_departure(buffer, "DEPARTURE")
        buffer = buffer.drop(columns=[c for c in ["ORIGIN", "DESTINATION"] if c in buffer.columns])
        buffer = buffer.dropna(subset=["DEPARTURE"])
        buffer["DEPARTURE"] = buffer["DEPARTURE"].dt.round("min")
        buffer = buffer.groupby("DEPARTURE").count().reset_index()
        variations.append(buffer)

    # maximum accumulation for the given variations
    max_acc = max(variation['ID'].max() for variation in variations)

    # Palette de couleurs par défaut
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False, constrained_layout=True)

    for i, variation in enumerate(variations):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        color = colors[i % len(colors)]

        ax.bar(variation["DEPARTURE"], variation["ID"], 
               width=pd.Timedelta(minutes=1), align='center', color=color)
        ax.set_title(labels[i])
        ax.set_xlabel("Temps")
        ax.set_ylabel("Nombre de départs")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True)
        ax.set_ylim(0, max_acc)

    # removes the unused graphs
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])
        
    plt.suptitle(title, fontsize=14)

    

    plt.show()
    

def display_demand_variation_auto(path="", cols=0):
    """
    Displays demand's variations on subplots with barplot.
    Titles and labels are automatically generated from filenames
    following the format loi__evenements.
    If loi or evenements is None, 'None' is displayed.
    
    Parameters
    ----------
    path : str
        Path to the folder containing CSV variations.
    cols : int, optional
        Number of subplots per row (default auto).
    """
    if not path or str(path).strip() == "": 
        raise ValueError("Invalid or null path.") 
    
    files = [p for p in Path(path).glob("*.csv") if not p.name.startswith(".")]
    files.sort()
    n = len(files)
    if n == 0:
        raise ValueError("No CSV files found in the given path.")

    def parse_section(section):
        """Parse a section: remplace - par ,"""
        if section is None:
            return "None"
        parts = section.split("-")
        return ",".join(parts) if parts else "None"

    def filename_to_label(name):
        """Transforme le nom de fichier en label"""
        name = name.replace(".csv", "")
        parts = name.split("__")
        loi = parts[0] if len(parts) > 0 else "None"
        evenements = parts[1] if len(parts) > 1 else "None"

        loi_parts = [parse_section(s) for s in loi.split("_")]
        event_parts = [parse_section(s) for s in evenements.split("_")]

        return "_".join(loi_parts + event_parts)

    # Crée labels
    labels = [filename_to_label(f.name) for f in files]
    
    # Crée titre général basé sur le premier fichier
    first_file = files[0].name.replace(".csv", "")
    parts = first_file.split("__")
    loi_part = parts[0] if len(parts) > 0 else "None"

    # récupère uniquement le premier élément de la loi (type)
    title = loi_part.split("_")[0] if loi_part else "None"

    # Organisation des subplots
    if cols <= 0 or cols > n:
        cols = min(3, n)
    rows = math.ceil(n / cols)

    # charge les dataframes
    variations = []
    for file in files:
        buffer = pd.read_csv(file, sep=';')
        if "DEPARTURE" not in buffer.columns or "ID" not in buffer.columns:
            raise ValueError(f"Missing required columns in file {file.name}")
        buffer =  process_departure(buffer, "DEPARTURE")
        buffer = buffer.drop(columns=[c for c in ["ORIGIN", "DESTINATION"] if c in buffer.columns])
        buffer = buffer.dropna(subset=["DEPARTURE"])
        buffer["DEPARTURE"] = buffer["DEPARTURE"].dt.round("min")
        buffer = buffer.groupby("DEPARTURE").count().reset_index()
        variations.append(buffer)

    max_acc = max(variation['ID'].max() for variation in variations)
    colors = plt.cm.tab10.colors
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False, constrained_layout=True)

    for i, variation in enumerate(variations):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        color = colors[i % len(colors)]

        ax.bar(variation["DEPARTURE"], variation["ID"], 
                width=pd.Timedelta(minutes=1), align='center', color=color)
        ax.set_title(labels[i],fontsize=10)
        ax.set_xlabel("Temps")
        ax.set_ylabel("Nombre de départs")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True)
        ax.set_ylim(0, max_acc)

    # supprime les axes inutilisés
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])
        
    plt.suptitle(title, fontsize=14)
    plt.show()


def display_naive_sensitivity(title="", df=[], TARGET=""):
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

        










# brouillons








import pandas as pd
import geopandas as gpd
import numpy as np
import math, re
from pathlib import Path
from shapely.geometry import Point
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


def display_spatiotemporal_maps(
    path, title, labels, cols, interval="10min", crs="EPSG:4326", size_scale=50
):
    """
    Affiche un tableau interactif de cartes spatio-temporelles animées.
    Chaque sous-carte représente un scénario (dossier) différent,
    et les points indiquent les positions agrégées dans le temps.

    Les tailles sont proportionnelles au nombre d'occurrences (ID),
    avec contour noir pour la lisibilité.

    Arguments :
    - path : dossier contenant les sous-dossiers avec veh.csv
    - title : titre global de la figure
    - labels : titres des sous-cartes
    - cols : nombre de colonnes dans la grille
    - interval : intervalle de regroupement temporel (ex: '10min')
    - crs : système de coordonnées initial (ex: 'EPSG:4326')
    - size_scale : facteur d'échelle pour la taille des points
    """

    path = Path(path)
    if not path.exists():
        raise ValueError("Path invalide ou inexistant.")
    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("Labels doit être une liste non vide.")

    folders = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(folders) != len(labels):
        raise ValueError("Le nombre de labels doit correspondre au nombre de sous-dossiers.")

    # --- Couleurs par dossier ---
    cmap = plt.get_cmap("tab10")
    colors = [to_hex(cmap(i % cmap.N)) for i in range(len(folders))]

    gdfs, max_count = [], 0
    for folder in folders:
        file_path = folder / "veh.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} manquant.")

        df = pd.read_csv(file_path, sep=";")

        if "POSITION" not in df.columns or "TIME" not in df.columns:
            raise ValueError(f"{file_path} doit contenir 'POSITION' et 'TIME'.")

        # --- Nettoyage des positions ---
        def is_valid_position(pos):
            if pd.isna(pos) or str(pos).strip() == "":
                return False
            return bool(re.match(r"^\s*-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s*$", str(pos)))

        df = df[df["POSITION"].apply(is_valid_position)]

        # --- Gestion du temps ---
        df["TIME"] = pd.to_datetime(
            df["TIME"].str.extract(r"(\d{2}:\d{2}:\d{2})")[0],
            format="%H:%M:%S",
            errors="coerce",
        )
        df = df.dropna(subset=["TIME"])
        df["TIME"] = df["TIME"].dt.round(interval)

        # --- Coordonnées ---
        df[["POSITION_X", "POSITION_Y"]] = df["POSITION"].str.split(" ", expand=True).astype(float)

        # --- GeoDataFrame ---
        gdf = gpd.GeoDataFrame(
            df,
            geometry=df.apply(lambda r: Point(r["POSITION_X"], r["POSITION_Y"]), axis=1),
            crs=crs,
        ).to_crs("EPSG:4326")

        gdf["LAT"] = gdf.geometry.y
        gdf["LON"] = gdf.geometry.x

        # --- Agrégation par temps et position ---
        gdf = gdf.groupby(["TIME", "LAT", "LON"]).agg({"ID": "count"}).reset_index()

        max_count = max(max_count, gdf["ID"].max())
        gdfs.append(gdf)

    #return gdfs

    # --- Création des sous-cartes ---
    rows = math.ceil(len(folders) / cols)
    specs = [[{"type": "scattermapbox"} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=labels)

    # --- Traces vides de base ---
    for i in range(len(gdfs)):
        row, col = divmod(i, cols)
        fig.add_trace(
            go.Scattermapbox(
                lat=[], lon=[], mode="markers",
                marker=dict(
                    size=[],
                    color=colors[i],
                    opacity=0.6,
                    sizemode="area"
                ),
                hoverinfo="skip", showlegend=False,
                subplot=f"mapbox{i+1}"
            ),
            row=row + 1, col=col + 1,
        )

        fig.update_layout({
            f"mapbox{i+1}": dict(
                style="open-street-map",
                zoom=8.5,
                center=dict(lat=gdfs[i]["LAT"].mean(), lon=gdfs[i]["LON"].mean()),
            )
        })

    # --- Construction des frames d’animation ---
    all_times = sorted(set(t for gdf in gdfs for t in gdf["TIME"]))
    frames = []

    for t in all_times:
        frame_data = []
        for i, gdf in enumerate(gdfs):
            df_t = gdf[gdf["TIME"] == t]
            print(f"{i}e gdf : {len(df_t)}")
            if df_t.empty:
                frame_data.append(go.Scattermapbox(lat=[], lon=[]))
                continue

            marker_size = (np.log1p(df_t["ID"]) / np.log1p(max_count)) * size_scale

            frame_data.append(go.Scattermapbox(
                lat=df_t["LAT"],
                lon=df_t["LON"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=colors[i],
                    opacity=0.6,
                    sizemode="area",
                ),
                hoverinfo="text",
                hovertext=df_t.apply(
                    lambda r: f"Comptage: {r['ID']}<br>Lat: {r['LAT']:.4f}<br>Lon: {r['LON']:.4f}",
                    axis=1
                ),
                showlegend=False,
                subplot=f"mapbox{i+1}"
            ))
        frames.append(go.Frame(data=frame_data, name=str(t.time()),traces=list(range(len(gdfs)))))

    fig.frames = frames

    # --- Layout et animation ---
    fig.update_layout(
        height=400 * rows,
        width=400 * cols,
        title=dict(text=title, x=0.5),
        margin=dict(r=0, l=0, b=0, t=40),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f.name], {"mode": "immediate", "frame": {"duration": 500, "redraw": True}}],
                        label=f.name,
                    )
                    for f in frames
                ],
                currentvalue=dict(prefix="Heure: ", visible=True, xanchor="center"),
            )
        ],
    )

    output_path = f"{title.replace(' ', '_')}_animated.html"
    pio.write_html(fig, file=output_path, auto_open=True)
    fig.show()


    

def display_spatiotemporal_maps_color(path, title, labels, cols, interval="10min", crs="EPSG:4326", cmap_name="viridis"):
    """
    Affiche un tableau interactif de cartes spatio-temporelles
    où la couleur des points représente le nombre d'occurrences (ID).

    Arguments :
    - path : dossier contenant les sous-dossiers avec veh.csv
    - title : titre global de la figure
    - labels : titres des sous-cartes
    - cols : nombre de colonnes dans la grille
    - interval : intervalle de regroupement temporel (ex: '10min')
    - crs : système de coordonnées initial
    - cmap_name : nom de la palette matplotlib (ex: 'viridis', 'plasma', 'inferno')
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Path invalide ou inexistant.")

    folders = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(folders) != len(labels):
        raise ValueError("Le nombre de labels doit correspondre au nombre de sous-dossiers.")

    rows = math.ceil(len(folders) / cols)
    cmap = plt.get_cmap(cmap_name)

    # --- Chargement et traitement des fichiers ---
    gdfs, max_count = [], 0
    for folder in folders:
        df = pd.read_csv(folder / "veh.csv", sep=';')
        df = df[df["POSITION"].notna()]
        df["TIME"] = pd.to_datetime(
            df["TIME"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0],
            format='%H:%M:%S', errors='coerce'
        ).dt.round(interval)
        df = df.dropna(subset=["TIME"])
        df[["POSITION_X", "POSITION_Y"]] = df["POSITION"].str.split(' ', expand=True).astype(float)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=df.apply(lambda r: Point(r["POSITION_X"], r["POSITION_Y"]), axis=1),
            crs=crs
        ).to_crs("EPSG:4326")
        gdf["LAT"], gdf["LON"] = gdf.geometry.y, gdf.geometry.x

        # Agrégation par temps et position
        gdf = gdf.groupby(["TIME", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
        max_count = max(max_count, gdf["ID"].max())
        gdfs.append(gdf)

    # --- Valeurs uniques de temps ---
    time_values = sorted(set(t for gdf in gdfs for t in gdf["TIME"].dt.strftime("%H:%M:%S").unique()))
    specs = [[{"type": "scattermapbox"} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=labels)

    # --- Traces initiales (vide pour animation) ---
    for i in range(len(gdfs)):
        row, col = divmod(i, cols)
        fig.add_trace(
            go.Scattermapbox(
                lat=[],
                lon=[],
                mode='markers',
                marker=dict(
                    size=4,
                    color=[],
                    opacity=0.8,
                    colorscale=cmap_name,
                    cmin=0,
                    cmax=max_count,
                    colorbar=dict(title="Comptes") if i == len(gdfs) - 1 else None
                ),
                hoverinfo='skip',
                showlegend=False,
                subplot=f"mapbox{i+1}"
            ),
            row=row + 1, col=col + 1
        )
        fig.update_layout({
            f"mapbox{i+1}": dict(
                style="open-street-map",
                zoom=8.5,
                center=dict(lat=gdfs[i]["LAT"].mean(), lon=gdfs[i]["LON"].mean())
            )
        })

    # --- Frames pour l'animation ---
    frames = []
    for t in time_values:
        frame_data = []
        for i, gdf in enumerate(gdfs):
            df_t = gdf[gdf["TIME"].dt.strftime("%H:%M:%S") == t]
            frame_data.append(
                go.Scattermapbox(
                    lat=df_t["LAT"],
                    lon=df_t["LON"],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=df_t["ID"],
                        opacity=0.8,
                        colorscale=cmap_name,
                        cmin=0,
                        cmax=max_count
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    subplot=f"mapbox{i+1}"  # 🔹 assignation explicite
                )
            )
        frames.append(go.Frame(data=frame_data, name=t))
    fig.frames = frames

    # --- Layout global ---
    fig.update_layout(
        height=400 * rows,
        width=400 * cols,
        title=dict(text=title, x=0.5),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f.name], {"mode": "immediate", "frame": {"duration": 500, "redraw": True}}],
                label=f.name
            ) for f in frames],
            currentvalue=dict(prefix="Heure: ", visible=True)
        )]
    )

    # --- Sauvegarde et affichage ---
    output_path = f"{title.replace(' ', '_')}_color.html"
    pio.write_html(fig, file=output_path, auto_open=True)
    fig.show()




def speed_analysis_proto(outputs=[]):
    """


    """
    # check
    if len(outputs) == 0 : 
        logger.info("Null outputs.")
        raise ValueError("Null outputs.")

    labels = outputs
    veh_paths = []
    for output in outputs : 
        veh_paths.append(Path(output) / "veh.csv")

    fig, ax = plt.subplots(1,2,figsize=(13,13))

    cmap = plt.get_cmap("tab10")
    colors = [to_hex(cmap(i % cmap.N)) for i in range(len(outputs))]
    i = 0
    for veh_path in veh_paths :
        veh = pd.read_csv(veh_path, sep=';')
        veh = veh[["TIME","ID","TYPE","POSITION","DISTANCE","SPEED"]]
        veh = process_time_variable(veh, "TIME")
        veh = veh.sort_values("TIME") 
        veh["SPEED"] = pd.to_numeric(veh["SPEED"], errors="coerce")
        veh["DISTANCE"] = pd.to_numeric(veh["DISTANCE"], errors="coerce")
        
        # temporal aggregation
        
        veh = (
            veh.groupby(["ID", "TYPE"])
        .agg(
            SPEED_MEAN=("SPEED", "mean"),

            DISTANCE=("DISTANCE", lambda x: x.iloc[-1] - x.iloc[0]),

            POSITION=("POSITION", lambda x: " ".join(x.astype(str))),

            TIME=("TIME", lambda x: x.min()),

            PERIOD=("TIME", lambda x: x.max() - x.min())
            )
            .reset_index()
        )
        veh["PERIOD"] = veh["PERIOD"].dt.total_seconds() / 3600
        
        # DISTANCE in meters, SPEED_MEAN in meters per second, conversion to km and km/h
        veh["DISTANCE"] = veh["DISTANCE"]/1000
        veh["SPEED_MEAN"] = veh["SPEED_MEAN"] * 3.6

        # speed calculation with DISTANCE_DIFF method
        veh["DISTANCE_DIFF"] = veh["DISTANCE"]/veh["PERIOD"]

        veh = veh.groupby("TIME").agg({"SPEED_MEAN":"mean", "DISTANCE_DIFF":"mean"}).reset_index()
        
        ax[0].plot(veh["TIME"], veh["SPEED_MEAN"], color=colors[i], marker='o', linestyle='solid',
     linewidth=1, markersize=1, label = labels[i])

        ax[1].plot(veh["TIME"], veh["DISTANCE_DIFF"], color=colors[i], marker='o', linestyle='solid',
     linewidth=1, markersize=1, label = labels[i])

        i += 1




    ax[0].set_xlabel("Departure")
    ax[0].set_ylabel("MFD mean speed")
    ax[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].grid(True)

    ax[1].set_xlabel("Departure")
    ax[1].set_ylabel("MnMS mean speed")
    ax[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].grid(True)
    
    fig.legend(loc="upper center", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()





    