
# dependencies

import os

from pathlib import Path

import re

import datetime as dt

import math
from matplotlib.colors import to_hex

import branca.colormap as cm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

import contextily as ctx

import folium
from folium.plugins import HeatMapWithTime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import logging



# logging
logger = logging.getLogger(__name__)


# inputs methods for one file

# diagram

def display_demand_variation_auto_one(csv_file):
    """
    Displays the variation of demand from a single CSV file with a bar plot.
    Title and labels are automatically generated from the filename
    following the format 'law__events'.
    If law or events is None, 'None' is displayed.

    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file containing columns 'DEPARTURE' and 'ID'.
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise ValueError(f"File not found: {csv_file}")

    def parse_section(section):
        """Parse a section: replace '-' with ','."""
        if section is None:
            return "None"
        parts = section.split("-")
        return ",".join(parts) if parts else "None"

    def filename_to_label(name):
        """Transform filename into a readable label."""
        name = name.replace(".csv", "")
        parts = name.split("__")
        law = parts[0] if len(parts) > 0 else "None"
        events = parts[1] if len(parts) > 1 else "None"

        law_parts = [parse_section(s) for s in law.split("_")]
        event_parts = [parse_section(s) for s in events.split("_")]

        return "_".join(law_parts + event_parts)

    # Create label and title
    label = filename_to_label(csv_file.name)
    law_part = csv_file.stem.split("__")[0] if "__" in csv_file.stem else "None"
    title = law_part.split("_")[0] if law_part else "None"

    # Load CSV
    df = pd.read_csv(csv_file, sep=';')
    if "DEPARTURE" not in df.columns or "ID" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_file.name}")

    # Clean and transform data
    df["DEPARTURE"] = df["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
    df["DEPARTURE"] = pd.to_datetime(df["DEPARTURE"], format='%H:%M:%S', errors='coerce')
    df = df.drop(columns=[c for c in ["ORIGIN", "DESTINATION"] if c in df.columns])
    df = df.dropna(subset=["DEPARTURE"])
    df["DEPARTURE"] = df["DEPARTURE"].dt.round("min")
    df = df.groupby("DEPARTURE").count().reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["DEPARTURE"], df["ID"], width=pd.Timedelta(minutes=1), align='center', color='steelblue')
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of departures")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# map

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
    # Convert WKT origin points into separate X and Y columns
    df[["ORIGIN_X", "ORIGIN_Y"]] = df["ORIGIN"].apply(
        lambda p: p.replace("POINT(", "").replace(")", "")
    ).str.split(' ', expand=True)

    # Convert WKT destination points into separate X and Y columns
    df[["DESTINATION_X", "DESTINATION_Y"]] = df["DESTINATION"].apply(
        lambda p: p.replace("POINT(", "").replace(")", "")
    ).str.split(' ', expand=True)

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


def display_origin_destination_maps_one(csv_file, crs="EPSG:4326"):
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
    ori_sum_gdf.plot(ax=ax[0], color="red", edgecolor="black", alpha=1, markersize=ori_sum_gdf["ID"] / 5)
    ctx.add_basemap(ax[0], source=ctx.providers.OpenStreetMap.Mapnik)
    ax[0].axis("off")
    ax[0].set_title("Demand by Origin", fontsize=14)

    # Destinations map
    dest_sum_gdf.plot(ax=ax[1], color="blue", edgecolor="black", alpha=1, markersize=dest_sum_gdf["ID"] / 5)
    ctx.add_basemap(ax[1], source=ctx.providers.OpenStreetMap.Mapnik)
    ax[1].axis("off")
    ax[1].set_title("Demand by Destination", fontsize=14)

    plt.suptitle(f"Origin and Destination Maps — {Path(csv_file).stem}", fontsize=16)
    plt.tight_layout()
    plt.show()


def display_origin_heatmap_time_one(data, crs="EPSG:3857", time_step="2h"):
    """
    Displays a time-based heatmap of trip origins using Folium.

    Parameters
    ----------
    data : GeoDataFrame or str
        Either a GeoDataFrame containing 'ORIGIN', 'DEPARTURE', and 'ID',
        or the path to a CSV file containing these columns.
    crs : str, optional
        Input coordinate reference system (default: EPSG:3857).
    time_step : str, optional
        Time rounding step (e.g., '30min', '1h', '2h').
    """
    # Load data
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data, sep=';')
        if not {"ORIGIN", "DEPARTURE", "ID"}.issubset(df.columns):
            raise ValueError("CSV must contain ORIGIN, DEPARTURE, and ID columns.")
    elif isinstance(data, gpd.GeoDataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a GeoDataFrame or a CSV file path.")

    # Prepare origin GeoDataFrame
    ori_gdf, _ = prepare_gdf(df, crs=crs)
    gdf = ori_gdf.to_crs("EPSG:4326")  # lat/lon for Folium

    # Round departure times
    gdf["DEPARTURE"] = pd.to_datetime(gdf["DEPARTURE"], errors="coerce")
    gdf = gdf.dropna(subset=["DEPARTURE"])
    gdf["DEPARTURE"] = gdf["DEPARTURE"].dt.round(time_step)

    # Group by time and origin
    grouped = gdf.groupby(["DEPARTURE", "ORIGIN"])["ID"].count().reset_index()
    grouped = gpd.GeoDataFrame(grouped, geometry="ORIGIN", crs="EPSG:4326")

    # Prepare data for HeatMapWithTime
    maxi = grouped["ID"].max()
    grouped["DEPARTURE"] = grouped["DEPARTURE"].dt.strftime("%H:%M")
    time_labels = sorted(grouped["DEPARTURE"].unique())
    hm_point_data = []
    x = 0
    for t in time_labels:
        points = []
        while x < grouped.shape[0] and grouped.loc[x, "DEPARTURE"] == t:
            val = float(grouped.loc[x, "ID"] / maxi)
            points.append([grouped.loc[x, "ORIGIN"].y, grouped.loc[x, "ORIGIN"].x, val])
            x += 1
        hm_point_data.append(points)

    # Create map
    mean_lat = grouped.geometry.y.mean()
    mean_lon = grouped.geometry.x.mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="cartodb positron")

    # Color gradient
    gradient = {0.0: 'white', 0.2: 'yellow', 0.4: 'orange', 0.6: 'red', 1.0: 'darkred'}
    cm.LinearColormap(colors=['white', 'yellow', 'orange', 'red', 'darkred'],
                       vmin=0, vmax=maxi, caption='Actual quantity').add_to(m)

    # Add HeatMapWithTime
    HeatMapWithTime(hm_point_data, index=time_labels, gradient=gradient, max_opacity=1,
                    radius=15, blur=0.8).add_to(m)

    return m


def display_destination_heatmap_time_one(data, crs="EPSG:3857", time_step="2h"):
    """
    Displays a time-based heatmap of trip destinations using Folium.

    Parameters
    ----------
    data : GeoDataFrame or str
        Either a GeoDataFrame containing 'DESTINATION', 'DEPARTURE', and 'ID',
        or the path to a CSV file containing these columns.
    crs : str, optional
        Input coordinate reference system (default: EPSG:3857).
    time_step : str, optional
        Time rounding step (e.g., '30min', '1h', '2h').
    """
    # Load data
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data, sep=';')
        if not {"DESTINATION", "DEPARTURE", "ID"}.issubset(df.columns):
            raise ValueError("CSV must contain DESTINATION, DEPARTURE, and ID columns.")
    elif isinstance(data, gpd.GeoDataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a GeoDataFrame or a CSV file path.")

    # Prepare destination GeoDataFrame
    _, dest_gdf = prepare_gdf(df, crs=crs)
    gdf = dest_gdf.to_crs("EPSG:4326")  # lat/lon for Folium

    # Round departure times
    gdf["DEPARTURE"] = pd.to_datetime(gdf["DEPARTURE"], errors="coerce")
    gdf = gdf.dropna(subset=["DEPARTURE"])
    gdf["DEPARTURE"] = gdf["DEPARTURE"].dt.round(time_step)

    # Group by time and destination
    grouped = gdf.groupby(["DEPARTURE", "DESTINATION"])["ID"].count().reset_index()
    grouped = gpd.GeoDataFrame(grouped, geometry="DESTINATION", crs="EPSG:4326")

    # Prepare data for HeatMapWithTime
    maxi = grouped["ID"].max()
    grouped["DEPARTURE"] = grouped["DEPARTURE"].dt.strftime("%H:%M")
    time_labels = sorted(grouped["DEPARTURE"].unique())
    hm_point_data = []
    x = 0
    for t in time_labels:
        points = []
        while x < grouped.shape[0] and grouped.loc[x, "DEPARTURE"] == t:
            val = float(grouped.loc[x, "ID"] / maxi)
            points.append([grouped.loc[x, "DESTINATION"].y, grouped.loc[x, "DESTINATION"].x, val])
            x += 1
        hm_point_data.append(points)

    # Create map
    mean_lat = grouped.geometry.y.mean()
    mean_lon = grouped.geometry.x.mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="cartodb positron")

    # Color gradient
    gradient = {0.0: 'white', 0.2: 'yellow', 0.4: 'orange', 0.6: 'red', 1.0: 'darkred'}
    cm.LinearColormap(colors=['white', 'yellow', 'orange', 'red', 'darkred'],
                       vmin=0, vmax=maxi, caption='Actual quantity').add_to(m)

    # Add HeatMapWithTime
    HeatMapWithTime(hm_point_data, index=time_labels, gradient=gradient, max_opacity=1,
                    radius=15, blur=0.8).add_to(m)

    return m






# inputs methods for multiple files

# diagram

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
        buffer["DEPARTURE"] = buffer["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        buffer["DEPARTURE"] = pd.to_datetime(buffer["DEPARTURE"], format='%H:%M:%S', errors='coerce')
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

    # labels
    labels = [filename_to_label(f.name) for f in files]
    
    # title
    first_file = files[0].name.replace(".csv", "")
    parts = first_file.split("__")
    loi_part = parts[0] if len(parts) > 0 else "None"

    # extracts the type
    title = loi_part.split("_")[0] if loi_part else "None"

    # organizes subplots
    if cols <= 0 or cols > n:
        cols = min(3, n)
    rows = math.ceil(n / cols)

    # loads dataframes
    variations = []
    for file in files:
        buffer = pd.read_csv(file, sep=';')
        if "DEPARTURE" not in buffer.columns or "ID" not in buffer.columns:
            raise ValueError(f"Missing required columns in file {file.name}")
        buffer["DEPARTURE"] = buffer["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        buffer["DEPARTURE"] = pd.to_datetime(buffer["DEPARTURE"], format='%H:%M:%S', errors='coerce')
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

    # removes unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])
        
    plt.suptitle(title, fontsize=14)
    plt.show()

# map
# ------------------------
# ORIGIN MAPS
# ------------------------
def display_origin_map_auto(path="", cols=0, crs="EPSG:4326"):
    """
    Displays origin maps for multiple CSV files containing geographic coordinates.
    Each file is displayed in a separate subplot with different colors and
    point sizes based on a global scale.
    """
    if not path or str(path).strip() == "":
        raise ValueError("Invalid or empty path.")
    
    files = [p for p in Path(path).glob("*.csv") if not p.name.startswith(".")]
    files.sort()
    n = len(files)
    if n == 0:
        raise ValueError("No CSV files found in the given folder.")
    
    if cols <= 0 or cols > n:
        cols = min(3, n)
    rows = math.ceil(n / cols)

    # ---- Collect all GeoDataFrames to compute global scale ----
    gdfs = []
    for file in files:
        data = pd.read_csv(file, sep=';')
        if "ORIGIN" not in data.columns or "ID" not in data.columns:
            raise ValueError(f"Missing required columns in file {file.name} (requires ORIGIN and ID).")
        ori_gdf, _ = prepare_gdf(data, crs=crs)
        gdf = gpd.GeoDataFrame(ori_gdf.groupby("ORIGIN")["ID"].count().reset_index(),
                               geometry="ORIGIN", crs=crs)
        gdf["ID"] = gdf["ID"].astype(float)
        gdfs.append(gdf)

    # global max for size scaling
    max_id = max(gdf["ID"].max() for gdf in gdfs)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=False, constrained_layout=True)
    colors = plt.cm.tab10.colors  # 10 distinct colors

    for i, (file, gdf) in enumerate(zip(files, gdfs)):
        gdf = gdf.to_crs("EPSG:3857")
        gdf["size"] = (gdf["ID"] / max_id) * 200  # scale sizes uniformly
        color = colors[i % len(colors)]

        r, c = divmod(i, cols)
        ax = axes[r][c]
        gdf.plot(ax=ax, color=color, edgecolor="black", alpha=0.8, markersize=gdf["size"])
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.axis("off")
        ax.set_title(f"Origins — {file.stem}", fontsize=10)

    # remove empty subplots
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])

    plt.suptitle("Demand by Origin (common scale)", fontsize=14)
    plt.show()


# ------------------------
# DESTINATION MAPS
# ------------------------
def display_destination_map_auto(path="", cols=0, crs="EPSG:4326"):
    """
    Displays destination maps for multiple CSV files containing geographic coordinates.
    Each file is displayed in a separate subplot with different colors and
    point sizes based on a global scale.
    """
    if not path or str(path).strip() == "":
        raise ValueError("Invalid or empty path.")
    
    files = [p for p in Path(path).glob("*.csv") if not p.name.startswith(".")]
    files.sort()
    n = len(files)
    if n == 0:
        raise ValueError("No CSV files found in the given folder.")
    
    if cols <= 0 or cols > n:
        cols = min(3, n)
    rows = math.ceil(n / cols)

    gdfs = []
    for file in files:
        data = pd.read_csv(file, sep=';')
        if "DESTINATION" not in data.columns or "ID" not in data.columns:
            raise ValueError(f"Missing required columns in file {file.name} (requires DESTINATION and ID).")
        _, dest_gdf = prepare_gdf(data, crs=crs)
        gdf = gpd.GeoDataFrame(dest_gdf.groupby("DESTINATION")["ID"].count().reset_index(),
                               geometry="DESTINATION", crs=crs)
        gdf["ID"] = gdf["ID"].astype(float)
        gdfs.append(gdf)

    max_id = max(gdf["ID"].max() for gdf in gdfs)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=False, constrained_layout=True)
    colors = plt.cm.Paired.colors  # another palette for distinction

    for i, (file, gdf) in enumerate(zip(files, gdfs)):
        gdf = gdf.to_crs("EPSG:3857")
        gdf["size"] = (gdf["ID"] / max_id) * 200
        color = colors[i % len(colors)]

        r, c = divmod(i, cols)
        ax = axes[r][c]
        gdf.plot(ax=ax, color=color, edgecolor="black", alpha=0.8, markersize=gdf["size"])
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.axis("off")
        ax.set_title(f"Destinations — {file.stem}", fontsize=10)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])

    plt.suptitle("Demand by Destination (common scale)", fontsize=14)
    plt.show()



# Sensitivity

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


