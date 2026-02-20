
# dependencies

import os
import sys

import ast

import importlib

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import wkt

import logging

import modules.Utils as Utils
importlib.reload(Utils)

# log
logger = logging.getLogger(__name__)




from shapely.geometry import Point, Polygon


import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import contextily as ctx


import logging

logger = logging.getLogger(__name__)



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

def associate(demand_path="", classification_input_path="", crs="", save=False, classification_output_path=""):
    """
    Associates a classification with a MnMS demand file.
    The line-to-cluster association takes the form of a one-variable DataFrame (the CLUSTER_ID variable),
    where each row contains the ID of the cluster associated with the row having the same index in the demand file.

    Parameters
    ----------
    demand_path : str
        Path to the dataframe.
    classification_input_path : str
        Path to the classification.
    crs : str
        Projection system.
    save : bool
        If true saves the result at classification_output_path.
    classification_output_path : str
        Path to save the dataframe.
        
    Returns
    -------
    output_df : pandas.DataFrame
        Dataframes with column CLUSTER_ID
    """

    # --- 1. Checks ---
    if not demand_path or demand_path.strip() == "":
        raise ValueError("Invalid or null demand path.")
    if not classification_input_path or classification_input_path.strip() == "":
        raise ValueError("Invalid or null classification input path.")
    if save and (not classification_output_path or classification_output_path.strip() == ""):
        raise ValueError("Invalid or null classification output path.")

    # --- 2. Load demand file ---
    demand_df = pd.read_csv(demand_path, sep=';')
    o, d = prepare_gdf(demand_df, crs)
    demand_gdf = o.copy()
    demand_gdf["DESTINATION"] = d["DESTINATION"]
    demand_gdf = demand_gdf.to_crs(crs)

    # --- 3. Load classification file ---
    class_df = pd.read_csv(classification_input_path, sep=';')

    # Charger les géométries
    class_df["ORIGIN"] = class_df["ORIGIN"].apply(wkt.loads)
    class_df["DESTINATION"] = class_df["DESTINATION"].apply(wkt.loads)

    # Construire un unique GeoDataFrame cohérent
    class_gdf = gpd.GeoDataFrame(class_df.copy(), geometry="ORIGIN", crs=crs)
    class_gdf["ORIGIN_GEOM"] = class_gdf["ORIGIN"]
    class_gdf["DEST_GEOM"] = class_gdf["DESTINATION"]

    # Créer des GeoSeries pour accéder à geom_type
    origin_geom_series = gpd.GeoSeries(class_gdf["ORIGIN_GEOM"])
    dest_geom_series = gpd.GeoSeries(class_gdf["DEST_GEOM"])

    # --- 4. Output ---
    cluster_ids = []

    # --- 5. Matching logic ---
    tolerance = 1e-1  # tolérance pour les points

    for i, row in demand_gdf.iterrows():

        origin = row["ORIGIN"]
        destination = row["DESTINATION"]

        # -------------------------
        # 1️⃣ POINT - POINT
        # -------------------------
        matched = class_gdf[
            (origin_geom_series.geom_type == "Point")
            & (dest_geom_series.geom_type == "Point")
            & class_gdf["ORIGIN_GEOM"].apply(lambda g: origin.distance(g) < tolerance)
            & class_gdf["DEST_GEOM"].apply(lambda g: destination.distance(g) < tolerance)
        ]
        if len(matched) == 1:
            cluster_ids.append(matched["CLUSTER_ID"].values[0])
            continue

        # -------------------------
        # 2️⃣ POINT - POLYGON/MULTIPOLYGON
        # -------------------------
        matched = class_gdf[
            (origin_geom_series.geom_type == "Point")
            & (dest_geom_series.geom_type.isin(["Polygon", "MultiPolygon"]))
            & class_gdf["ORIGIN_GEOM"].apply(lambda g: origin.distance(g) < tolerance)
            & class_gdf["DEST_GEOM"].apply(lambda g: destination.within(g))
        ]
        if len(matched) == 1:
            cluster_ids.append(matched["CLUSTER_ID"].values[0])
            continue

        # -------------------------
        # 3️⃣ POLYGON/MULTIPOLYGON - POINT
        # -------------------------
        matched = class_gdf[
            (origin_geom_series.geom_type.isin(["Polygon", "MultiPolygon"]))
            & (dest_geom_series.geom_type == "Point")
            & class_gdf["ORIGIN_GEOM"].apply(lambda g: origin.within(g) or origin.equals(g))
            & class_gdf["DEST_GEOM"].apply(lambda g: destination.distance(g) < tolerance)
        ]
        if len(matched) == 1:
            cluster_ids.append(matched["CLUSTER_ID"].values[0])
            continue

        # -------------------------
        # 4️⃣ POLYGON/MULTIPOLYGON - POLYGON/MULTIPOLYGON
        # -------------------------
        matched = class_gdf[
            (origin_geom_series.geom_type.isin(["Polygon", "MultiPolygon"]))
            & (dest_geom_series.geom_type.isin(["Polygon", "MultiPolygon"]))
            & class_gdf["ORIGIN_GEOM"].apply(lambda g: origin.within(g) or origin.equals(g))
            & class_gdf["DEST_GEOM"].apply(lambda g: destination.within(g) or destination.equals(g))
        ]
        if len(matched) == 1:
            cluster_ids.append(matched["CLUSTER_ID"].values[0])
            continue

        # -------------------------
        # ❌ Aucun match trouvé
        # -------------------------
        logger.error(f"No match for row {i}: Origin={origin}, Destination={destination}")
        raise ValueError(
            f"Error: expected exactly 1 matching cluster after all 4 geometry-level tests, "
            f"found {len(matched)} at demand row {i}."
        )




    # --- 6. Build final output DF ---
    output_df = pd.DataFrame({"CLUSTER_ID": cluster_ids})

    # --- 7. Output filename ---
    demand_name = os.path.basename(demand_path)
    demand_parts = demand_name.split("__")

    if len(demand_parts) < 3:
        raise ValueError("Demand filename must contain at least 3 '__' categories.")

    class_name = os.path.basename(classification_input_path)

    output_name = "__".join(demand_parts[:3]) + f"__{class_name}"

    # --- 8. Save ---
    if save:
        output_path = os.path.join(classification_output_path, output_name)
        output_df.to_csv(output_path, sep=';', index=False)

    return output_df

def ordered_classification(classified_demand):
    """
    Returns a dataframe with cluster id ordered by accumulation in the classified_demand df.

    Parameters
    ----------
    classified_demand : pandas.DataFrame
        Classified demand dataframe containing at least the column 'CLUSTER_ID'.

    Returns
    -------
    pandas.DataFrame
        Dataframe with:
            - CLUSTER_ID
            - COUNT  (number of occurrences)
        Ordered by decreasing COUNT.
    """

    # -------------------------
    # VALIDATION
    # -------------------------
    if classified_demand is None:
        logger.error("classified_demand is None.")
        raise ValueError("classified_demand cannot be None.")

    if not isinstance(classified_demand, pd.DataFrame):
        logger.error("classified_demand is not a dataframe.")
        raise ValueError("classified_demand must be a pandas DataFrame.")

    if classified_demand.empty:
        logger.error("classified_demand is empty.")
        raise ValueError("classified_demand cannot be empty.")

    if "CLUSTER_ID" not in classified_demand.columns:
        logger.error("Missing CLUSTER_ID column.")
        raise ValueError("classified_demand must contain column 'CLUSTER_ID'.")

    # -------------------------
    # CORE COMPUTATION
    # -------------------------
    classified_clusters = (
        classified_demand
        .groupby("CLUSTER_ID")
        .agg(COUNT=("ID", "count"))
        .reset_index()
        .sort_values("COUNT", ascending=False)
        .reset_index(drop=True)
    )

    return classified_clusters

def plot_ordered_classification(df_cluster, path_polygons_csv, zoom=12, extent=None):
    """
    Display the clusters on a map with an OSM background,
    allowing the zoom level (tile level) and the map extent to be adjusted.
    
    Parameters 
    ----------
    df_cluster : pandas.DataFrame
        DataFrame containing CLUSTER_ID, COUNT
    path_polygons_csv : str
        CSV (sep=';') containing CLUSTER_ID, ORIGIN (WKT), DESTINATION (WKT)
    zoom : int
        background tile detail level (context) — higher = more detail.
    extent : tuple (xmin, xmax, ymin, ymax) or None. 
        If provided, sets the map boundaries.

    Returns
    -------
    None
    """

    df_poly = pd.read_csv(path_polygons_csv, sep=';')
    required = {"CLUSTER_ID", "ORIGIN", "DESTINATION"}
    if not required.issubset(df_poly.columns):
        raise ValueError(f"Le CSV doit contenir {required}")

    df_poly["ORIGIN"] = df_poly["ORIGIN"].apply(wkt.loads)
    df_poly["DESTINATION"] = df_poly["DESTINATION"].apply(wkt.loads)
    df_poly = gpd.GeoDataFrame(df_poly, geometry="ORIGIN", crs="EPSG:32631")

    for idx, row in df_cluster.iterrows():
        cid = row["CLUSTER_ID"]
        count = row["COUNT"]

        sel = df_poly[df_poly["CLUSTER_ID"] == cid]
        if sel.empty:
            print(f"⚠️ Aucun polygone pour CLUSTER_ID = {cid}")
            continue

        origin = sel.iloc[0]["ORIGIN"]
        destination = sel.iloc[0]["DESTINATION"]

        gdf_o = gpd.GeoDataFrame([{"type": "origin"}],
                                 geometry=[origin],
                                 crs="EPSG:32631")
        gdf_d = gpd.GeoDataFrame([{"type": "destination"}],
                                 geometry=[destination],
                                 crs="EPSG:32631")

        fig, ax = plt.subplots(figsize=(10, 8))

        gdf_o.plot(ax=ax, facecolor="red", edgecolor="black", alpha=0.6, label="ORIGIN")
        gdf_d.plot(ax=ax, facecolor="blue", edgecolor="black", alpha=0.4, label="DESTINATION")

        ax.set_title(f"Cluster {cid} — COUNT = {count}", fontsize=14)

        # Si une extent est donnée, on fixe les limites
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        # Ajouter le fond de carte avec zoom contrôlé
        try:
            ctx.add_basemap(ax,
                            zoom=zoom,
                            crs="EPSG:32631",
                            source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print("⚠️ Impossible de charger le fond de carte:", e)

        ax.legend(loc="upper right")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

def ordered_zones(classified_demand_path="", classification_path="", n=10) : 
    """
    Generate the ordered clusters (origin–destination couples, origins, destinations, and combined zones)
    based on demand counts, and return the top n most significant elements.

    Parameters
    ----------
    classified_demand_path : str
        Path to the CSV file (sep=';') containing individual demand data 
        with at least CLUSTER_ID and ID columns.
    classification_path : str
        Path to the CSV file (sep=';') containing cluster definitions 
        with CLUSTER_ID, ORIGIN_CLUSTER, DESTINATION_CLUSTER, ORIGIN, DESTINATION.
    n : int, optional
        Number of top elements to return (default = 10).

    Returns
    -------
    ordered_couples : pandas.DataFrame
        Top n origin–destination cluster pairs sorted by COUNT (descending).
    ordered_origins : pandas.DataFrame
        Top n origin clusters aggregated by total COUNT.
    ordered_destinations : pandas.DataFrame
        Top n destination clusters aggregated by total COUNT.
    ordered_zones : pandas.DataFrame
        Top n combined zones (origin + destination counts), 
        sorted by total COUNT (descending).
    """


    # check
    if not classified_demand_path or classified_demand_path.strip() == "" : 
        logger.info("Invalid or null classified demand path.")
        raise ValueError("Invalid or null classified demand path.")
    if not classification_path or classification_path.strip() == "" : 
        logger.info("Invalid or null classification path.")
        raise ValueError("Invalid or null classification path.")

    classified_demand = pd.read_csv(classified_demand_path, sep=';')
    classification = pd.read_csv(classification_path, sep=';')

    buffer = classified_demand.groupby("CLUSTER_ID").agg({"ID":"count"}).reset_index()
    buffer = buffer.rename(columns={"ID":"COUNT"})
    df = pd.merge(buffer, classification, left_on="CLUSTER_ID", right_on="CLUSTER_ID")

    ordered_couples = df.sort_values(by="COUNT", ascending=False).head(n)

    ordered_origins = df.groupby("ORIGIN_CLUSTER").agg({"COUNT":"sum", "ORIGIN":"first"}).reset_index()
    ordered_origins = ordered_origins.rename(columns={"ORIGIN_CLUSTER":"CLUSTER_ID", "ORIGIN":"GEOMETRY"})
    ordered_origins = ordered_origins.sort_values(by="COUNT", ascending=False).head(n)

    ordered_destinations = df.groupby("DESTINATION_CLUSTER").agg({"COUNT":"sum", "DESTINATION":"first"}).reset_index()
    ordered_destinations = ordered_destinations.rename(columns={"DESTINATION_CLUSTER":"CLUSTER_ID", "DESTINATION":"GEOMETRY"})
    ordered_destinations = ordered_destinations.sort_values(by="COUNT", ascending=False).head(n)

    ordered_zones = pd.merge(ordered_origins, ordered_destinations, left_on="CLUSTER_ID", right_on="CLUSTER_ID")

    ordered_zones["COUNT"] = ordered_zones["COUNT_x"] + ordered_zones["COUNT_y"]
    ordered_zones = ordered_zones.drop(columns=["COUNT_x", "COUNT_y", "GEOMETRY_y"])
    ordered_zones = ordered_zones.rename(columns={"GEOMETRY_x":"GEOMETRY"})
    ordered_zones = ordered_zones.sort_values(by="COUNT", ascending=False).head(n)

    return ordered_couples, ordered_origins, ordered_destinations, ordered_zones
    
def plot_ordered_zone(df, crs, zoom=11, width=1000, height=700):
    """
    Display the ordered zones on an interactive map,
    handling Point, Polygon, and MultiPolygon geometries,
    with ranking and legend.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing CLUSTER_ID, COUNT, and GEOMETRY 
        (WKT strings or shapely geometries).
    crs : str or pyproj.CRS
        Coordinate reference system of the input geometries.
    zoom : int, optional
        Initial zoom level of the map (default = 11).
    width : int, optional
        Width of the figure in pixels (default = 1000).
    height : int, optional
        Height of the figure in pixels (default = 700).

    Returns
    -------
    None
    """

    # -----------------------------
    # 1️⃣ Préparation des données
    # -----------------------------
    df = df.copy(deep=True)

    if isinstance(df["GEOMETRY"].iloc[0], str):
        df.loc[:, "GEOMETRY"] = df["GEOMETRY"].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="GEOMETRY", crs=crs)
    gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.sort_values("COUNT", ascending=False).reset_index(drop=True)
    gdf["RANK"] = gdf.index + 1
    gdf["CENTROID"] = gdf.geometry.centroid

    # -----------------------------
    # 2️⃣ Palette sans noir/blanc
    # -----------------------------
    colors_raw = px.colors.qualitative.Dark24
    def is_color_ok(hex_color):
        rgb = mcolors.hex2color(hex_color)
        lum = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
        return 0.1 < lum < 0.9  # éviter trop sombre ou trop clair
    colors = [c for c in colors_raw if is_color_ok(c)]

    fig = go.Figure()

    # -----------------------------
    # 3️⃣ Tracé des géométries
    # -----------------------------
    for idx, row in gdf.iterrows():
        geom = row["GEOMETRY"]
        color = colors[idx % len(colors)]
        rank = row["RANK"]
        cluster_id = row["CLUSTER_ID"]
        count = row["COUNT"]
        legend_name = f"{rank} | Cluster {cluster_id} | Count {count}"
        hover_text = f"Cluster ID: {cluster_id}<br>Rang: {rank}<br>Count: {count}"

        # ---- POINT
        if geom.geom_type == "Point":
            fig.add_trace(
                go.Scattermap(
                    lon=[geom.x],
                    lat=[geom.y],
                    mode="markers+text",
                    marker=dict(size=10, color=color),
                    text=[str(rank)],
                    textposition="middle right",
                    name=legend_name,
                    hovertext=[hover_text],
                    hoverinfo="text"
                )
            )

        # ---- POLYGON / MULTIPOLYGON
        else:
            polygons = [geom] if geom.geom_type == "Polygon" else geom.geoms

            # Pour la légende : seule la première trace a showlegend=True
            first = True
            for poly in polygons:
                x, y = poly.exterior.xy
                fig.add_trace(
                    go.Scattermap(
                        lon=list(x),
                        lat=list(y),
                        mode="lines",
                        fill="toself",
                        fillcolor=color,
                        opacity=0.5,
                        line=dict(color=color),
                        name=legend_name if first else None,
                        showlegend=first,
                        hoverinfo="skip"  # hover via centroïde
                    )
                )
                first = False

            # centroid invisible pour hover
            centroid = row["CENTROID"]
            fig.add_trace(
                go.Scattermap(
                    lon=[centroid.x],
                    lat=[centroid.y],
                    mode="markers+text",
                    marker=dict(size=1, color="rgba(0,0,0,0)"),  # invisible
                    text=[str(rank)],
                    textposition="middle center",
                    name=None,
                    hovertext=[hover_text],
                    hoverinfo="text",
                    showlegend=False
                )
            )

    # -----------------------------
    # 4️⃣ Layout
    # -----------------------------
    fig.update_layout(
        width=width,
        height=height,
        map=dict(
            style="carto-positron",
            zoom=zoom,
            center=dict(
                lat=gdf["CENTROID"].y.mean(),
                lon=gdf["CENTROID"].x.mean()
            )
        ),
        legend=dict(title="Rang | Cluster | Count"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()


    