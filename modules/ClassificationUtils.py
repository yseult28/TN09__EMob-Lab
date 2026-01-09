
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




import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, Polygon
import logging

logger = logging.getLogger(__name__)


def associate(demand_path="", classification_input_path="", crs="", save=False, classification_output_path=""):
    """
    Associates a classification with a MnMS demand file.
    The line-to-cluster association takes the form of a one-variable DataFrame (the CLUSTER_ID variable),
    where each row contains the ID of the cluster associated with the row having the same index in the demand file.
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
    o, d = Utils.prepare_gdf(demand_df, crs)
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


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely import wkt

def plot_ordered_classification(df_cluster, path_polygons_csv, zoom=12, extent=None):
    """
    Affiche les clusters sur une carte avec fond OSM,  
    en permettant de régler le zoom (niveau de tuile) et l'étendue de la vue.

    Paramètres :
      - df_cluster : DataFrame avec CLUSTER_ID, COUNT
      - path_polygons_csv : CSV (sep=';') contenant CLUSTER_ID, ORIGIN (WKT), DESTINATION (WKT)
      - zoom : int, niveau de détail de la tuile de fond (contexte) — plus grand = plus de détails. 
      - extent : tuple (xmin, xmax, ymin, ymax) ou None. Si donné, fixe les limites de la carte.
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



    


    