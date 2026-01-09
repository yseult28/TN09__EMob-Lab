
# dependencies

import os

import ast

from pathlib import Path

import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex

import branca.colormap as cm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

import contextily as ctx
import geoplot as gplt
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
import plotly.express as px
import plotly.graph_objects as go

from itertools import product

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift, estimate_bandwidth, AgglomerativeClustering

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from itertools import product
from tqdm import tqdm
from time import time

import logging



# log
logger = logging.getLogger(__name__)

class Clustering:

    # Constructor

    def __init__(self, path):
        """
        ClusteringVariation's constructor.
        loads the original demand and prepocesses it : changes all variables type to numeric ones, normalizes.

        Parameters
        ----------
        path : string
            Path to the original demand.
        """

        # check
        if not path or path.strip() == "":
            logger.error("Invalid or null path")
            raise ValueError("Invalid or null path")

        # assignment
        self._path = path # path to the demand file
        self._original_demand = pd.read_csv(path, sep=';') # original demand dataframe, the separator must be ';'
        self._demand = self._original_demand.copy() # to be preprocess dataframes
        self._labels = []

        self._parameters_analysis_name = ""
        self._parameters_analysis_results = []

        self._clustering_name = ""
        self._clustering_result = []

        
        
        # preprocessing
        self._demand["DEPARTURE"] = self._demand["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._demand["DEPARTURE"] = pd.to_datetime(self._demand["DEPARTURE"], format='%H:%M:%S')
        self._demand["DEPARTURE"] = self._demand["DEPARTURE"].apply(lambda x : x.hour*3600 + x.minute*60 + x.second)

        self._demand[["ORIGIN_X","ORIGIN_Y"]] = self._demand["ORIGIN"].str.split(' ',expand=True)
        self._demand[["DESTINATION_X","DESTINATION_Y"]] = self._demand["DESTINATION"].str.split(' ',expand=True)
        self._demand[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]] = self._demand[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]].astype(float)
        self._demand = self._demand.drop(columns=["ORIGIN","DESTINATION"])

        self._demand = self._demand.drop(columns=["ID"])

        scaler = StandardScaler()
        self._demand = scaler.fit_transform(self._demand)

        logger.info("ClusteringVariation initialized")



    # configuration's method

    def save_labels(self, path, name=""):
        """
        Saves _labels in a csv file at path.

        Parameters
        ----------
        path : string
             Path to save _labels
        """

        # check
        if len(self._labels) == 0:
            logger.error("Invalid or null _labels.")
            raise ValueError("Invalid or null _labels.")
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not name or name.strip() == "":
            name = self._clustering_name 

        os.makedirs(path, exist_ok=True)

        labels_name = "labels_" + name
        labels_path = os.path.join(path, f"{labels_name}.csv")
        results_name = "results_" + name
        results_path =  os.path.join(path, f"{results_name}.csv")

        label_df = self._labels.copy()
        label_df.to_csv(labels_path, sep=';', index=False)

        results_df = pd.DataFrame(self._clustering_result)
        results_df.to_csv(results_path, sep=';', index=False)

        logger.info(f"_labels and _results saved : {path}.")


    def load_labels(self, path):
        """
        Loads labels from path in _labels.

        Parameters
        ----------
        path : string
            Path to the labels file.
        """

        # check 
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        buffer = pd.read_csv(path, sep=';')

        # check
        if buffer.shape[1] != 1:
            logger.error("Invalid labels.")
            raise ValueError("Invalid labelsh.")
        if buffer.shape[0] == self._demand.shape[0]:
            logger.error("Invalid labels.")
            raise ValueError("Invalid labelsh.")

        self._labels = buffer

        logger.info(f"labels loads from {path}.")

    def load_parameters_analysis_results(self, path=""):
        """
        Loads a parameters_analysis_results .csv from path.

        Parameters
        ----------
        path : string
            Path to the .csv.
        """

        # check
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        buffer = pd.read_csv(path, sep=';')

        paramaters_analysis_columns = {"n_clusters", "parameters", "scores"}

        # check
        if set(buffer.columns) != paramaters_analysis_columns:
            logger.error("Invalid file.")
            raise ValueError("Invalid file.")

        self._parameters_analysis_results = buffer
        path = Path(path) 
        name = path.stem
        self._parameters_analysis_name = name

        logger.info(f"_parameters_analysis_results loaded from {path}.")


    def save_parameters_analysis(self, path, name=""):
        """
        Saves the current parameters analysis result at path with name.

        Parameters
        ----------
        path : string
            Path to save the file.
        name : string, optional
            Name of the file.
        """

        # check
        if self._parameters_analysis_results.empty:
            logger.error("Invalid or null _parameters_analysis_results.")
            raise ValueError("Invalid or null _parameters_analysis_results.")
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not name or name.strip() == "":
            name = self._parameters_analysis_name

        parameters_path = os.path.join(path, f"{name}.csv")

        parameters_df = pd.DataFrame(self._parameters_analysis_results)

        parameters_df.to_csv(parameters_path, sep=';', index=False)

        logger.info(f"_parameters_analysis_results saved : {path}s.")
            

    # clustering

    def KMeans(self, n_clusters_value, n_init_value=10, max_iter_value=300, random_state=42):
        """
        Clustering with the K-means algorithm.
        The resulting labels are saved in the dataframes _labels.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to create.
        n_init : int, optional
            Number of time the algortihm will be run with different initial centroids.
        max_iter : int, optional
            Maximum number of iterations for a single run
        """

         # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        n_clusters = n_clusters_value
        n_init = n_init_value
        max_iter = max_iter_value
        results = []

        # algorithm
        try: 
            model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init, max_iter=max_iter, random_state=random_state)
            labels = model.fit_predict(buffer)

            silhouette = silhouette_score(buffer, labels)
            davies = davies_bouldin_score(buffer, labels)
            calinski_harabasz = calinski_harabasz_score(buffer, labels)

            results.append({
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'parameters':{'n_clusters': n_clusters, 'n_init': n_init},
                'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}
            })

        except Exception:
            raise

        self._clustering_name = "K-means"
        self._labels = pd.DataFrame(labels, columns=["label"])
        self._clustering_result = results
        self._clustering_result = pd.DataFrame(self._clustering_result)

        logger.info(f"Clustering K-means with {n_clusters} clusters.")

            

    def DBSCAN(self, eps_value, min_samples_value, random_state=42):
        """
        Clustering with DBSCAN algorithm. 
        The resulting labels are saved in the dataframes _labels

        Parameters
        ----------
        eps_value : float
            The maximum distance between two samples for them to be considered neighbors. Two points are in the same neighborhood if their distance ≤ eps.

        min_samples : int
            The minimum number of points (including the point itself) required within an eps-radius neighborhood for a point to be considered a core point.
            Points that don’t meet this density threshold are labeled as noise or border points.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        eps = eps_value
        min_samples = min_samples_value
        results = []

        # algorithm
        try:
            
            model = DBSCAN(eps=eps, min_samples=min_samples, random_state=random_state)
            labels = model.fit_predict(buffer)
            
            silhouette = silhouette_score(buffer, labels)
            davies = davies_bouldin_score(buffer, labels)
            calinski_harabasz = calinski_harabasz_score(buffer, labels)

            results.append({
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'parameters':{'eps': eps, 'min_samples': min_samples},
                'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}
            })
                    
        except Exception:
            raise

        self._clustering_name = "DBSCAN"
        self._labels = pd.DataFrame(labels, columns=["label"])
        self._clustering_result = results
        self._clustering_result = pd.DataFrame(self._clustering_result)

        logger.info(f"Clustering DBSCAN with eps value {eps} and min_samples {min_samples}.")


    def SpectralClustering(self, n_clusters_value, gamma_value, assign_labels_value, random_state=42):
        """
        Clustering with DBSCAN algorithm. 
        The resulting labels are saved in the dataframes _labels

        Parameters
        ----------
        eps_value : float
            The maximum distance between two samples for them to be considered neighbors. Two points are in the same neighborhood if their distance ≤ eps.

        min_samples : int
            The minimum number of points (including the point itself) required within an eps-radius neighborhood for a point to be considered a core point.
            Points that don’t meet this density threshold are labeled as noise or border points.

        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        n_cluster = n_clusters_value
        gamma = gamma_value
        assign_labels = assign_labels_value
        results = []

        # algorithm
        try:
            
            model = SpectralClustering(n_clusters=n_clusters, gamma=gamma, affinity='rbf', assign_labels=assign_labels, random_state=random_state)
            labels = model.fit_predict(buffer)
            
            silhouette = silhouette_score(buffer, labels)
            davies = davies_bouldin_score(buffer, labels)
            calinski_harabasz = calinski_harabasz_score(buffer, labels)

            results.append({
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'parameters':{'n_clusters': n_clusters, 'gamma': gamma, 'assign_labels': assign_labels},
                'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}         
            })
                    
        except Exception:
            raise

        self._clustering_name = "SpectralClustering"
        self._labels = pd.DataFrame(labels, columns=["label"])
        self._clustering_result = results
        self._clustering_result = pd.DataFrame(self._clustering_result)

        logger.info(f"Clustering SpectralCLustering with n_clusters value {n_clusters}, gamma {gamma} and assign_labels {assign_labels}.")


    def AffinityPropagation(self, preference_value, damping_value, random_state=42):
        """
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        preference = preference_value
        damping  = damping_value
        results = []

        # algorithm
        try:
            
            model = AffinityPropagation(preference=preference, damping=damping, random_state=random_state)
            labels = model.fit_predict(buffer)
            
            silhouette = silhouette_score(buffer, labels)
            davies = davies_bouldin_score(buffer, labels)
            calinski_harabasz = calinski_harabasz_score(buffer, labels)

            results.append({
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'parameters':{'preference': preference, 'damping': damping},
                'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}          
            })
                    
        except Exception:
            raise

        self._clustering_name = "AffinityPropagation"
        self._labels = pd.DataFrame(labels, columns=["label"])
        self._clustering_result = results
        self._clustering_result = pd.DataFrame(self._clustering_result)

        logger.info(f"Clustering AffinityPropagation with preference value {preference} and damping {damping}.")
            

    def MeanShift(self, quantile_value=0.2, bin_seeding=True, cluster_all=True):
        """
        Clustering with the MeanShift algorithm.
        The resulting labels are saved in the dataframe _labels.

        Parameters
        ----------
        quantile_value : float
            Quantile parameter used to estimate the bandwidth.
        bin_seeding : bool, optional
            If True, initial kernel locations are not all points, which speeds up the algorithm.
        cluster_all : bool, optional
            If True, all points are clustered; if False, points with low density become noise.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        results = []

        try:
            # Estimation automatique du rayon (bandwidth)
            bandwidth = estimate_bandwidth(buffer, quantile=quantile_value, n_samples=len(buffer))

            # Cas invalides
            if bandwidth <= 0 or np.isnan(bandwidth):
                logger.error(f"Invalid bandwidth estimated: {bandwidth}")
                raise ValueError("Invalid bandwidth estimated.")

            # Clustering
            model = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding, cluster_all=cluster_all)
            labels = model.fit_predict(buffer)

            # Ignore les cas dégénérés
            if len(set(labels)) <= 1:
                logger.warning("MeanShift produced a single cluster — ignored.")
                raise ValueError("Single cluster detected.")

            # Scores
            silhouette = silhouette_score(buffer, labels)
            davies = davies_bouldin_score(buffer, labels)
            calinski = calinski_harabasz_score(buffer, labels)

            results.append({
                'n_clusters': len(set(labels)),
                'parameters': {
                    'bandwidth': bandwidth,
                    'quantile': quantile_value,
                    'bin_seeding': bin_seeding,
                    'cluster_all': cluster_all
                },
                'scores': {
                    'silhouette': silhouette,
                    'davies_bouldin': davies,
                    'calinski_harabasz': calinski
                }
            })

        except Exception as e:
            logger.error(f"MeanShift clustering failed: {e}")
            raise

        # Save results
        self._clustering_name = "MeanShift"
        self._labels = pd.DataFrame(labels, columns=["label"])
        self._clustering_result = pd.DataFrame(results)

        logger.info(f"Clustering MeanShift completed with {len(set(labels))} clusters.")


    def Agglomerative(self, n_clusters_value=3, linkage="ward", metric="euclidean"):
        """
        Clustering with the Agglomerative Clustering algorithm.
        The resulting labels are saved in the dataframe _labels.

        Parameters
        ----------
        n_clusters_value : int
            The number of clusters to find.
        linkage : str, optional
            The linkage criterion to use: 'ward', 'complete', 'average', 'single'.
        affinity : str, optional
            The distance metric to use ('euclidean', 'l1', 'l2', 'manhattan', 'cosine').
            Note: 'ward' linkage only works with 'euclidean'.
        """

        # check 
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        results = []

        try:
            # Vérification de compatibilité
            if linkage == "ward" and metric != "euclidean":
                logger.warning("Ward linkage only supports Euclidean metric. Adjusting automatically.")
                metric = "euclidean"

            # Clustering
            model = AgglomerativeClustering(
                n_clusters=n_clusters_value,
                linkage=linkage,
                metric=metric
            )
            labels = model.fit_predict(buffer)

            # Ignore les cas dégénérés
            if len(set(labels)) <= 1:
                logger.warning("Agglomerative produced a single cluster — ignored.")
                raise ValueError("Single cluster detected.")

            # Scores
            silhouette = silhouette_score(buffer, labels)
            davies = davies_bouldin_score(buffer, labels)
            calinski = calinski_harabasz_score(buffer, labels)

            results.append({
                'n_clusters': n_clusters_value,
                'parameters': {
                    'n_clusters': n_clusters_value,
                    'linkage': linkage,
                    'metric': metric
                },
                'scores': {
                    'silhouette': silhouette,
                    'davies_bouldin': davies,
                    'calinski_harabasz': calinski
                }
            })

        except Exception as e:
            logger.error(f"Agglomerative clustering failed: {e}")
            raise

        # Save results
        self._clustering_name = "Agglomerative"
        self._labels = pd.DataFrame(labels, columns=["label"])
        self._clustering_result = pd.DataFrame(results)

        logger.info(f"Clustering Agglomerative completed with {n_clusters_value} clusters.")


    

    # parameter analysis

    def KMeans_parameters_analysis(self, n_clusters_params=[], n_init_params=[]):
        """
        For the given demand self._demand, performs a clustering using the K-means algorithm with all combinations
        of parameters from the lists n_clusters_params and n_init_params. The results are saved in list in the following format : 
        [<number of cluster>,[<parameters>], [<scores>]].

        Parameters
        ----------
        n_clusters_params : list
            A list of three values :  [start, end, step] used to produce a range of values for the n_clusters parameter.
        n_init_params : list
            A list of three values : [start, end, step] used to produce a range of values for the n_init parameter.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        n_clusters_values = range(n_clusters_params[0], n_clusters_params[1], n_clusters_params[2])  
        n_init_values = range(n_init_params[0], n_init_params[1], n_init_params[2]) 
        param_grid = list(product(n_clusters_values, n_init_values))
        max_iter=500
        
        buffer = self._demand.copy()

        # chronometer
        start_time = time()

        results = []

        for n_clusters, n_init in tqdm(param_grid, desc="🔍 Optimisation K-means", ncols=100):
            try:
                model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init, max_iter=max_iter)
                labels = model.fit_predict(buffer)

                # Ignore case with 0 or 1 cluster
                if len(set(labels)) <= 1:
                    continue

                silhouette = silhouette_score(buffer, labels)
                davies = davies_bouldin_score(buffer, labels)
                calinski_harabasz = calinski_harabasz_score(buffer, labels)

                results.append({
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'parameters':{'n_clusters': n_clusters, 'n_init': n_init},
                    'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}
                })
                    
            except Exception:
                continue

        

        # end of the time
        elapsed = time() - start_time

        self._parameters_analysis_name = "KMeans_parameter_analysis"
        self._parameters_analysis_results = results.copy()
        self._parameters_analysis_results = pd.DataFrame(self._parameters_analysis_results)

        logger.info(f"KMeans with {n_clusters_params[2]} n_clusters values between {n_clusters_params[0]} and {n_clusters_params[1]} and {n_init_params[2]} n_clusters values between {n_init_params[0]} and {n_init_params[1]}. ")
        

    def DBSCAN_parameters_analysis(self, eps_params=[], min_samples_params=[],metric='euclidean') :
        """
        For the given demand self._demand, performs a clustering using the DBSCAN algorithm with all combinations
        of parameters from the lists eps_params and min_sample_params. The results are saved in list in the following format : 
        [<number of cluster>,[<parameters>], [<scores>]].

        Parameters
        ----------
        eps_params : list
            A list of three values : [start, end, number of values] used to produce a range of values for the eps parameter.
        min_samples_params : list
            A list of three values : [start, end, step] used to produce a range of values for the min_sample parameter.
        metric : string
            the distance metric used to measure similarity between points.
            Default is euclidean.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        eps_values = np.linspace(eps_params[0], eps_params[1],eps_params[2]) # eps is a continuous parameter
        min_samples_values = range(min_samples_params[0], min_samples_params[1], min_samples_params[2]) # min sample is a discrete parameter
        param_grid = list(product(eps_values, min_samples_values))
        
        buffer = self._demand.copy()

        # chronometer
        start_time = time()

        results = []

        for eps, min_samples in tqdm(param_grid, desc="🔍 Optimisation DBSCAN", ncols=100):
            try:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(buffer)

                # Ignore case with 0 or 1 cluster
                if len(set(labels)) <= 1:
                    continue

                silhouette = silhouette_score(buffer, labels)
                davies = davies_bouldin_score(buffer, labels)
                calinski_harabasz = calinski_harabasz_score(buffer, labels)

                results.append({
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'parameters':{'eps': eps, 'min_samples': min_samples},
                    'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}
                })
                    
            except Exception:
                continue

        

        # end of the time
        elapsed = time() - start_time

        self._parameters_analysis_name = "DBSCAN_parameter_analysis"
        logger.info(f"results : {results}")
        self._parameters_analysis_results = results.copy()
        self._parameters_analysis_results = pd.DataFrame(self._parameters_analysis_results)
        logger.info(f"_parameters_analysis_results : {self._parameters_analysis_results}")

        logger.info(f"DBSCAN with {eps_params[2]} eps values between {eps_params[0]} and {eps_params[1]}, min samples values between {min_samples_params[0]} and {min_samples_params[1]} with step value {min_samples_params[2]}. ")


    def SpectralClustering_parameters_analysis(self, n_clusters_params=[], gamma_params=[]):
        """
        For the given demand and self._demand, performs a clustering using the spectral clustering algorithm with all combinations
        of parameters and the default affinity argument ('rbf') from the lists n_clusters_params and gamma_params. The results are saved in list in the following format : 
        [<number of cluster>,[<parameters>], [<scores>]].

        Parameters
        ----------
        n_clusters_params : list
            A list of three values : [start, end, step] used to produce a range of values for the n_clusters parameter.
        gamma_params: list : list
            A list of three values : [start, end, number of values] used to produce a range of values for the gamma parameters.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        n_clusters_values = range(n_clusters_params[0], n_clusters_params[1], n_clusters_params[2])
        gamma_values = np.linspace(gamma_params[0], gamma_params[1], gamma_params[2])
        assign_labels_values = ["kmeans", "discretize"]
        param_grid = list(product(n_clusters_values, gamma_values,assign_labels_values))

        buffer = self._demand.copy()

        # chronometer
        start_time = time()

        results = []

        for n_clusters, gamma, assign_labels in tqdm(param_grid, desc="🔍 Optimisation SpectralClustering", ncols=100):
            try:
                    model = SpectralClustering(n_clusters=n_clusters, gamma=gamma, affinity='nearest_neighbors', assign_labels=assign_labels, random_state=42)
                    labels = model.fit_predict(buffer)
            

                    # Ignore case with 0 or 1 cluster
                    if len(set(labels)) <= 1:
                 
                        continue

                    silhouette = silhouette_score(buffer, labels)
                    davies = davies_bouldin_score(buffer, labels)
                    calinski_harabasz = calinski_harabasz_score(buffer, labels)

                    results.append({
                        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                        'parameters':{'n_clusters': n_clusters, 'gamma': gamma, 'assign_labels':assign_labels},
                        'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}
                    })
        

            except Exception:
                raise
                continue

        self._parameters_analysis_name = "SpectralClustering_parameter_analysis"
        self._parameters_analysis_results = results.copy()
        logger.info(f"_parameters_analysis_results {self._parameters_analysis_results}")
        self._parameters_analysis_results = pd.DataFrame(self._parameters_analysis_results)

        logger.info(f"SpectralClustering with n_clusters values between {n_clusters_params[0]} and {n_clusters_params[1]} with step value {n_clusters_params[2]}, {gamma_params[2]} gamma values between {gamma_params[0]} and {gamma_params[1]}. ")


    def AffinityPropagation_parameters_analysis(self, preference_params=[], damping_params=[]):
        """
        For the given demand and self._demand, performs a clustering using the affinity propagation algorithm with all combinations
        of parameters from the lists preference_params and damping_params. The results are saved in list in the following format : 
        [<number of cluster>,[<parameters>], [<scores>]].

        Parameters
        ----------
        preference_params : list
            A list of three values : [start, end, number of values] used to produce a range of values for the preference parameter.
        damping_params: list : list
            A list of three values : [start, end, number of values] used to produce a range of values for the damping parameters.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        preference_values = np.linspace(preference_params[0], preference_params[1], preference_params[2])
        damping_values = np.linspace(damping_params[0], damping_params[1], damping_params[2])
        param_grid = list(product(preference_values, damping_values))

        buffer = self._demand.copy()

        # chronometer
        start_time = time()

        results = []

        for  preference, damping, in  tqdm(param_grid, desc="🔍 Optimisation AffinityPropagation", ncols=100):
            try:
                    model = model = AffinityPropagation(preference=preference, damping=damping, random_state=42)
                    labels = model.fit_predict(buffer)

                    # Ignore case with 0 or 1 cluster
                    if len(set(labels)) <= 1:
                        continue
                        
                    silhouette = silhouette_score(buffer, labels)
                    davies = davies_bouldin_score(buffer, labels)
                    calinski_harabasz = calinski_harabasz_score(buffer, labels)

                    results.append({
                        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                        'parameters':{'preference': preference, 'damping': damping},
                        'scores':{'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz' : calinski_harabasz}
                    })
        

            except Exception:
                raise
                continue

        self._parameters_analysis_name = "AffinityPropagation_parameter_analysis"
        self._parameters_analysis_results = results.copy()
        self._parameters_analysis_results = pd.DataFrame(self._parameters_analysis_results)

        logger.info(f"AffinityPropagation with {preference_params[2]} preference values between {preference_params[0]} and {preference_params[1]} and {damping_params[2]} damping values between {damping_params[0]} and {damping_params[1]}. ")

    def MeanShift_parameters_analysis(self, quantile_params=[]):
        """
        For the given demand self._demand, performs a clustering using the MeanShift algorithm
        with all quantile values from quantile_params. All possible options for bin_seeding and cluster_all
        are tested automatically. The results are saved in list in the following format:
        [<number of clusters>, [<parameters>], [<scores>]].

        Parameters
        ----------
        quantile_params : list
        A list of three values: [start, end, step] used to produce a range of values for
        the quantile parameter in bandwidth estimation.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()

        # Paramètres à explorer
        quantile_values = np.arange(quantile_params[0], quantile_params[1], quantile_params[2])
        bin_seeding_options = [True, False]
        cluster_all_options = [True, False]

        param_grid = [(q, b, c) for q in quantile_values for b in bin_seeding_options for c in cluster_all_options]

        results = []
        start_time = time()

        for quantile, bin_seed, cluster_all in tqdm(param_grid, desc="🔍 Optimisation MeanShift", ncols=100):
            try:
                bandwidth = estimate_bandwidth(buffer, quantile=quantile, n_samples=len(buffer))
                if bandwidth <= 0 or np.isnan(bandwidth):
                    continue

                model = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seed, cluster_all=cluster_all)
                labels = model.fit_predict(buffer)

                # Ignore les cas dégénérés
                if len(set(labels)) <= 1:
                    continue

                silhouette = silhouette_score(buffer, labels)
                davies = davies_bouldin_score(buffer, labels)
                calinski = calinski_harabasz_score(buffer, labels)

                results.append({
                    'n_clusters': len(set(labels)),
                    'parameters': {'bandwidth_quantile': quantile, 'bandwidth': bandwidth,
                                    'bin_seeding': bin_seed, 'cluster_all': cluster_all},
                    'scores': {'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz': calinski}
                })
            except Exception as e:
                logger.warning(f"MeanShift failed for quantile={quantile}, bin_seeding={bin_seed}, cluster_all={cluster_all}: {e}")
                continue

        elapsed = time() - start_time
        logger.info(f"MeanShift parameter analysis completed in {elapsed:.2f} seconds.")

        self._parameters_analysis_name = "MeanShift_parameter_analysis"
        self._parameters_analysis_results = pd.DataFrame(results)

        logger.info(f"MeanShift with quantile_values values between {quantile_values[0]} and {quantile_values[1]} with step value {quantile_values[2]}.")



    def Agglomerative_parameters_analysis(self, n_clusters_params=[]):
        """
        For the given demand self._demand, performs a clustering using the Agglomerative Clustering algorithm
        with all values of n_clusters from n_clusters_params. All possible linkage and affinity combinations
        are tested automatically. The results are saved in list in the following format:
        [<number of clusters>, [<parameters>], [<scores>]].

        Parameters
        ----------
        n_clusters_params : list
            A list of three values : [start, end, step] used to produce a range of values for the n_clusters parameter.
        """

        # check
        if self._demand.size == 0:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")

        buffer = self._demand.copy()
        n_clusters_values = range(n_clusters_params[0], n_clusters_params[1], n_clusters_params[2])

        # Toutes les combinaisons possibles
        linkage_options = ['ward', 'complete', 'average', 'single']
        metric_options = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']

        # Certaines combinaisons sont invalides → ward ne fonctionne qu’avec euclidean
        param_grid = [
            (n, linkage, metric)
            for n in n_clusters_values
            for linkage in linkage_options
            for metric in metric_options
            if not (linkage == 'ward' and metric != 'euclidean')
        ]

        results = []
        start_time = time()

        for n_clusters, linkage, metric in tqdm(param_grid, desc="🔍 Optimisation Agglomerative", ncols=100):
            try:
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
                labels = model.fit_predict(buffer)

                if len(set(labels)) <= 1:
                    continue

                silhouette = silhouette_score(buffer, labels)
                davies = davies_bouldin_score(buffer, labels)
                calinski = calinski_harabasz_score(buffer, labels)

                results.append({
                    'n_clusters': n_clusters,
                    'parameters': {'n_clusters': n_clusters, 'linkage': linkage, 'metric': metric},
                    'scores': {'silhouette': silhouette, 'davies_bouldin': davies, 'calinski_harabasz': calinski}
                })
            except Exception as e:
                logger.warning(f"Agglomerative failed for n_clusters={n_clusters}, linkage={linkage}, metric={metric}: {e}")
                continue

        elapsed = time() - start_time
        logger.info(f"Agglomerative Clustering parameter analysis completed in {elapsed:.2f} seconds.")

        self._parameters_analysis_name = "Agglomerative_parameter_analysis"
        self._parameters_analysis_results = pd.DataFrame(results)

        logger.info(f"Agglomerative Clustering with n_clusters_values values between {n_clusters_values[0]} and {n_clusters_values[1]} with step value {n_clusters_values[2]}.")



    # variations


    # display

    def display_clusters_on_map(self):
        """
        Diplay the clusters on a map without the temporal dimension.

        No parameters
        """

        # check
        if self._labels.empty:
            logger.error("_labels is empty.")
            raise ValueError("_labels is empty.")
            
        demand = pd.read_csv(self._path, sep=';')
        buffer = demande.copy()
        labels = self._labels.copy()

        # formattage, separation en deux geodataframes
        buffer["DEPARTURE"] = buffer["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        buffer["DEPARTURE"] = pd.to_datetime(buffer["DEPARTURE"], format='%H:%M:%S')
        buffer["DEPARTURE"] = buffer["DEPARTURE"].dt.round("min")

        buffer[["ORIGIN_X","ORIGIN_Y"]] = buffer["ORIGIN"].str.split(' ',expand=True)
        buffer[["DESTINATION_X","DESTINATION_Y"]] = buffer["DESTINATION"].str.split(' ',expand=True)
        buffer[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]] = buffer[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]].astype(float)

        gdf_buffer = buffer.copy()
        gdf_buffer["ORIGIN"] = buffer.apply(lambda row: Point(row["ORIGIN_X"], row["ORIGIN_Y"]), axis=1)
        gdf_buffer["DESTINATION"] = buffer.apply(lambda row: Point(row["DESTINATION_X"], row["DESTINATION_Y"]), axis=1)
        gdf_buffer.drop(columns=["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"], inplace=True)
        crs = "EPSG:32631"

        gdf_origin_buffer = gdf_buffer.drop(columns="DESTINATION")
        gdf_origin_buffer = gpd.GeoDataFrame(gdf_origin_buffer,geometry="ORIGIN",crs=crs)

        gdf_destination_buffer = gdf_buffer.drop(columns="ORIGIN")
        gdf_destination_buffer = gpd.GeoDataFrame(gdf_destination_buffer,geometry="DESTINATION",crs=crs)

        # association avec les labels correspondants
        gdf_origin_buffer["LABEL"] = labels 
        gdf_destination_buffer["LABEL"] = labels

        # agrégations
        gdf_origin_buffer = gdf_origin_buffer.groupby(["ORIGIN","LABEL"]).agg({"ID":"count"}).reset_index() 
        gdf_destination_buffer = gdf_destination_buffer.groupby(["DESTINATION","LABEL"]).agg({"ID":"count"}).reset_index()

        gdf_origin_buffer["ID"] = gdf_origin_buffer["ID"].astype("float")
        gdf_destination_buffer["ID"] = gdf_destination_buffer["ID"].astype("float")

        gdf_origin_buffer = gpd.GeoDataFrame(gdf_origin_buffer,geometry="ORIGIN",crs=crs)
        gdf_destination_buffer = gpd.GeoDataFrame(gdf_destination_buffer,geometry="DESTINATION",crs=crs)

        gdf_origin_buffer.to_crs("EPSG:3857",inplace=True)
        gdf_destination_buffer.to_crs("EPSG:3857",inplace=True)


        # Obtenir la liste des clusters communs aux deux dfs (ou à l'un des deux)
        clusters = sorted(set(gdf_origin_buffer["LABEL"].unique()) | set(gdf_destination_buffer["LABEL"].unique()))

    
        n = len(clusters)
    
        cmap = plt.get_cmap('viridis', n)

        fig, ax = plt.subplots(1,2,figsize=(13,13))

        x = 0
        for cluster in clusters : 
        
            gdf_origin_cluster = gdf_origin_buffer.loc[gdf_origin_buffer["LABEL"]==cluster]
            gdf_destination_cluster = gdf_destination_buffer.loc[gdf_destination_buffer["LABEL"]==cluster]

            gdf_origin_cluster.plot(ax=ax[0],color="none",edgecolor=cmap(x/(n-1)),alpha=1,markersize=gdf_origin_cluster["ID"])
            gdf_destination_cluster.plot(ax=ax[1],color="none",edgecolor=cmap(x/(n-1)),alpha=1,markersize=gdf_destination_cluster["ID"])
        
            x += 1

        ctx.add_basemap(ax[0],source=ctx.providers.OpenStreetMap.Mapnik)
        ctx.add_basemap(ax[1],source=ctx.providers.OpenStreetMap.Mapnik)

        ax[0].axis("off")
        ax[0].set_title("Cluster en fonction des origines")

        ax[1].axis("off")
        ax[1].set_title("Demande en fonction de la destination")
    
        plt.tight_layout()  
        plt.show()

    def display_clusters_on_map_with_time(self, interval="10min"):
        """
        Display the clusters on a map with time.
        Opacity differentiates origin and destination.

        Parameters
        ----------
        interval : string, optional
            steps, time interval for the display.
            interval must respect the following format : "<int>min" or "<int>h"
        """

        # check
        if self._labels.empty:
            logger.error("_labels is empty.")
            raise ValueError("_labels is empty.")
        if not self._path or self._path.strip() == "":
            logger.error("Invalid or null _path.")
            raise ValueError("Invalid or null _path.")

        # assignment
        demand = pd.read_csv(self._path, sep=';')
        buffer = demand.copy()
        labels = self._labels.copy()

        # check
        if buffer.shape[0] != labels.shape[0]:
            logger.error("demand and labels do not have the same size.")
            raise ValueError("demand and labels do not have the same size.")

            
        # preprocessing
        buffer["DEPARTURE"] = buffer["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        buffer["DEPARTURE"] = pd.to_datetime(buffer["DEPARTURE"], format='%H:%M:%S')
        buffer["DEPARTURE"] = buffer["DEPARTURE"].dt.round("min")

        buffer[["ORIGIN_X","ORIGIN_Y"]] = buffer["ORIGIN"].str.split(' ',expand=True)
        buffer[["DESTINATION_X","DESTINATION_Y"]] = buffer["DESTINATION"].str.split(' ',expand=True)
        buffer[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]] = buffer[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]].astype(float)

        gdf_buffer = buffer.copy()
        gdf_buffer["ORIGIN"] = buffer.apply(lambda row: Point(row["ORIGIN_X"], row["ORIGIN_Y"]), axis=1)
        gdf_buffer["DESTINATION"] = buffer.apply(lambda row: Point(row["DESTINATION_X"], row["DESTINATION_Y"]), axis=1)
        gdf_buffer.drop(columns=["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"], inplace=True)
        crs = "EPSG:32631"

        gdf_origin_buffer = gdf_buffer.drop(columns="DESTINATION")
        gdf_origin_buffer = gpd.GeoDataFrame(gdf_origin_buffer, geometry="ORIGIN", crs=crs)
        gdf_origin_buffer = gdf_origin_buffer.to_crs("epsg:4326")
        gdf_origin_buffer["LAT"] = gdf_origin_buffer["ORIGIN"].y
        gdf_origin_buffer["LON"] = gdf_origin_buffer["ORIGIN"].x

        gdf_destination_buffer = gdf_buffer.drop(columns="ORIGIN")
        gdf_destination_buffer = gpd.GeoDataFrame(gdf_destination_buffer, geometry="DESTINATION", crs=crs)
        gdf_destination_buffer = gdf_destination_buffer.to_crs("epsg:4326")
        gdf_destination_buffer["LAT"] = gdf_destination_buffer["DESTINATION"].y
        gdf_destination_buffer["LON"] = gdf_destination_buffer["DESTINATION"].x

        # Association avec les labels correspondants
        gdf_origin_buffer["CLUSTER"] = labels
        gdf_destination_buffer["CLUSTER"] = labels 

        # Agrégation par intervalle de temps
        gdf_origin_buffer["DEPARTURE"] = gdf_origin_buffer["DEPARTURE"].dt.round(interval)
        gdf_origin_buffer = gdf_origin_buffer.groupby(["DEPARTURE", "CLUSTER", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
        gdf_origin_buffer["TYPE"] = "ORIGIN"

        gdf_destination_buffer["DEPARTURE"] = gdf_destination_buffer["DEPARTURE"].dt.round(interval)
        gdf_destination_buffer = gdf_destination_buffer.groupby(["DEPARTURE", "CLUSTER", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
        gdf_destination_buffer["TYPE"] = "DESTINATION"

        # Fusion
        gdf = pd.concat([gdf_origin_buffer, gdf_destination_buffer]).reset_index(drop=True)
        gdf["TIME_STR"] = gdf["DEPARTURE"].dt.strftime("%H:%M:%S")

        clusters = sorted(gdf["CLUSTER"].unique())

        # Clusters colors
        cmap = plt.get_cmap("tab10")
        cluster_colors = {
            cluster: to_hex(cmap(i % cmap.N))
            for i, cluster in enumerate(clusters)
        }

        # animation frames
        time_values = sorted(gdf["TIME_STR"].unique())
        frames = []

        for t in time_values:
            df_t = gdf[gdf["TIME_STR"] == t]
            df_origin = df_t[df_t["TYPE"] == "ORIGIN"]
            df_dest = df_t[df_t["TYPE"] == "DESTINATION"]

            frame = go.Frame(
                data=[
                    go.Scattermap(
                        lat=df_origin["LAT"],
                        lon=df_origin["LON"],
                        mode='markers',
                        marker=dict(
                            size=df_origin["ID"] * 5,
                            color=[cluster_colors[c] for c in df_origin["CLUSTER"]],
                            opacity=0.5,
                            sizemode='area',
                            sizemin=3,
                            symbol='circle'
                        ),
                        name="Origine",
                        hoverinfo='text',
                        hovertext=df_origin.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: ORIGIN", axis=1)
                    ),
                    go.Scattermap(
                        lat=df_dest["LAT"],
                        lon=df_dest["LON"],
                        mode='markers',
                        marker=dict(
                            size=df_dest["ID"] * 5,
                            color=[cluster_colors[c] for c in df_dest["CLUSTER"]],
                            opacity=0.2,
                            sizemode='area',
                            sizemin=3,
                            symbol='circle'
                        ),
                        name="Destination",
                        hoverinfo='text',
                        hovertext=df_dest.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: DESTINATION", axis=1)
                    )
                ],
                name=t
            )
            frames.append(frame)

        # initial figure
        df_init = gdf[gdf["TIME_STR"] == time_values[0]]
        df_init_origin = df_init[df_init["TYPE"] == "ORIGIN"]
        df_init_dest = df_init[df_init["TYPE"] == "DESTINATION"]

        fig = go.Figure(
            data=[
                go.Scattermap(
                    lat=df_init_origin["LAT"],
                    lon=df_init_origin["LON"],
                    mode='markers',
                    marker=dict(
                        size=df_init_origin["ID"] * 5,
                        color=[cluster_colors[c] for c in df_init_origin["CLUSTER"]],
                        opacity=0.5,
                        sizemode='area',
                        sizemin=3,
                        symbol='circle'
                    ),
                    name="Origine",
                    hoverinfo='text',
                    hovertext=df_init_origin.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: ORIGIN", axis=1)
                ),
                go.Scattermap(
                    lat=df_init_dest["LAT"],
                    lon=df_init_dest["LON"],
                    mode='markers',
                    marker=dict(
                        size=df_init_dest["ID"] * 5,
                        color=[cluster_colors[c] for c in df_init_dest["CLUSTER"]],
                        opacity=0.2,
                        sizemode='area',
                        sizemin=3,
                        symbol='circle'
                    ),
                    name="Destination",
                    hoverinfo='text',
                    hovertext=df_init_dest.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: DESTINATION", axis=1)
                )
            ],
            layout=go.Layout(
                height=1000,
                width=1000,
                title="Clusters : dimension spatio-temporelle",
                autosize=True,
                hovermode='closest',
                map=dict(
                    center=dict(lat=gdf["LAT"].mean(), lon=gdf["LON"].mean()),
                    zoom=11,
                    style="streets",  # style map libre compatible, par défaut "streets", "dark", "light", "outdoors", "satellite" etc.
                ),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=1.05,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 500, "redraw": True},
                                           "fromcurrent": True}]),
                             dict(label="Pause",
                                  method="animate",
                                  args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                 "mode": "immediate",
                                                 "transition": {"duration": 0}}])]
                )],
                sliders=[dict(
                    steps=[dict(method='animate',
                                args=[[f.name],
                                      dict(mode='immediate',
                                           frame=dict(duration=500, redraw=True),
                                           transition=dict(duration=0))],
                                label=f.name) for f in frames],
                    transition=dict(duration=0),
                    x=0,
                    y=0,
                    currentvalue=dict(font=dict(size=12), prefix="Heure: ", visible=True, xanchor='center'),
                    len=1.0
                )]
            ),
            frames=frames
        )

        # legend
        for cluster, color in cluster_colors.items():
            fig.add_trace(go.Scattermap(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"Cluster {cluster}"
            ))

        fig.show()


    def display_cluster_on_map_with_time(self, cluster_id, interval="10min"):
        """
        Display one clister on a map with time.
        Opacity differentiates origin and destination.

        Parameters
        ----------
        cluster_id : int
            identifier of the cluster to display
        interval : string, optional
            steps, time interval for the display.
            interval must respect the following format : "<int>min" or "<int>h"
        """

        # check
        if not self._path or self._path.strip() == "":
            logger.error("Invalid or null _path.")
            raise ValueError("Invalid or null _path.")

        # assignment
        demand = pd.read_csv(self._path, sep=';')
        buffer = demand.copy()
        labels = self._labels.copy()

        # check
        if buffer.shape[0] != labels.shape[0]:
            logger.error("demand and labels do not have the same size.")
            raise ValueError("demand and labels do not have the same size.")

        # preprocessing
        buffer["DEPARTURE"] = buffer["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        buffer["DEPARTURE"] = pd.to_datetime(buffer["DEPARTURE"], format='%H:%M:%S')
        buffer["DEPARTURE"] = buffer["DEPARTURE"].dt.round("min")

        buffer[["ORIGIN_X","ORIGIN_Y"]] = buffer["ORIGIN"].str.split(' ',expand=True)
        buffer[["DESTINATION_X","DESTINATION_Y"]] = buffer["DESTINATION"].str.split(' ',expand=True)
        buffer[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]] = buffer[["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"]].astype(float)

        gdf_buffer = buffer.copy()
        gdf_buffer["ORIGIN"] = buffer.apply(lambda row: Point(row["ORIGIN_X"], row["ORIGIN_Y"]), axis=1)
        gdf_buffer["DESTINATION"] = buffer.apply(lambda row: Point(row["DESTINATION_X"], row["DESTINATION_Y"]), axis=1)
        gdf_buffer.drop(columns=["ORIGIN_X","ORIGIN_Y","DESTINATION_X","DESTINATION_Y"], inplace=True)
        crs = "EPSG:32631"

        gdf_origin_buffer = gdf_buffer.drop(columns="DESTINATION")
        gdf_origin_buffer = gpd.GeoDataFrame(gdf_origin_buffer, geometry="ORIGIN", crs=crs)
        gdf_origin_buffer = gdf_origin_buffer.to_crs("epsg:4326")
        gdf_origin_buffer["LAT"] = gdf_origin_buffer["ORIGIN"].y
        gdf_origin_buffer["LON"] = gdf_origin_buffer["ORIGIN"].x

        gdf_destination_buffer = gdf_buffer.drop(columns="ORIGIN")
        gdf_destination_buffer = gpd.GeoDataFrame(gdf_destination_buffer, geometry="DESTINATION", crs=crs)
        gdf_destination_buffer = gdf_destination_buffer.to_crs("epsg:4326")
        gdf_destination_buffer["LAT"] = gdf_destination_buffer["DESTINATION"].y
        gdf_destination_buffer["LON"] = gdf_destination_buffer["DESTINATION"].x

        # labels
        gdf_origin_buffer["CLUSTER"] = labels
        gdf_destination_buffer["CLUSTER"] = labels

        # check
        unique_clusters = set(gdf_origin_buffer["CLUSTER"].unique()).union(set(gdf_destination_buffer["CLUSTER"].unique()))
        if cluster_id not in unique_clusters:
            logger.error(f"Erreur : Le cluster {cluster_id} n'existe pas dans les labels.")
            raise ValueError(f"Erreur : Le cluster {cluster_id} n'existe pas dans les labels.")

        # filters 
        gdf_origin_buffer = gdf_origin_buffer[gdf_origin_buffer["CLUSTER"] == cluster_id]
        gdf_destination_buffer = gdf_destination_buffer[gdf_destination_buffer["CLUSTER"] == cluster_id]

        # agregates
        gdf_origin_buffer["DEPARTURE"] = gdf_origin_buffer["DEPARTURE"].dt.round(interval)
        gdf_origin_buffer = gdf_origin_buffer.groupby(["DEPARTURE", "CLUSTER", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
        gdf_origin_buffer["TYPE"] = "ORIGIN"

        gdf_destination_buffer["DEPARTURE"] = gdf_destination_buffer["DEPARTURE"].dt.round(interval)
        gdf_destination_buffer = gdf_destination_buffer.groupby(["DEPARTURE", "CLUSTER", "LAT", "LON"]).agg({"ID": "count"}).reset_index()
        gdf_destination_buffer["TYPE"] = "DESTINATION"

        # fuses
        gdf = pd.concat([gdf_origin_buffer, gdf_destination_buffer]).reset_index(drop=True)
        gdf["TIME_STR"] = gdf["DEPARTURE"].dt.strftime("%H:%M:%S")

        # color
        cmap = plt.get_cmap("tab10")
        cluster_colors = {cluster_id: to_hex(cmap(0))}

        # animation frames
        time_values = sorted(gdf["TIME_STR"].unique())
        frames = []

        for t in time_values:
            df_t = gdf[gdf["TIME_STR"] == t]
            df_origin = df_t[df_t["TYPE"] == "ORIGIN"]
            df_dest = df_t[df_t["TYPE"] == "DESTINATION"]

            frame = go.Frame(
                data=[
                    go.Scattermap(
                        lat=df_origin["LAT"],
                        lon=df_origin["LON"],
                        mode='markers',
                        marker=dict(
                            size=df_origin["ID"] * 5,
                            color=cluster_colors[cluster_id],
                            opacity=0.5,
                            sizemode='area',
                            sizemin=3,
                            symbol='circle'
                        ),
                        name="Origine",
                        hoverinfo='text',
                        hovertext=df_origin.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: ORIGIN", axis=1)
                    ),
                    go.Scattermap(
                        lat=df_dest["LAT"],
                        lon=df_dest["LON"],
                        mode='markers',
                        marker=dict(
                            size=df_dest["ID"] * 5,
                            color=cluster_colors[cluster_id],
                            opacity=0.2,
                            sizemode='area',
                            sizemin=3,
                            symbol='circle'
                        ),
                        name="Destination",
                        hoverinfo='text',
                        hovertext=df_dest.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: DESTINATION", axis=1)
                    )
                ],
                name=t
            )
            frames.append(frame)

        # Initial figure
        df_init = gdf[gdf["TIME_STR"] == time_values[0]]
        df_init_origin = df_init[df_init["TYPE"] == "ORIGIN"]
        df_init_dest = df_init[df_init["TYPE"] == "DESTINATION"]

        fig = go.Figure(
            data=[
                go.Scattermap(
                    lat=df_init_origin["LAT"],
                    lon=df_init_origin["LON"],
                    mode='markers',
                    marker=dict(
                        size=df_init_origin["ID"] * 5,
                        color=cluster_colors[cluster_id],
                        opacity=0.5,
                        sizemode='area',
                        sizemin=3,
                        symbol='circle'
                    ),
                    name="Origine",
                    hoverinfo='text',
                    hovertext=df_init_origin.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: ORIGIN", axis=1)
                ),
                go.Scattermap(
                    lat=df_init_dest["LAT"],
                    lon=df_init_dest["LON"],
                    mode='markers',
                    marker=dict(
                        size=df_init_dest["ID"] * 5,
                        color=cluster_colors[cluster_id],
                        opacity=0.2,
                        sizemode='area',
                        sizemin=3,
                        symbol='circle'
                    ),
                    name="Destination",
                    hoverinfo='text',
                    hovertext=df_init_dest.apply(lambda row: f"Cluster: {row['CLUSTER']}<br>Count: {row['ID']}<br>Type: DESTINATION", axis=1)
                )
            ],
            layout=go.Layout(
                height=1000,
                width=1000,
                title=f"Cluster {cluster_id} : dimension spatio-temporelle",
                autosize=True,
                hovermode='closest',
                map=dict(
                    center=dict(lat=gdf["LAT"].mean(), lon=gdf["LON"].mean()),
                    zoom=11,
                    style="streets",
                ),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=1.05,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 500, "redraw": True},
                                               "fromcurrent": True}]),
                             dict(label="Pause",
                                  method="animate",
                                  args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                 "mode": "immediate",
                                                 "transition": {"duration": 0}}])]
                )],
                sliders=[dict(
                    steps=[dict(method='animate',
                                args=[[f.name],
                                      dict(mode='immediate',
                                           frame=dict(duration=500, redraw=True),
                                           transition=dict(duration=0))],
                                label=f.name) for f in frames],
                    transition=dict(duration=0),
                    x=0,
                    y=0,
                    currentvalue=dict(font=dict(size=12), prefix="Heure: ", visible=True, xanchor='center'),
                    len=1.0
                )]
            ),
            frames=frames
        )

        fig.show()


    def display_parameters_analysis_results_by_evaluations(self):
        """
        Displays parameter analysis results in bar plots for each evaluation metric.
        """


        if self._parameters_analysis_results.empty:
            logger.error("_parameters_analysis_results is empty.")
            raise ValueError("_parameters_analysis_results is empty.")

        buffer = self._parameters_analysis_results.copy()

        def str_to_dict_safe(s):
            """
            Convertit une chaîne représentant un dictionnaire en dict Python pur.
            Les valeurs sont converties en float ou int natifs.
            """
            if isinstance(s, dict):
                # déjà un dict, juste convertir les types si besoin
                return {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                        for k, v in s.items()}

            if isinstance(s, str):
                try:
                    d = ast.literal_eval(s)  # tente de transformer la chaîne en dict
                    if not isinstance(d, dict):
                        raise ValueError("Ce n'est pas un dict")
                    # convertir les types numériques en natifs Python
                    d = {k: float(v) if isinstance(v, (np.floating, float)) 
                         else int(v) if isinstance(v, (np.integer, int)) 
                         else v
                         for k, v in d.items()}
                    return d
                except (ValueError, SyntaxError):
                    raise ValueError(f"Impossible de convertir la chaîne en dict : {s}")
            raise TypeError(f"Type non supporté : {type(s)}")

       
        buffer["scores"] = buffer["scores"].apply(str_to_dict_safe)
        buffer["parameters"] = buffer["parameters"].apply(str_to_dict_safe)

        df_scores = buffer.join(pd.json_normalize(buffer["scores"]))

        def select_best_per_group(df):
            result = []
            for n, group in df.groupby("n_clusters"):
                max_sil = group["silhouette"].max()
                min_db = group["davies_bouldin"].min()
                max_ch = group["calinski_harabasz"].max()

                mask = (
                    (group["silhouette"] == max_sil)
                    | (group["davies_bouldin"] == min_db)
                    | (group["calinski_harabasz"] == max_ch)
                )
                result.append(group[mask])
            return pd.concat(result)

        df_best = select_best_per_group(df_scores)


        fig, axes = plt.subplots(1, 3, figsize=(30, 12))  # 👈 plus large et haute
        metrics = [
            ("silhouette", "Silhouette (↑)", "skyblue"),
            ("davies_bouldin", "Davies-Bouldin (↓)", "orange"),
            ("calinski_harabasz", "Calinski-Harabasz (↑)", "green"),
        ]

        for ax, (metric, title, color) in zip(axes, metrics):
            # indices espacés pour éviter les chevauchements
            #x_positions = range(len(df_best))
            #bars = ax.bar(x_positions, df_best[metric], color=color, width=0.4)

            gap = 2  # facteur d'espacement, plus grand = plus d'espace
            x_positions = [i * gap for i in range(len(df_best))]

            bars = ax.bar(x_positions, df_best[metric],color=color, width=0.5)  # largeur conservée

            ax.set_xlim(min(x_positions) - gap/2, max(x_positions) + gap/2)
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Configurations", fontsize=14)
            ax.set_ylabel("Score", fontsize=14)

            # Légendes au-dessus de chaque barre
            for i, (bar, (_, row)) in enumerate(zip(bars, df_best.iterrows())):
                params_txt = ", ".join(f"{k}={v}" for k, v in row["parameters"].items())
                label = f"n={row['n_clusters']} | {params_txt}"
                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.05 * abs(height),  # espace vertical + large
                    label,
                    ha="center",
                    va="bottom",
                    rotation=90,        # 👈 rotation plus douce
                    fontsize=10,        # 👈 police plus grande
                    wrap=True
                )
            
            ax.set_xticks(x_positions)
            ax.set_xticks([])
            #ax.set_xticklabels([f"#{i}" for i in x_positions], fontsize=12)
            ax.margins(x=0.3)  # 👈 espace horizontal encore plus large

        plt.suptitle(self._parameters_analysis_name, fontsize=18, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


    
    # getters

    def get_path(self):
        # check
        if not self.path or self.path.strip() == "":
            logger.error("Invalid or null path")
            raise ValueError("Invalid or null path")
        return self.path

    def get_original_demand(self):
        # check
        if not self._original_demand.empty:
            logger.info("_original_demand is empty.")
        return self._original_demand
    

    def get_demand(self):
        # check
        if not self._demand.empty:
            logger.error("_demand is empty.")
        return self._demand

    def get_labels(self):
        # check
        if not self._labels.empty:
            logger.error("_labels is empty.")
        return self._labels

    def get_parameters_analysis_name(self):
        # check
        if not self._parameters_analysis_name or self._parameters_analysis_name.strip() == "":
            logger.error("Invalid or null _parameters_analysis_name")
            raise ValueError("Invalid or null _parameters_analysis_name")
        return self._parameters_analysis_name


    def get_parameters_analysis_results(self):
        # check
        if self._parameters_analysis_results.empty:
            logger.error("_parameters_analysis_results is empty.")
        return self._parameters_analysis_results

    def get_clustering_name(self):
        # check
        if not self._clustering_name or self._clustering_name.strip() == "":
            logger.error("Invalid or null _clustering_name")
            raise ValueError("Invalid or null _clustering_name")
        return self._clustering_name

    def get_clustering_result(self):
        # check
        if self._clustering_result.empty:
            logger.error("_clustering_result is empty.")
        return self._clustering_result

    