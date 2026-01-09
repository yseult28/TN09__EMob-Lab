

# dependencies

import os 
from pathlib import Path

import math, re

import numpy as np
import pandas as pd

import copy


import logging

from .CopertEstimator import CopertEstimator



logger = logging.getLogger(__name__)

# useful fonctions

def parse_filename(name):
    """
    Parse filenames of the form:
    Origin__Variation_x__Total_Ratio__method_law__param_str__v1-v2-v3

    Parameters
    ----------
    name : str
        File name.

    Return
    ----------  
    result : dict
        Returns a dict ready to become a DataFrame row.
    """


    categories = name.split("__")
    
    if len(categories) < 7:
        raise ValueError("Invalid format : 6 categories are needed.")

    cat0 = categories[0]
    cat1 = categories[1]
    cat2 = categories[2]
    cat3 = categories[3]
    cat4 = categories[4]
    cat5 = categories[5]
    cat6 = categories[6]

    ORIGIN, PERIOD, OPERATION = cat0.split("_")
    PERIOD = PERIOD.split("-")
    START = PERIOD[0]
    END = PERIOD[1]
    
    SCALE, TYPE = cat2.split("_")

    parts3 = cat3.split("_")
    if len(parts3) != 2:
        raise ValueError(f"La catégorie 3 doit être 'method_law', trouvé : {cat3}")
    METHOD, LAW = parts3

    LAW_PARAMETERS = cat4

    TOTAL_RATIO = cat5

    TARGET_CLUSTER, TARGET_RATIO = cat6.split("_")


    result = {
        "ORIGIN": ORIGIN,
        "START": START,
        "END": END,
        "OPERATION": OPERATION,
        "SCALE": SCALE,
        "TYPE": TYPE,
        "METHOD": METHOD,
        "LAW": LAW,
        "LAW_PARAMETERS": LAW_PARAMETERS,
        "TOTAL_RATIO": TOTAL_RATIO,
        "TARGET_CLUSTER": TARGET_CLUSTER,
        "TARGET_CHANGE": TARGET_RATIO,
    }


    return result


class Sensitivity:
    """
    Creates a dataframe for sensitivy analysis purposes from a simulation output directory or from a copert directory

    Attributes
    ----------
    _sensitivity_df : pandas.DataFrame
        Dataframe for sensitivity purposes.
    """

    def __init__(self):
        """
        Sensitivity constructor.

        No parameters
        """

        # assignment
        self._sensitivity_df = pd.DataFrame(columns=[])

        logger.info(f"Sensitivity object built.")

    # Configuration methods

    def create_copert_dfs(self, simulation_outputs_directory, copert_dfs_directory, crs, copert_data_path, period="6min", method="SPEED_MEAN"):
        """
        Creates and saves copert dfs from the simulations outputs saved in simulation_outputs_directory.

        Parameters
        ----------
        simulation_output_directory : str
            Simulation outputs directory.
        copert_dfs_directory : str
            Directory to save the copert dfs
        crs : str
            Projection system of the given simulation
        copert_data_path : str
            Path to the copert data file.
        period : str, optionnal
            Temporal aggregation value (default:"6min").
        method : str, optionnal
            Speed calculation method name, SPEED uses the mean of the SPEED variable, DISTANCE calculates the relative distance value divided 
            the period (default:"SPEED").
        """

        # check
        if not simulation_outputs_directory or simulation_output_directory.strip() == "" : 
            logger.info("Invalid or null simulation outputs directory.")
            raise ValueError("Invalid or null simulation outputs directory.")

        if not copert_dfs_directory or copert_dfs_directory.strip() == "" : 
            logger.info("Invalid or null copert dfs directory.")
            raise ValueError("Invalid or null copert dfs directory.")

        
        folders = [f for f in Path(simulation_outputs_directory).iterdir() if f.is_dir()]

        os.mkdirs(copert_dfs_directory, exist_ok=True)

        for folder in folders :
            
            CE = CopertEstimator(folder, crs, copert_data_path)
            CE.copert_estimation(period, method)
            buffer = CE.get_estimation()
            
            folder_name = Path(folder).stem
            categories_list = folder_name.split("__")
            copert_df_name = (f"Copert_{period}_{method}"
                              f"__{categories_list[1:]}")
            full_path = os.path.join(copert_dfs_directory,f"{copert_df_name}.csv")
            buffer.to_csv(full_path, sep=';', index=False)

        logger.info(f"create_copert_dfs done.")


    # Sensitivity 

    def create_sensitivity_df_from_copert_dfs_directory(self, original_outputs_directory, copert_dfs_directory, period=None, area=None, mobility_services=None,simulation_outputs_directory="", crs="epsg:4326", copert_data_path=""):
        """
        Creates the sensitivity df from the copert dfs directory using given parameters.

        Parameters
        ----------
        original_outputs_directory : str
            Directory to load the original simulation outputs.
        copert_dfs_directory : str
            Directory to load the copert dfs from.
        period : None or [str, str]
            Time period as strings "HH:MM:SS". Only the hour part is used.
            Example: ["08:00:00", "10:30:00"]
            If None → entire dataset.
        area : None or shapely Polygon
            Spatial filtering polygon.
        mobility_services : None or list[str]
            TYPE values to keep.
        original_outputs_path : str
            Path to the original outputs' folder.
        variation_outputs_path : str
            Path to the variations outputs' folder.
        copert_data_path : str
            Path to the copert data file.
        networkmanager : NetworkManager
            NetworkManager object for display purpose, network dfs have to be created.
        """

        # check

        if not copert_dfs_directory or copert_dfs_directory.strip() == "" : 
            logger.info("Invalid or null copert dfs directory.")
            raise ValueError("Invalid or null copert dfs directory.")

        if not original_outputs_directory or original_outputs_directory.strip() == "" : 
            logger.info("Invalid or null original outputs directory.")
            raise ValueError("Invalid or null original outputs directory.")

        CE = CopertEstimator(str(original_outputs_directory), crs, copert_data_path)
        CE.copert_estimation(time_period, method)
        nox_original = CE.get_nox(period, area, mobility_services)
        co2_original = CE.get_co2(period, area, mobility_services)

        name = Path(original_outputs_directory).stem
        row = parse_filename(name)

        row["NOX"] = nox_original
        row["CO2"] = co2_original
        row["NOX_RATIO"] = 1
        row["CO2_RATIO"] = 1

        sensitivity_tab = pd.DataFrame(columns=["ORIGIN","START","END","OPERATION","SCALE","TYPE","METHOD","LAW","LAW_PARAMETERS","TOTAL_RATIO","TARGET_CLUSTER","TARGET_CHANGE","NOX","CO2","NOX_RATIO","CO2_RATIO"])

        sensitivity_tab = pd.DataFrame([row])

        copert_df_files = Path(copert_dfs_directory).glob("*.csv")

        CE = CopertEstimator(simulation_outputs_directory, crs, copert_data_path)

        for file in files :
            name = Path(folder).stem
            row = parse_filename(name)

            CE.load_copert(file)
            
            row["NOX"] = CE.get_nox(period, area, mobility_services)
            row["CO2"] = CE.get_co2(period, area, mobility_services)
            row["NOX_RATIO"] = row["NOX"]/nox_original
            row["CO2_RATIO"] = row["CO2"]/co2_original


            if sensitivity_tab.empty:
                sensitivity_tab = pd.DataFrame([row])
            else:
                sensitivity_tab = pd.concat([sensitivity_tab, pd.DataFrame([row])], ignore_index=True)

        self._sensitivity_df = sensitivity_tab.copy()

        logger.info(f"create_sensitivity_df_from_copert_dfs done with copert_dfs_directory : {copert_dfs_directory}, period : {period}, area : {area}, mobility_services : {mobility_services},simulation_outputs_directory : {simulation_outputs_directory}, crs : {crs}, copert_data_path : {copert_data_path}.")


    def create_sensitivity_df_from_simulation_outputs_directory(self, original_outputs_directory, simulation_outputs_directory, period=None, area=None, mobility_services=None, crs="epsg:4326", copert_data_path="", time_period="6min", method="SPEED_MEAN"):
        """
        Creates the sensitivity df from the simulation outputs directory using given parameters.

        Parameters
        ----------
        original_outputs_directory : str
            Directory to load the original simulation outputs.
        simulation_outputs_directory : str
            Directory to load the simulation outputs from.
        period : None or [str, str]
            Time period as strings "HH:MM:SS". Only the hour part is used.
            Example: ["08:00:00", "10:30:00"]
            If None → entire dataset.
        area : None or shapely Polygon
            Spatial filtering polygon.
        mobility_services : None or list[str]
            TYPE values to keep.
        original_outputs_path : str
            Path to the original outputs' folder.
        variation_outputs_path : str
            Path to the variations outputs' folder.
        copert_data_path : str
            Path to the copert data file.
        networkmanager : NetworkManager
            NetworkManager object for display purpose, network dfs have to be created.
        """

        # check

        if not simulation_outputs_directory or simulation_outputs_directory.strip() == "" : 
            logger.info("Invalid or null simulation outputs directory.")
            raise ValueError("Invalid or null simulation outputs directory.")

        if not original_outputs_directory or original_outputs_directory.strip() == "" : 
            logger.info("Invalid or null original outputs directory.")
            raise ValueError("Invalid or null original outputs directory.")

        CE = CopertEstimator(str(original_outputs_directory), crs, copert_data_path)
        CE.copert_estimation(time_period, method)
        nox_original = CE.get_nox(period, area, mobility_services)
        co2_original = CE.get_co2(period, area, mobility_services)

        name = Path(original_outputs_directory).stem
        row = parse_filename(name)

        row["NOX"] = nox_original
        row["CO2"] = co2_original
        row["NOX_RATIO"] = 1
        row["CO2_RATIO"] = 1

        sensitivity_tab = pd.DataFrame(columns=["ORIGIN","START","END","OPERATION","SCALE","TYPE","METHOD","LAW","LAW_PARAMETERS","TOTAL_RATIO","TARGET_CLUSTER","TARGET_CHANGE","NOX","CO2","NOX_RATIO","CO2_RATIO"])

        sensitivity_tab = pd.DataFrame([row])

        folders = [f for f in Path(simulation_outputs_directory).iterdir() if f.is_dir() and not f.name.startswith(".")]


        for folder in folders :

            logger.info(f"{folder}")
            CE = CopertEstimator(str(folder), crs, copert_data_path)
            CE.copert_estimation(time_period, method)
            
            name = Path(folder).name
            row = parse_filename(name)
            
            row["NOX"] = CE.get_nox(period, area, mobility_services)
            row["CO2"] = CE.get_co2(period, area, mobility_services)
            row["NOX_RATIO"] = row["NOX"]/nox_original
            row["CO2_RATIO"] = row["CO2"]/co2_original

            if sensitivity_tab.empty:
                sensitivity_tab = pd.DataFrame([row])
            else:
                sensitivity_tab = pd.concat([sensitivity_tab, pd.DataFrame([row])], ignore_index=True)

    
        self._sensitivity_df = sensitivity_tab.copy()

        logger.info(f"create_sensitivity_df_from_simulation_outputs_file done with copert_dfs_directory : {simulation_outputs_directory}, period : {period}, area : {area}, mobility_services : {mobility_services}, crs : {crs}, copert_data_path : {copert_data_path}, time_period : {time_period}, method : {method}.")
            
            

        
    


    # getters

    def get_sensitivity_df(self):
        return self._sensitivity_df.copy()

    