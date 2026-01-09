# dependencies
import os 
from pathlib import Path

import math, re

import numpy as np
import pandas as pd

import copy


import logging




logger = logging.getLogger(__name__)




from .Copert import Copert




class ClusteringSensitivity:

    # Constructor
    
    def __init__(self,original_outputs_path="",variation_outputs_path="",copert_data_path="",network_manager=None):
        """
        Sensitivity's constructor.
    
        Parameters
        ----------
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
        if not original_outputs_path or str(original_outputs_path).strip() == "": 
            logger.error("Invalid or null original_outputs_path.")
            raise ValueError("Invalid or null original_outputs_path.")
        if not variation_outputs_path or str(variation_outputs_path).strip() == "": 
            logger.error("Invalid or null variation_outputs_path.")
            raise ValueError("Invalid or null variation_outputs_path.")
        if not copert_data_path or copert_data_path.strip == "": 
            logger.error("Invalid or null copert_path.")
            raise ValueError("Invalid or null copert_path.")
        if not network_manager:
            logger.error("Null network_manager.")
            raise ValueError("Null network_manager.")

        # assignment
        self._original_outputs_path = original_outputs_path 
        self._variation_outputs_path = variation_outputs_path
        self._copert_data_path = copert_data_path
        self._network_manager = network_manager
        self._original_copert = None
        self._variation_copert= []
        self._variation_copert_df = []
        self._profile_parameters = ["Summer", "Autumn", "Winter", "Spring", "Holyday", "Not Holyday"]
        self._events_parameters = ["METRO", "METRO_TRAM", "METRO_TRAM_BUS", "METRO_BUS", "TRAM", "BUS", "TRAM_BUS", "None"]

        # loads 
        self.load_original_copert()
        self.load_variation_copert()

        logger.info("Sensitivity initialized.")



    # configuration's methods

    def change_sensitivity(self,original_outputs_path="",variation_outputs_path="",copert_data_path="",network_manager=None):
        """
        Changes the attributes' values of the sensitivity object.
    
        Parameters
        ----------
        original_outputs_path : string
            Path to the original outputs' folder.
        variation_outputs_path : string
            Path to the variations outputs' folder.
        copert_data_path : string
            Path to the copert data file.
        """
        
        # check 
        if not original_outputs_path or str(original_outputs_path).strip() == "": 
            logger.error("Invalid or null original_outputs_path.")
            raise ValueError("Invalid or null original_outputs_path.")
        if not variation_outputs_path or str(variation_outputs_path).strip() == "": 
            logger.error("Invalid or null variation_outputs_path.")
            raise ValueError("Invalid or null variation_outputs_path.")
        if not copert_data_path or copert_data_path.strip == "": 
            logger.error("Invalid or null copert_path.")
            raise ValueError("Invalid or null copert_path.")
        if not network_manager:
            logger.error("Null network_manager.")
            raise ValueError("Null network_manager.")

        # assignment
        self._original_outputs_path = original_outputs_path 
        self._variation_outputs_path = variation_outputs_path
        self._copert_data_path = copert_data_path
        self._network_manager = network_manager
        self._original_copert = None
        self._variation_copert= []

        self._  

        # loads 
        self.load_parameters_from_csvs()
        self.load_original_copert()
        self.load_variation_copert()

        logger.info("Sensitivity changed.")
        

    def load_original_copert(self):
        """
        Create a Copert object linked to the original outputs.
    
        No parameters
        """
        
        # check 
        if not self._original_outputs_path or str(self._original_outputs_path).strip() == "": 
            log.error("Invalid or null original_outputs_path.")
            raise ValueError("Invalid or null original_outputs_path.")

        self._original_copert = Copert(self._original_outputs_path, self._network_manager, self._copert_data_path)
        self._original_copert.create_copert()

        logger.info("original_copert created.")

    
    def load_variation_copert(self):
        """
        Creates Copert objects linked to the variations outputs.
    
        No parameters
        """
        
        # check 
        if not self._variation_outputs_path or str(self._variation_outputs_path).strip() == "": 
            log.error("Invalid or null variation_outputs_path.")
            raise ValueError("Invalid or null variation_outputs_path.")

        path = Path(self._variation_outputs_path)
        folders = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]

        for folder in folders:
            self._variation_copert.append(Copert(folder, self._network_manager, self._copert_data_path))
            
        for copert in self._variation_copert:
            copert.create_copert()

        logger.info("variation_copert created.")

    def load_parameters_from_csvs(self):
        """
        Loads profile parameters and event distributions from CSV files located in
        `self._path / "parameters" /`.

        Expected files:
            - profile_parameters.csv : describes each profile's parameters and their probabilities
            - events_distributions.csv : describes the probability of each event

        Updates internal attributes:
            - self._profile_parameters
            - self._events_distributions
            - self._events_list
        """

        # Base directory for parameter files
        parameters_dir = os.path.join(self._path, "parameters")
        if not os.path.exists(parameters_dir):
            logger.error(f"Parameters directory not found: {parameters_dir}")
            raise FileNotFoundError(f"Parameters directory not found: {parameters_dir}")

        # --- 1️⃣ Load profile parameters ---
        profile_params_path = os.path.join(parameters_dir, "profile_parameters.csv")
        if os.path.exists(profile_params_path):
            df_profiles = pd.read_csv(profile_params_path, sep=';|,', engine='python')
            logger.info(f"Loaded profile parameters from {profile_params_path}")

            # The CSV should have one row, each column = profile value
            self._profile_parameters = {
                col: df_profiles[col].dropna().to_list()
                if len(df_profiles) > 1 else df_profiles[col].iloc[0]
                for col in df_profiles.columns
            }

            # Optionally, you might also reconstruct self._profile_list
            self._profile_parameters = list(df_profiles.columns)
        else:
            logger.warning(f"Profile parameters file not found at {profile_params_path}")
            self._profile_parameters = {}
            self._profile_list = []

        # --- 2️⃣ Load event distributions ---
        events_path = os.path.join(parameters_dir, "events_distributions.csv")

        if os.path.exists(events_path):
            df_events = pd.read_csv(events_path, sep=';|,', engine='python')
            logger.info(f"Loaded events distributions from {events_path}")

            # The CSV must have exactly one row
            if len(df_events) != 1:
                logger.warning(
                    f"Expected a single-row dataframe for events_distributions.csv, "
                    f"but got {len(df_events)} rows. Using the first row."
                )

            # Convert the single row into a dictionary
            self._events_distributions = df_events.iloc[0].to_dict()

            # Update the events list (columns names)
            self._events_list = list(df_events.columns)

        else:
            logger.warning(f"Events distributions file not found at {events_path}")
            self._events_distributions = {}
            self._events_list = []

        logger.info("Profile parameters and event distributions successfully loaded.")


    def save_coperts_df(self, path):
        """
        Saves every copert dataframes at path.

        Parameters
        ----------
        path : string
            Directory path to save the copert dataframes.
        """

        # check
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        os.makedirs(path, exist_ok=True)

        outputs_path = Path(self._variation_outputs_path)
        folders = [p for p in outputs_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
        
        paths = []
        for folder in folders : 
            name = folder.name
            full_path = os.path.join(path,f"{name}.csv")
            paths.append(full_path)

        for i in range(len(self._variation_copert)):
            buffer = self._variation_copert[i].get_copert().copy()
            buffer.to_csv(paths[i], sep=';', index=False)

        logger.info(f"Coperts saved at {path}.")

    def load_copert_dfs(self, path):
        """
        loads copert dfs from path

        Parameters
        ----------
        path : string
            Directory path to load the copert dataframes.
        """

        # check
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        files = list(Path(path).glob("*.csv"))
        self._variation_copert_dfs = []
        for file in files : 
            self._variation_copert_dfs.append(pd.read_csv(file, sep=';'))

        logger.info(f"Coperts loaded from {path}.")
        



        

    # analysis methods
    
    

    def create_total_sensitivity_df(self):
        """
        Creates a sensitivity dataframe with binary profile/event encoding in the filename.
        Each line corresponds to one variation (folder) in _variation_outputs_path.

        Returns
        -------
        df : pandas.DataFrame
            The complete sensitivity dataframe.
        """

        # Initialize column names: profiles + events + pollutant metrics
        columns = (
            [p for p in self._profile_parameters] +
            [e for e in self._events_parameters] +
            ["NOX", "CO2", "NOX_RATIO", "CO2_RATIO", "NOX_RATE", "CO2_RATE"]
        )
        df = pd.DataFrame(columns=columns)

        # Original values
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()

        # Add first line: reference (original)
        first_row = {col: 0 for col in columns}
        first_row["NOX"] = original_nox
        first_row["CO2"] = original_co2
        first_row["NOX_RATIO"] = 1
        first_row["CO2_RATIO"] = 1
        first_row["NOX_RATE"] = 0
        first_row["CO2_RATE"] = 0
        df = pd.concat([df, pd.DataFrame([first_row])], ignore_index=True)

        # Regex pattern for parsing filenames
        pattern = re.compile(
            r"^variation_(?P<index>\d+)__(?P<profiles>[0-1_]+)__(?P<events>[0-1_]+)$"
        )

        # List all variation folders
        folders = [f for f in Path(self._variation_outputs_path).iterdir() if f.is_dir()]

        x = 1
        for folder in sorted(folders, key=lambda f: f.name):
            match = pattern.match(folder.name)
            if not match:
                continue

            d = match.groupdict()
            profiles = [int(x) for x in d["profiles"].split("_")]
            events = [int(x) for x in d["events"].split("_")]

            # Build row
            row = {}
            # Fill profiles (binary values in order)
            for i, p in enumerate(self._profile_list):
                row[p] = profiles[i] if i < len(profiles) else 0
            # Fill events
            for j, e in enumerate(self._events_list):
                row[e] = events[j] if j < len(events) else 0

            # Pollutant calculations
            copert = self._variation_copert[x-1]
            nox = copert.get_tot_NOX_per_interval()
            co2 = copert.get_tot_CO2_per_interval()
            row["NOX"] = nox
            row["CO2"] = co2
            row["NOX_RATIO"] = nox / original_nox
            row["CO2_RATIO"] = co2 / original_co2
            row["NOX_RATE"] = (nox - original_nox) / original_nox
            row["CO2_RATE"] = (co2 - original_co2) / original_co2

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            x += 1

        return df

    def create_total_sensitivity_df_from_path(self, path):
        """
        Creates a sensitivity dataframe (like create_total_sensitivity_df)
        from saved Copert CSVs in the given path.

        Parameters
        ----------
        path : str
            Directory where Copert CSVs are stored.
        """

        # Load all Copert CSVs into memory
        self.load_copert_dfs(path)

        # Initialize dataframe
        columns = (
            [p for p in self._profile_parameters] +
            [e for e in self._events_parameters] +
            ["NOX", "CO2", "NOX_RATIO", "CO2_RATIO", "NOX_RATE", "CO2_RATE"]
        )
        

        # Original values
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()

        # First row = reference
        first_row = {col: 0 for col in columns}
        first_row["NOX"] = original_nox
        first_row["CO2"] = original_co2
        first_row["NOX_RATIO"] = 1
        first_row["CO2_RATIO"] = 1
        first_row["NOX_RATE"] = 0
        first_row["CO2_RATE"] = 0
        df = pd.DataFrame([first_row],columns=columns)

        # Regex to parse filenames
        pattern = re.compile(
            r"^variation_(?P<index>\d+)__(?P<profiles>[0-1_]+)__(?P<events>[0-1_]+)$"
        )

        files = list(Path(path).glob("*.csv"))
        files.sort()

        copert = copy.deepcopy(self._original_copert)

        x = 1
        for file in files:
            name = Path(file).stem
            match = pattern.match(name)
            if not match:
                continue

            d = match.groupdict()
            profiles = [int(x) for x in d["profiles"].split("_")]
            events = [int(x) for x in d["events"].split("_")]

            # Construct row
            row = {}
            for i, p in enumerate(self._profile_parameters):
                row[p] = profiles[i] if i < len(profiles) else 0
            for j, e in enumerate(self._events_parameters):
                row[e] = events[j] if j < len(events) else 0

            # Compute pollutant ratios
            copert.load_copert(file)
            nox = copert.get_tot_NOX_per_interval()
            co2 = copert.get_tot_CO2_per_interval()
            row["NOX"] = nox
            row["CO2"] = co2
            row["NOX_RATIO"] = nox / original_nox
            row["CO2_RATIO"] = co2 / original_co2
            row["NOX_RATE"] = (nox - original_nox) / original_nox
            row["CO2_RATE"] = (co2 - original_co2) / original_co2

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            x += 1

        return df


    def create_total_sensitivity_df_ratio_cluster_from_path(self, path):
        """
        Crée un DataFrame de sensibilité à partir des fichiers Copert CSV,
        avec des noms de fichiers du type :
        variante_1__WeightedSampling-Normal__tot-1.0__0-0.9_1-1.061_2-1.013.csv

        Colonnes du DataFrame :
            TYPE, LAW, TOTAL_RATIO, {numéro}_RATIO, NOX, CO2,
            NOX_RATIO, CO2_RATIO, NOX_RATE, CO2_RATE
        """

        # Charger tous les CSV Copert
        self.load_copert_dfs(path)

        # Colonnes principales (fixes)
        base_cols = ["TYPE", "LAW", "TOTAL_RATIO", "TARGET_CLUSTER", "TARGET_RATIO", "NOX", "CO2",
                     "NOX_RATIO", "CO2_RATIO", "NOX_RATE", "CO2_RATE"]
        df = pd.DataFrame(columns=base_cols)

        # Valeurs de référence
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()

        # Fichiers CSV à lire
        files = list(Path(path).glob("*.csv"))
        files.sort()

        #logger.info(f"{len(files)} files.")

        # Regex pour extraire les infos
        pattern = re.compile(
            r"^variante_(?P<index>\d+)__"
            #r"(?P<type>[A-Za-z0-9]+)-(?P<law>[A-Za-z0-9]+)__"
            #r"tot-(?P<total>[\d\.]+)__"
            #r"(?P<target>[\d\-\.]+)__"
            r".*$"
            #r"(?P<ratios>-?\d+(?:\.\d+)?--?\d+(?:\.\d+)?(?:_-?\d+(?:\.\d+)?--?\d+(?:\.\d+)?)*)$"
        )

        pattern = re.compile(
            r"^variation_(?P<index>\d+)__"
            r"(?P<type>[A-Za-z0-9]+)-(?P<law>[A-Za-z0-9]+)__"
            r"tot-(?P<total>[\d\.]+)__"
            r"(?P<target>[\d\-\.]+)__"
            r"(?P<ratios>-?\d+(?:\.\d+)?-?\d+(?:\.\d+)?(?:_-?\d+(?:\.\d+)?-?\d+(?:\.\d+)?)*)$"
        )


        copert = copy.deepcopy(self._original_copert)

        for file in files:
            name = Path(file).stem

            match = pattern.match(name)
            if not match:
                logger.info(name)
                # ignorer les fichiers non conformes
                continue

            d = match.groupdict()
            target = d["target"].split("-")
            row = {
                "TYPE": d["type"],
                "LAW": d["law"],
                "TOTAL_RATIO": float(d["total"]),
                "TARGET_CLUSTER": int(target[0]),
                "TARGET_RATIO": float(target[1])
             }

            logger.info(row)

            # Extraire les ratios individuels : 0-0.9_1-1.061_2-1.013
            ratio_strs = d["ratios"].split("_")
            for r in ratio_strs:
                if "-" in r:
                    key, val = r.split("-")
                    row[f"{key}_RATIO"] = float(val)

            # Charger et calculer les polluants
            copert.load_copert(file)
            nox = copert.get_tot_NOX_per_interval()
            co2 = copert.get_tot_CO2_per_interval()

            row["NOX"] = nox
            row["CO2"] = co2
            row["NOX_RATIO"] = nox / original_nox
            row["CO2_RATIO"] = co2 / original_co2
            row["NOX_RATE"] = (nox - original_nox) / original_nox
            row["CO2_RATE"] = (co2 - original_co2) / original_co2

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

       
        row = {
            "TYPE": "original",
            "LAW": "original",
            "TOTAL_RATIO": 1,
             "TARGET_CLUSTER": -1,
             "TARGET_RATIO": -1,
        }

        for col in df.columns[11:]:
            row[col] = 1


        row["NOX"] = original_nox
        row["CO2"] = original_co2
        row["NOX_RATIO"] = 1
        row["CO2_RATIO"] = 1
        row["NOX_RATE"] = 0
        row["CO2_RATE"] = 0

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)


        

        return df

    def create_total_sensitivity_df_quantity_cluster_from_path(self, path):
        """
        Crée un DataFrame de sensibilité à partir des fichiers Copert CSV,
        avec des noms de fichiers du type :
        variante_1__WeightedSampling-Normal__tot-1.0__0-0.9_1-1.061_2-1.013.csv

        Colonnes du DataFrame :
            TYPE, LAW, TOTAL_RATIO, {numéro}_RATIO, NOX, CO2,
            NOX_RATIO, CO2_RATIO, NOX_RATE, CO2_RATE
        """

        # Charger tous les CSV Copert
        self.load_copert_dfs(path)

        # Colonnes principales (fixes)
        base_cols = ["TYPE", "LAW", "TOTAL_RATIO", "TARGET_CLUSTER", "TARGET_RATIO", "NOX", "CO2",
                     "NOX_RATIO", "CO2_RATIO", "NOX_RATE", "CO2_RATE"]
        df = pd.DataFrame(columns=base_cols)

        # Valeurs de référence
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()

        # Fichiers CSV à lire
        files = list(Path(path).glob("*.csv"))
        files.sort()

        #logger.info(f"{len(files)} files.")

        # Regex pour extraire les infos
        pattern = re.compile(
            r"^variante_(?P<index>\d+)__"
            #r"(?P<type>[A-Za-z0-9]+)-(?P<law>[A-Za-z0-9]+)__"
            #r"tot-(?P<total>[\d\.]+)__"
            #r"(?P<target>[\d\-\.]+)__"
            r".*$"
            #r"(?P<ratios>-?\d+(?:\.\d+)?--?\d+(?:\.\d+)?(?:_-?\d+(?:\.\d+)?--?\d+(?:\.\d+)?)*)$"
        )

        pattern = re.compile(
            r"^variation_(?P<index>\d+)__"
            r"(?P<type>[A-Za-z0-9]+)-(?P<law>[A-Za-z0-9]+)__"
            r"tot-(?P<total>[\d\.]+)__"
            r"(?P<target>[\d\-\.]+)__"
            r"(?P<ratios>-?\d+(?:\.\d+)?-?\d+(?:\.\d+)?(?:_-?\d+(?:\.\d+)?-?\d+(?:\.\d+)?)*)$"
        )


        copert = copy.deepcopy(self._original_copert)

        for file in files:
            name = Path(file).stem

            match = pattern.match(name)
            if not match:
                logger.info(name)
                # ignorer les fichiers non conformes
                continue

            d = match.groupdict()
            target = d["target"].split("-")
            row = {
                "TYPE": d["type"],
                "LAW": d["law"],
                "TOTAL_RATIO": float(d["total"]),
                "TARGET_CLUSTER": int(target[0]),
                "TARGET_RATIO": float(target[1])
             }

            logger.info(row)

            # Extraire les ratios individuels : 0-0.9_1-1.061_2-1.013
            ratio_strs = d["ratios"].split("_")
            for r in ratio_strs:
                if "-" in r:
                    key, val = r.split("-")
                    row[f"{key}_RATIO"] = float(val)

            # Charger et calculer les polluants
            copert.load_copert(file)
            nox = copert.get_tot_NOX_per_interval()
            co2 = copert.get_tot_CO2_per_interval()

            row["NOX"] = nox
            row["CO2"] = co2
            row["NOX_RATIO"] = nox / original_nox
            row["CO2_RATIO"] = co2 / original_co2
            row["NOX_RATE"] = (nox - original_nox) / original_nox
            row["CO2_RATE"] = (co2 - original_co2) / original_co2

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

       
        row = {
            "TYPE": "original",
            "LAW": "original",
            "TOTAL_RATIO": 1,
             "TARGET_CLUSTER": -1,
             "TARGET_RATIO": -1,
        }

        for col in df.columns[11:]:
            row[col] = 1


        row["NOX"] = original_nox
        row["CO2"] = original_co2
        row["NOX_RATIO"] = 1
        row["CO2_RATIO"] = 1
        row["NOX_RATE"] = 0
        row["CO2_RATE"] = 0

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)


        

        return df





    # getters

    def get_original_outputs_path(self):
        # check
        if not self._original_outputs_path or str(self._original_outputs_path).strip() == "": 
            logger.error("Invalid or null _original_outputs_path.")
            raise ValueError("Invalid or null _original_outputs_path.")
        return self._original_outputs_path 
        

    def get_variation_outputs_path(self):
        # check
        if not self._variation_outputs_path or str(self._variation_outputs_path).strip() == "": 
            logger.error("Invalid or null _variation_outputs_path.")
            raise ValueError("Invalid or null _variation_outputs_path.")
        return self._variation_outputs_path 

    
    def get_copert_data_path(self):
        # check
        if not self._copert_data_path or str(self._copert_data_path).strip() == "": 
            logger.error("Invalid or null _copert_data_path.")
            raise ValueError("Invalid or null _copert_data_path.")
        return self._copert_data_path

    
    def get_network_manager(self):
        # check
        if not self._network_manager:
            logger.error("Null _network_manager.")
            raise ValueError("Null _network_manager.")
        return self._network_manager

    def get_variations_copert(self):
        # check
        if self._variations_copert.empty:
            logger.error("_variations_copert is empty.")
            raise ValueError("_variations_copert is empty.")
        return self._variations_copert
        
