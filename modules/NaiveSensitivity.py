 
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




class NaiveSensitivity:

    # Constructor
    
    def __init__(self,original_outputs_path="",variation_outputs_path="",copert_data_path="",network_manager=None):
        """
        Sensitivity's constructor.
    
        Parameters
        ----------
        original_outputs_path : string
            Path to the original outputs' folder.
        variation_outputs_path : string
            Path to the variations outputs' folder.
        copert_data_path : string
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

 

        # loads 
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
    
    def create_total_sensitivity_df_test(self):
        """
        Creates a dataframe for study purpose.
        
    
        No parameters
        """
        # initializes
        df = pd.DataFrame(columns=["TYPE","PARAMETERS","RATIO","START","END","EVENT","NOX","CO2","NOX_RATIO","CO2_RATIO","NOX_RATE","CO2_RATE"])

        # adds the first line
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()
        first_row = {
            "TYPE": "ORIGINAL",
            "PARAMETERS": "None",
            "RATIO": 1,
            "START": "None",
            "END": "None",
            "EVENT": "None",
            "NOX": original_nox,
            "CO2": original_co2,
            "NOX_RATIO": 1,
            "CO2_RATIO": 1,
            "NOX_RATE": 0,
            "CO2_RATE": 0
        }
        df = pd.concat([df, pd.DataFrame([first_row])], ignore_index=True)

        # regex to parse the name of the folders
        pattern = re.compile(
            r'^(?P<TYPE>[^_]+)_'
            r'(?P<PARAMETERS>[^_]+)_'
            r'(?P<RATIO>[^_]+)_'
            r'(?P<PERIODE>[^_]+)_'
            r'(?P<EVENTS>[^_]+)$'
        )

        def parse_section(section, is_period=False):
            if section == "None":
                return "None"
            if is_period:
                parts = section.split('-')
                if len(parts) == 2:
                    return parts  # [start, end]
                else:
                    return [section, section]
            else:
                return section.split('-')

        # retrieves the name of each folder
        folders = Path(self._variation_outputs_path).iterdir()
        folders = [f for f in folders if f.is_dir() and not f.name.startswith(".")]

        # creates a line for each folder
        for folder in folders:
            name = folder.name
            match = pattern.match(name)
            if match:
                d = match.groupdict()
                periode = parse_section(d["PERIODE"], is_period=True)
                row = {
                    "TYPE": parse_section(d["TYPE"]),
                    "PARAMETERS": parse_section(d["PARAMETERS"]),
                    "RATIO": parse_section(d["RATIO"]),
                    "START": periode[0] if periode else None,
                    "END": periode[1] if periode else None,
                    "EVENT": parse_section(d["EVENTS"]),
                    "NOX": 0,
                    "CO2": 0,
                    "NOX_RATIO": 0,
                    "CO2_RATIO": 0,
                    "NOX_RATE": 0,
                    "CO2_RATE": 0
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        x = 1
        for copert in self._variation_copert:
            nox = copert.get_tot_NOX_per_interval()
            nox_r = nox / original_nox
            nox_ra = (nox - original_nox) / original_nox
            co2 = copert.get_tot_CO2_per_interval()
            co2_r = co2 / original_co2
            co2_ra = (co2 - original_co2) / original_co2
            
            df.loc[x,"NOX"] = nox
            df.loc[x,"NOX_RATIO"] = nox_r
            df.loc[x,"NOX_RATE"] = nox_ra
            df.loc[x,"CO2"] = co2
            df.loc[x,"CO2_RATIO"] = co2_r
            df.loc[x,"CO2_RATE"] = co2_ra

            x += 1

        return df

    

    def create_total_sensitivity_df(self):
        """
        Creates a dataframe for study purpose.

        No parameters
        """

        # initializes
        df = pd.DataFrame(columns=[
            "TYPE", "PARAMETERS", "RATIO", "START", "END",
            "EVENTS", "NOX", "CO2", "NOX_RATIO", "CO2_RATIO", "NOX_RATE", "CO2_RATE"
        ])

        # adds the first line
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()
        first_row = {
            "TYPE": "ORIGINAL",
            "PARAMETERS": "None",
            "RATIO": 1,
            "START": "None",
            "END": "None",
            "EVENTS": ["None"],
            "NOX": original_nox,
            "CO2": original_co2,
            "NOX_RATIO": 1,
            "CO2_RATIO": 1,
            "NOX_RATE": 0,
            "CO2_RATE": 0
        }
        df = pd.concat([df, pd.DataFrame([first_row])], ignore_index=True)

        # regex to parse the name of the folders
        pattern_full = re.compile(
            r'^(?P<TYPE>[^_]+)_'
            r'(?P<PARAMETERS>[^_]+)_'
            r'(?P<RATIO>[^_]+)_'
            r'(?P<PERIODE>[^_]+)__'
            r'(?P<EVENTS>.+)$'
        )

        # helper to parse events string
        def parse_events(events_str):
            """Parses the events format string into a structured list."""
            if events_str == "None":
                return ["None"]
            events_parts = events_str.split("_")
            result = []
            for e in events_parts:
                if "-" in e:
                    result.append(e.split("-"))
                else:
                    result.append([e])
            return result

        # helper to parse period
        def parse_period(section):
            if section == "None":
                return ["None", "None"]
            parts = section.split('-')
            if len(parts) == 2:
                return parts
            else:
                return [section, section]

        # retrieves the name of each folder
        folders = Path(self._variation_outputs_path).iterdir()
        folders = [f for f in folders if f.is_dir() and not f.name.startswith(".")]

        # creates a line for each folder
        for folder in folders:
            name = folder.name

            # Handle case "None_(events format)"
            if name.startswith("None__"):
                # Exemple: None_A-B_C-D_E-F_G-H
                events_str = name.split("None__")[1]
                row = {
                    "TYPE": "None",
                    "PARAMETERS": "None",
                    "RATIO": "None",
                    "START": "None",
                    "END": "None",
                    "EVENTS": parse_events(events_str),
                    "NOX": 0,
                    "CO2": 0,
                    "NOX_RATIO": 0,
                    "CO2_RATIO": 0,
                    "NOX_RATE": 0,
                    "CO2_RATE": 0
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                continue

            # Handle standard case
            match = pattern_full.match(name)
            if match:
                d = match.groupdict()
                periode = parse_period(d["PERIODE"])
                events = parse_events(d["EVENTS"])

                row = {
                    "TYPE": d["TYPE"],
                    "PARAMETERS": d["PARAMETERS"],
                    "RATIO": d["RATIO"],
                    "START": periode[0],
                    "END": periode[1],
                    "EVENTS": events,
                    "NOX": 0,
                    "CO2": 0,
                    "NOX_RATIO": 0,
                    "CO2_RATIO": 0,
                    "NOX_RATE": 0,
                    "CO2_RATE": 0
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Fill NOX/CO2 ratios and rates
        x = 1
        for copert in self._variation_copert:
            nox = copert.get_tot_NOX_per_interval()
            co2 = copert.get_tot_CO2_per_interval()
            nox_r = nox / original_nox
            co2_r = co2 / original_co2
            nox_ra = (nox - original_nox) / original_nox
            co2_ra = (co2 - original_co2) / original_co2

            df.loc[x, "NOX"] = nox
            df.loc[x, "NOX_RATIO"] = nox_r
            df.loc[x, "NOX_RATE"] = nox_ra
            df.loc[x, "CO2"] = co2
            df.loc[x, "CO2_RATIO"] = co2_r
            df.loc[x, "CO2_RATE"] = co2_ra

            x += 1

        return df

    def create_total_sensitivity_df_from_path(self, path):
        """
        Creates a dataframe for study purpose from the copert dataframes saved at path

        parameters
        path : string
            The path where copert dfs are saved.The dfs must be copert dataframes from variations of the original demand of this NaiveSensitivity object.
        """

        # loads the copert dfs
        self.load_copert_dfs(path)

        # initializes
        #df = pd.DataFrame(columns=[
            #"TYPE", "PARAMETERS", "RATIO", "START", "END",
            #"EVENTS", "NOX", "CO2", "NOX_RATIO", "CO2_RATIO", "NOX_RATE", "CO2_RATE"
       # ])

        # adds the first line
        original_nox = self._original_copert.get_tot_NOX_per_interval()
        original_co2 = self._original_copert.get_tot_CO2_per_interval()
        first_row = {
            "TYPE": "ORIGINAL",
            "PARAMETERS": "None",
            "RATIO": 1.0,
            "START": "None",
            "END": "None",
            "EVENTS": ["None"],
            "NOX": original_nox,
            "CO2": original_co2,
            "NOX_RATIO": 1.0,
            "CO2_RATIO": 1.0,
            "NOX_RATE": 0.0,
            "CO2_RATE": 0.0
        }
        df = pd.DataFrame([first_row])
        #df = pd.concat([df, pd.DataFrame([first_row])], ignore_index=True)

        # regex to parse the name of the folders
        pattern_full = re.compile(
            r'^(?P<TYPE>[^_]+)_'
            r'(?P<PARAMETERS>[^_]+)_'
            r'(?P<RATIO>[^_]+)_'
            r'(?P<PERIODE>[^_]+)__'
            r'(?P<EVENTS>.+)$'
        )

        # helper to parse events string
        def parse_events(events_str):
            """Parses the events format string into a structured list."""
            if events_str == "None":
                return ["None"]
            events_parts = events_str.split("_")
            result = []
            for e in events_parts:
                if "-" in e:
                    result.append(e.split("-"))
                else:
                    result.append([e])
            return result

        # helper to parse period
        def parse_period(section):
            if section == "None":
                return ["None", "None"]
            parts = section.split('-')
            if len(parts) == 2:
                return parts
            else:
                return [section, section]

        # retrieves the name of each file
        files = list(Path(path).glob("*.csv"))
        files.sort()

        # creates a line for each folder
        for file in files:
            name = Path(file).stem

            # Handle case "None_(events format)"
            if name.startswith("None__"):
                # Exemple: None_A-B_C-D_E-F_G-H
                events_str = name.split("None__")[1]
                row = {
                    "TYPE": "None",
                    "PARAMETERS": "None",
                    "RATIO": "None",
                    "START": "None",
                    "END": "None",
                    "EVENTS": parse_events(events_str),
                    "NOX": 0,
                    "CO2": 0,
                    "NOX_RATIO": 0,
                    "CO2_RATIO": 0,
                    "NOX_RATE": 0,
                    "CO2_RATE": 0
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                continue

            # Handle standard case
            match = pattern_full.match(name)
            if match:
                d = match.groupdict()
                periode = parse_period(d["PERIODE"])
                events = parse_events(d["EVENTS"])

                row = {
                    "TYPE": d["TYPE"],
                    "PARAMETERS": d["PARAMETERS"],
                    "RATIO": d["RATIO"],
                    "START": periode[0],
                    "END": periode[1],
                    "EVENTS": events,
                    "NOX": 0,
                    "CO2": 0,
                    "NOX_RATIO": 0,
                    "CO2_RATIO": 0,
                    "NOX_RATE": 0,
                    "CO2_RATE": 0
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Fill NOX/CO2 ratios and rates
        copert = copy.deepcopy(self._original_copert)
        x = 1
        for file in files:
            copert.load_copert(file)
            nox = copert.get_tot_NOX_per_interval()
            co2 = copert.get_tot_CO2_per_interval()
            nox_r = nox / original_nox
            co2_r = co2 / original_co2
            nox_ra = (nox - original_nox) / original_nox
            co2_ra = (co2 - original_co2) / original_co2

            df.loc[x, "NOX"] = nox
            df.loc[x, "NOX_RATIO"] = nox_r
            df.loc[x, "NOX_RATE"] = nox_ra
            df.loc[x, "CO2"] = co2
            df.loc[x, "CO2_RATIO"] = co2_r
            df.loc[x, "CO2_RATE"] = co2_ra

            x += 1

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

    def get_variation_copert(self):
        # check
        if len(self._variation_copert) == 0:
            logger.error("_variation_copert is empty.")
            raise ValueError("_variation_copert is empty.")
        return self._variation_copert
        
