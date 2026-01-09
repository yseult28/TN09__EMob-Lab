# AJOUTER DES MESURES POUR QUANTIFIER LE NOMBRE DE LIGNE TOTAL SUPPRIMÉ, ESTIMÉ CE QUI A ÉTÉ PERDU, TROUVER UNE SOLUTION POUR LES SECTIONS OU VALEURS NON VALIDES

# dépendances

import os 
from pathlib import Path

from datetime import time

import pandas as pd
import geopandas as gpd

import matplotlib.cm as cm
import matplotlib.colors as colors
from branca.colormap import LinearColormap

import folium
from folium.plugins import TimestampedGeoJson
from branca.element import Template, MacroElement
from folium.plugins import HeatMapWithTime

import ast

import re

import logging

from .NetworkManager import NetworkManager



# log
logger = logging.getLogger(__name__)



class Copert:

    # Constructor 
    
    def __init__(self,path:str="", network_manager:"NetworkManager"=None, copert_path:str=""):
        """
        Copert's constructor.
        loads the outputs.
    
        Parameters
        ----------
        path : string
            Path to the outputs directory.
            It must have the following columns : TYPE, SPEED, TRAVEL_TIME, COUNT
        network_manager : NetworkManager object
            A NetworkManager object used to link nodes with sections, sections with their length and coordinates.
            It must have been created from the same network's files as those used for the outputs' production.
        copert_path : string
            The path to the copert data file.
        """

        # check 
        if not path or str(path).strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not network_manager: 
            logger.error("Null NetworkManager.")
            raise ValueError("Null NetworkManager.")
        if not copert_path or copert_path.strip == "": 
            logger.error("Invalid or null copert_path.")
            raise ValueError("Invalid or null copert_path.")

        # assignment
        self._path = path # path to the outputs' directory
        self._network_manager = network_manager # NetworkManager object
        self._copert_path = copert_path # path to the copert file

        # outputs' files
        self._flow_data = []
        self._path_data = []
        self._timetravellink_data = []
        self._user_data = []
        self._veh_data = []

        # loads the outputs' files
        self.load_outputs_files()     

        # initializes the NetworkManager 
        self.init_network_manager()

        # copert file
        self._copert_data = []

        # loads the copert files
        self.load_copert_data()

        self._copert = [] # DataFrame with the following variables : TIME, TRAVEL_TIME (in hour), SECTION (section's name), TYPE (vehicle's type),          # COUNT (number of vehicles on the section at the given time), LENGTH (section's length),
        # LENGTH (section's length), SPEED (mean speed on the section at the given time), 
        # GEOMETRY (segment in coordinates of the section), NOX (NOX emissions), CO2 (CO2 emission)
        
        logger.info("Copert initialized with path : %s.", path)


    
    # configuration's methods 
    
    def set_new_outputs(self,path:str="", network_manager:"NetworkManager"=None):
        """
        Set up new parameters.
        Explicit version of __init__. To be used on an already initialized Copert object
    
        Parameters
        ----------
        path : string
            The path to the outputs directory.
            It must have the following columns : TYPE, SPEED, TRAVEL_TIME, COUNT
        network_manager : NetworkManager object
            A NetworkManager object used to link nodes with sections, sections with their length and coordinates.
            It must have been created from the same network's files as those used for the outputs' production.
        copert_path : string
            The path to the copert data file.
        """

        # check 
        if not path or str(path).strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not network_manager: 
            logger.error("Null NetworkManager.")
            raise ValueError("Null NetworkManager.")
        if not copert_path or copert_path.strip == "": 
            logger.error("Invalid or null copert_path.")
            raise ValueError("Invalid or null copert_path.")
            
        # assignment
        self._path = path # path to the outputs' directory
        self._network_manager = network_manager # NetworkManager object
        self._copert_path = copert_path # path to the copert file

        # outputs' files
        self._flow_data = []
        self._path_data = []
        self._timetravellink_data = []
        self._user_data = []
        self._veh_data = []

        # loads the outputs' files
        self.load_outputs_files()      

        # initializes the NetworkManager 
        self.init_network_manager()

        # copert data file
        self._copert_data = []

        # loads the copert files
        self.load_copert_data()

        self._copert = [] # DataFrame with the following variables : TIME, TRAVEL_TIME (in hour), SECTION (section's name), TYPE (vehicle's type),          # COUNT (number of vehicles on the section at the given time), LENGTH (section's length),
        # LENGTH (section's length), SPEED (mean speed on the section at the given time), 
        # GEOMETRY (segment in coordinates of the section), NOX (NOX emissions), CO2 (CO2 emission)
        
        logger.info("Copert initialized with path : %s.", path)


    def load_outputs_files(self):
        """
        Loads each output file in a variable.
        Can only be called by __init__() or set_new_outputs()

        No parameters
        """

        # check
        if not self._path or str(self._path).strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        
        # looks only for visible files
        paths = [p for p in Path(self._path).glob("*.csv") if not p.name.startswith(".")]
        paths.sort()

        # info
        logger.info(f"files found : {paths}")

        # loading
        self._flow_data=pd.read_csv(paths[0], sep=';')
        self._path_data=pd.read_csv(paths[1], sep=';')
        self._travel_time_link_data=pd.read_csv(paths[2], sep=';')
        self._user_data=pd.read_csv(paths[3], sep=';')
        self._veh_data=pd.read_csv(paths[4], sep=';', dtype={"TIME":str, "ID":str, "TYPE":str, "LINK":str, "POSITION":str, "SPEED":str, "STATE":str, "DISTANCE":str, "PASSENGERS":str, "TRAVELED_NODES":str}, on_bad_lines='skip')
        
        # format
        self._flow_data["TIME"] = self._flow_data["TIME"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._flow_data["TIME"] = pd.to_datetime(self._flow_data["TIME"], format='%H:%M:%S',errors="coerce")
        self._flow_data["TIME"] = self._flow_data["TIME"].dt.round("min")

        self._path_data["TIME"] = self._path_data["TIME"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._path_data["TIME"] = pd.to_datetime(self._path_data["TIME"], format='%H:%M:%S', errors="coerce")
        self._path_data["TIME"] = self._path_data["TIME"].dt.round("min")

        self._travel_time_link_data["TIME"] = self._travel_time_link_data["TIME"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._travel_time_link_data["TIME"] = pd.to_datetime(self._travel_time_link_data["TIME"], format='%H:%M:%S', errors="coerce")
        self._travel_time_link_data["TIME"] = self._travel_time_link_data["TIME"].dt.round("min")

        self._user_data["TIME"] = self._user_data["TIME"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._user_data["TIME"] = pd.to_datetime(self._user_data["TIME"], format='%H:%M:%S', errors="coerce")
        self._user_data["TIME"] = self._user_data["TIME"].dt.round("min")

        self._veh_data["TIME"] = self._veh_data["TIME"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._veh_data["TIME"] = pd.to_datetime(self._veh_data["TIME"], format='%H:%M:%S', errors="coerce")
        self._veh_data["TIME"] = self._veh_data["TIME"].dt.round("min")

        logger.info("files loaded.")

    
    def save_copert(self, path="", name=""):
        """
        Saves the copert dataframe with name at path.

        Parameters
        ----------
        path : string
            relative path to the copert file.
        name : string
            name fo the copert file
        """

        # check
        if not path or str(path).strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not name or name.strip() == "": 
            logger.error("Invalid or null name.")
            raise ValueError("Invalid or null name.")
        if self._copert.empty: 
            logger.error("_copert is empty")
            raise ValueError("_copert is empty")

        full_path = os.path.join(path,f"{name}.csv")
        
        self._copert.to_csv(full_path, sep=';', index=False)

        logger.info(f"{full_path} saved.")

    def load_copert(self, path=""):
        """
        Loads the copert dataframe from path.

        Parameters
        ----------
        path : string
            relative path to the copert file. It must a variation from the same original demand as the one from this Copert object.
        """

        # check
        if not path or str(path).strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        expected_columns = {
        "TIME", "TRAVEL_TIME", "SECTION", "TYPE", "COUNT",
        "LENGTH", "SPEED", "GEOMETRY", "NOX", "CO2"
        }

        buffer = pd.read_csv(path, sep=';')

        # check
        expected_columns = {
        "TIME", "TRAVEL_TIME", "SECTION", "TYPE", "COUNT",
        "LENGTH", "SPEED", "GEOMETRY", "NOX", "CO2"
        }

        if expected_columns != set(buffer.columns):
            logger.error("Invalid dataframe.")
            raise ValueError("Invalid dataframe.")

        self._copert = buffer.copy()

        logger.info(f"copert loaded from {path}.")
        

    
    def init_network_manager(self):
        """
        Initializes the NetworkManager object.
        Can only be called by __init__ or set_new_outputs().

        No parameters
        """

        # check
        if not self._network_manager: 
            logger.error("Null NetworkManager.")
            raise ValueError("Null NetworkManager.")
        
        self._network_manager.load_network_file()
        self._network_manager.create_network_dfs()

        logger.info("_network_manager initialized.")

    
    def load_copert_data(self):
        """
        Loads the copert data file.
        Can only be called by __init__ or set_new_outputs().

        No parameters
        """
        if not self._copert_path or self._copert_path.strip == "": 
            logger.error("Invalid or null copert_path.")
            raise ValueError("Invalid or null copert_path.")
            
        buffer = pd.read_csv(self._copert_path, sep=',')

        # check


        self._copert_data = buffer.copy()
        #logger.info(self._copert_data)

        logger.info(f"Copert loaded from {self._copert_path}.")

    
    
    # creation's methods 

    def create_copert(self, interval="6min"):
    
        """
        Creates the copert dataframe from the given outputs directory using the veh.csv file.
        Aggregates per section, time and vehicle type.
    
        Parameters
        ----------
        interval : string, optional
            time interval used to aggregate lines before copert calculation.
            It has to follow the following pattern : "<int>min" or "<int>h"
        """

        # loads network_manager's dataframes
        links = self._network_manager.get_links()
        sections = self._network_manager.get_sections()

        # check
        if self._veh_data.empty: 
            logger.error("_veh_data is empty.")
            raise ValueError("_veh_data is empty.")
        if links.empty: 
            logger.error("_links is empty.")
            raise ValueError("_links is empty.")
        if sections.empty: 
            logger.error("_sections is empty.")
            raise ValueError("_sections is empty.")
        if not bool(re.match(r"^\d+(min|h)$", interval)):
            logger.error("Invalid time interval pattern : the patter must be <int>min or <int>h.")
            raise ValueError("Invalid time interval pattern.")

        # buffer
        buffer = self._veh_data.copy()
        
        na_tot = 0
            
        # links and sections to dict for speed purpose
        adj = links.set_index("NODES")["SECTION"].to_dict() # "node1 node2" : "section"
        length = sections.set_index("SECTION")["LENGTH"].to_dict() # "section" : section's length
            
        # for each couple of nodes, finds the section
        buffer["SECTION"] = buffer["LINK"].map(adj)

        # info
        log_na = buffer[buffer["SECTION"].isna()]
        logger.info(f"Removed {len(log_na)} rows with sections not present in the network description file, {len(self._veh_data)} total lines.")
        na_tot += len(log_na)

        a = len(log_na.loc[log_na["TYPE"]=="Car"])
        tot_car = len(self._veh_data.loc[self._veh_data["TYPE"]=="Car"])
        b = len(log_na.loc[log_na["TYPE"]=="Bus"])
        tot_bus = len(self._veh_data.loc[self._veh_data["TYPE"]=="Bus"])
        c = len(log_na.loc[log_na["TYPE"]=="Metro"])
        tot_metro = len(self._veh_data.loc[self._veh_data["TYPE"]=="Metro"])
        d = len(log_na.loc[log_na["TYPE"]=="Tram"])
        tot_tram = len(self._veh_data.loc[self._veh_data["TYPE"]=="Tram"])
        logger.info(f"{a}/{tot_car} cars, {b}/{tot_bus} bus, {c}/{tot_metro} metro, {d}/{tot_tram} tram.")
            
        # drops rows with nan value in SECTION
        buffer = buffer.dropna(subset=["SECTION"])

        # for each section, its length
        buffer["LENGTH"] = buffer["SECTION"].map(length)

        # drops unused columns
        buffer = buffer.drop(columns=["LINK","STATE","DISTANCE","TRAVELED_NODES"])

        # info
        invalid_length = pd.to_numeric(buffer["LENGTH"], errors="coerce").isna().sum()
        invalid_speed  = pd.to_numeric(buffer["SPEED"], errors="coerce").isna().sum()
        
        #invalid_rows = buffer[pd.to_numeric(buffer["SPEED"], errors="coerce").isna()]
        #lower_limit = time(16, 10)   
        #upper_limit = time(17, 50 )  
        #filtered = invalid_rows[(invalid_rows["TIME"].dt.time < lower_limit) | (invalid_rows["TIME"].dt.time > upper_limit)]
        #logger.info(filtered)
        
        logger.info(f"Invalid LENGTH values: {invalid_length}, invalid SPEED values: {invalid_speed}, {len(self._veh_data)} total lines")
        na_tot += invalid_length + invalid_speed

        # Keeps only valid values in LENGTH and SPEED
        buffer["LENGTH"] = pd.to_numeric(buffer["LENGTH"], errors="coerce")
        buffer["SPEED"]  = pd.to_numeric(buffer["SPEED"],  errors="coerce")

        # agregation by time interval
        buffer["TIME"] = buffer["TIME"].dt.round(interval)

        # conversion in hour
        if interval[-1] == 'h': interval = int(interval[:-1])
        else : interval = int(interval[:-3]) / 60 #xmin_to_hours(interval

        # conversion in km/h
        buffer["SPEED"] = buffer["SPEED"]*3.6

        # conversion in km
        buffer["LENGTH"] = buffer["LENGTH"]/1000

         # info
        na = buffer.isna().sum()
        #n_to_drop = buffer.isna().any(axis=1).sum()
        logger.info(f"Removed {na_tot} lines with NaN values, lines with NaN values outside of the TIME, SECTION, LENGTH and SPEED variables are not removed, {len(self._veh_data)} total lines : \n{na}")
        #na_tot += n_to_drop

        # drops lines with NaN values
        buffer= buffer.dropna(subset=["LENGTH","SPEED"])
        
        # for each time, each section and each type of vehicle, the number of vehicle during the time periode, the mean length and speed
        buffer = buffer.groupby(["TIME","SECTION","TYPE"]).agg({"ID":"count","LENGTH":"mean","SPEED":"mean"}).reset_index()
        buffer = buffer.rename(columns={"ID":"COUNT"})

        # time interval between each agregations
        buffer["TRAVEL_TIME"] = interval
            
        # sections to dict for speed purpose
        sections = sections.set_index("SECTION")["GEOMETRY"].to_dict() # "section" : coordinates

        # for each sections, finds the position
        buffer["GEOMETRY"] = buffer["SECTION"].map(sections)
            
        self._copert=gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=self._network_manager.get_crs()) 
            
        # copert
        self._copert["NOX"] = self._copert.apply(self.get_NOX_per_row, axis=1)
        self._copert["CO2"] = self._copert.apply(self.get_CO2_per_row, axis=1)

        logger.info(f"Removed {na_tot} lines on {len(self._veh_data)} total lines.")
        logger.info("_copert created.")



    # Copert's calculation methods

    def get_NOX_per_row(self, row):
        """
        return the NOX emission values for the given row. 
        Can only be called by a creation's method in the Copert class.
    
        Parameters
        ----------
        row : Series
            row used for calculation.
            It must have the following columns : TYPE, SPEED, TRAVEL_TIME, COUNT
        """

        # conversion of SPEED values from float to string
        speed = str(round(row["SPEED"]))

        # vehicle type
        veh_type = row["TYPE"]

        # copert's calculation
        
        if veh_type == "Bus": 
            mask = (self._copert_data["type"]=="Buses") & (self._copert_data["pollutant"]=="NOX") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            return self._copert_data.loc[mask , speed ].values[0] * row["TRAVEL_TIME"] * row["COUNT"]

        if veh_type == "Car": 
            mask = (self._copert_data["type"]=="Passenger Cars") & (self._copert_data["pollutant"]=="NOX") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            if self._copert_data.loc[mask, speed ].size == 0 : print(row)
            return self._copert_data.loc[mask, speed ].values[0] * row["TRAVEL_TIME"] * row["COUNT"]

        if veh_type == "Metro": 
            return 0

        if veh_type == "Tram": 
            return 0


    def get_CO2_per_row(self, row):
        """
        return the CO2 emission values for the given row. 
        Can only be called by a creation's method in the Copert class.
    
        Parameters
        ----------
        row : Series
            row used for calculation.
            It must have the following columns : TYPE, SPEED, TRAVEL_TIME, COUNT
        """

        # conversion of SPEED values from float to string
        speed = str(round(row["SPEED"]))

        # vehicle type
        veh_type = row["TYPE"]

        # copert's calculation

        if veh_type == "Bus": 
            mask = (self._copert_data["type"]=="Buses") & (self._copert_data["pollutant"]=="CO2") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            return self._copert_data.loc[mask , speed ].values[0] * row["TRAVEL_TIME"] * row["COUNT"]

        if veh_type == "Car": 
            mask = (self._copert_data["type"]=="Passenger Cars") & (self._copert_data["pollutant"]=="CO2") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            if self._copert_data.loc[mask, speed ].size == 0 : print(row)
            return self._copert_data.loc[mask, speed ].values[0] * row["TRAVEL_TIME"] * row["COUNT"]

        if veh_type == "Metro": 
            return 0

        if veh_type == "Tram": 
            return 0


    def get_tot_NOX_per_interval(self, period=[]): # period holds two dt.time values
        if not period:
            # complete time period
            return self._copert["NOX"].sum()
        else: 
            buffer = self._copert.loc[(self._copert["TIME"].dt.time >= period[0]) & (self._copert["TIME"].dt.time <= period[1])]
            if buffer.empty : logger.info("time period not included in the data")
            return buffer["NOX"].sum()

            
    def get_tot_CO2_per_interval(self, period=[]): # period holds two dt.time values
        if not period:
            # complete time period
            return self._copert["CO2"].sum()
        else: 
            buffer = self._copert.loc[(self._copert["TIME"].dt.time >= period[0]) & (self._copert["TIME"].dt.time <= period[1])]
            if buffer.empty : logger.info("time period not included in the data")
            return buffer["CO2"].sum()



    # getters 

    def get_path(self):
        return self._path


    def get_network_manager(self):
        return self._network_manager


    def get_copert_path(self):
        return self._copert_path

        
    def get_flow_data(self):
        if self._flow_data.empty: logger.info("_flow_data is empty.")
        return self._flow_data


    def get_path_data(self):
        if self._path_data.empty: logger.info("_path_data is empty.")
        return self._path_data


    def get_time_travel_link_data(self):
        if self._time_travel_link_data.empty: logger.info("_time_travel_link_data is empty.")
        return self._time_travel_link_data


    def get_user_data(self):
        if self._user_data.empty: logger.info("_user_data is empty.")
        return self._user_data


    def get_veh_data(self):
        if self._veh_data.empty: logger.info("_veh_data is empty.")
        return self._veh_data


    def get_outputs(self):
        l = []
        l.append(self.get_flow_data())
        l.append(self.get_path_data())
        l.append(self.get_time_travel_link_data())
        l.append(self.get_user_data())
        l.append(self.get_veh_data())
        return l


    def get_copert_data(self):
        return self._copert_data


    def get_copert(self):
        if self._copert.empty: logger.info("_copert_by_sections is empty.")
        return self._copert.copy()


    # display methods
    
    def display_NOX(self, interval="10min",max_opacity=1 ,radius=15, blur=0.8):

        buffer = self._copert.copy()
        buffer = buffer.to_crs("epsg:4326")
        buffer["TIME"] = pd.to_datetime(buffer["TIME"]).dt.round(interval)
        
        def get_midpoint(linestring):
            if linestring.geom_type == "LineString":
                return linestring.interpolate(0.5, normalized=True)
            return None

        buffer["MIDPOINT"] = buffer["GEOMETRY"].apply(get_midpoint)
        buffer = buffer.groupby(["TIME", "MIDPOINT"])["NOX"].sum().reset_index()
        buffer = gpd.GeoDataFrame(buffer, geometry="MIDPOINT", crs="epsg:4326")


        maxi = buffer["NOX"].max()

        hm_time_data = buffer.apply(lambda row : row["TIME"].strftime("%H:%M"), axis=1)
        buffer["TIME"] = hm_time_data
        hm_time_data = list(set(hm_time_data))
        hm_time_data.sort()

        hm_point_data = []
        t_data = []

        t_data = buffer.apply(lambda row : {"lat":row["MIDPOINT"].y, "lon":row["MIDPOINT"].x, "val":row["NOX"]}, axis=1)

        x = 0


        for time in hm_time_data:
            points = []
            if(x > buffer.shape[0]-1) : break
            #print(time)
            while(buffer.loc[x,"TIME"]==time):
                #print(buffer.loc[x,"DEPARTURE"])
                val = float(buffer.loc[x,"NOX"]/maxi)
                points.append([buffer.loc[x,"MIDPOINT"].y,buffer.loc[x,"MIDPOINT"].x,val])
                x+=1
                if(x > buffer.shape[0]-1) : break
            #print(points)
            hm_point_data.append(points)


        # Création de la carte
        mean_lat = buffer.MIDPOINT.y.mean()
        mean_lon = buffer.MIDPOINT.x.mean()
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


        HeatMapWithTime(
            data=hm_point_data,
            index=hm_time_data,
            gradient=gradient,
            max_opacity=max_opacity,
            radius=radius,
            blur=blur
        ).add_to(m)

        return m


    def display_CO2(self, interval="10min", max_opacity=1, radius=15, blur=0.8):

        buffer = self._copert.copy()
        buffer = buffer.to_crs("epsg:4326")
        buffer["TIME"] = pd.to_datetime(buffer["TIME"]).dt.round(interval)
        
        def get_midpoint(linestring):
            if linestring.geom_type == "LineString":
                return linestring.interpolate(0.5, normalized=True)
            return None

        buffer["MIDPOINT"] = buffer["GEOMETRY"].apply(get_midpoint)
        buffer = buffer.groupby(["TIME", "MIDPOINT"])["CO2"].sum().reset_index()
        buffer = gpd.GeoDataFrame(buffer, geometry="MIDPOINT", crs="epsg:4326")


        maxi = buffer["CO2"].max()

        hm_time_data = buffer.apply(lambda row : row["TIME"].strftime("%H:%M"), axis=1)
        buffer["TIME"] = hm_time_data
        hm_time_data = list(set(hm_time_data))
        hm_time_data.sort()

        hm_point_data = []
        t_data = []

        t_data = buffer.apply(lambda row : {"lat":row["MIDPOINT"].y, "lon":row["MIDPOINT"].x, "val":row["CO2"]}, axis=1)

        x = 0


        for time in hm_time_data:
            points = []
            if(x > buffer.shape[0]-1) : break
            #print(time)
            while(buffer.loc[x,"TIME"]==time):
                #print(buffer.loc[x,"DEPARTURE"])
                val = float(buffer.loc[x,"CO2"]/maxi)
                points.append([buffer.loc[x,"MIDPOINT"].y,buffer.loc[x,"MIDPOINT"].x,val])
                x+=1
                if(x > buffer.shape[0]-1) : break
            #print(points)
            hm_point_data.append(points)


        # Création de la carte
        mean_lat = buffer.MIDPOINT.y.mean()
        mean_lon = buffer.MIDPOINT.x.mean()
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






        HeatMapWithTime(
            data=hm_point_data,
            index=hm_time_data,
            gradient=gradient,
            max_opacity=max_opacity,
            radius=radius,
            blur=blur
        ).add_to(m)

        return m

        
        
    

    
                

        