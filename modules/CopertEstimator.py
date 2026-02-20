
# dependencies

import os

import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import wkt

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon

import logging




# log
logger = logging.getLogger(__name__)


# useful fonctions

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

def restore_time_variable(df, column="DEPARTURE"):
    """
    Convert a datetime column back to a string time format HH:MM:SS.
    Restore '24:' prefix if the original time was midnight after rounding.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time column.
    column : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        The transformed DataFrame with the column as string HH:MM:SS.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Round to nearest minute
    df[column] = df[column].dt.round("min")

    # Convert to string HH:MM:SS
    df[column] = df[column].dt.strftime("%H:%M:%S")

    def restore_24_format(t):
        # If time is exactly midnight "00:xx:yy" AND original was "24:xx:yy"
        # → We cannot detect original directly anymore, so we only restore pure midnight
        if t.startswith("00:") and t != "00:00:00":
            return "24:" + t.split(":", 1)[1]
        return t

    df[column] = df[column].apply(restore_24_format)

    return df

def process_position(df, crs="EPSG:4326"):
    """
    Return a geodataframe for a dataframe containing WKT points in the POSITION column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'POSITION' column as WKT strings.
    crs : str, optional
        Coordinate reference system for the output GeoDataFrames (default: "EPSG:4326").

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame.
    """

    df = df.copy()

    if not "POSITION" in df.columns :
        logger.info("Invalid dataframe.")
        raise ValueError("Invalid dataframe.")

    # Convert WKT ORIGIN or Point geometry to X and Y
    if isinstance(df["POSITION"].iloc[0], str):  # cas WKT en string
        df[["POSITION_X", "POSITION_Y"]] = df["POSITION"].apply(
            lambda p: p.replace("POINT(", "").replace(")", "")
        ).str.split(' ', expand=True).astype(float)
    else:  # cas Shapely Point
        df["POSITION_X"] = df["POSITION"].apply(lambda p: p.x)
        df["POSITION_Y"] = df["POSITION"].apply(lambda p: p.y)


    # Convert all coordinates to float
    df[["POSITION_X","POSITION_Y"]] = df[["POSITION_X","POSITION_Y"]].astype(float)

    # Recreate Point objects for origin and destination
    df["GEOMETRY"] = df.apply(lambda row: Point(row["POSITION_X"], row["POSITION_Y"]), axis=1)

    # Drop temporary coordinate columns
    df.drop(columns=["POSITION_X", "POSITION_Y"], inplace=True)

    # Create separate GeoDataFrames for origins and destinations
    gdf = gpd.GeoDataFrame(df, geometry="GEOMETRY", crs=crs)

    return gdf.copy()

def freq_to_seconds(freq: str):
    """
    Converts a pandas frequence (ex: '5min', '2h', '30s', '250ms', '1D')
    in secondes.
    """
    td = pd.to_timedelta(freq)
    return td.total_seconds()


def build_linestring(geoms):
    """
    Build a linestring from a string of points

    Parameters
    ----------
    geoms : list
        The list of points.
        
    Returns
    -------
    LineString(coords) : LineString
        LineString.
    """
    
    coords = [p for p in geoms]

    # Aucun point : LineString vide
    if len(coords) == 0:
        return LineString()

    # Un seul point : dupliquer pour faire un LineString valide
    if len(coords) == 1:
        return LineString([coords[0], coords[0]])

    # Plusieurs points : LineString normal
    return LineString(coords)


class CopertEstimator():
    """
    Estimates NOx and CO2 emissions for a given MnMS simulation.


    Attributes
    ---------
    _simulation_outputs_directory : str
        Simulation outputs directory for a given MnMS simulation.
    _veh_data : pandas.DataFrame
        veh.csv dataframe.
    _crs : str
        Projection system of the given simulation.
    _copert_data_path : str
        Path to the copert data file.
    _copert_data : pandas.DataFrame
        copert data dataframe
    """

    # Constructor

    def __init__(self, simulation_outputs_directory="", crs="epsg:4326", copert_data_path=""):
        """
        CopertEstimator constructor.

        Parameters
        --------
        _simulation_outputs_directory : str
            Simulation outputs directory for a given MnMS simulation.
        _method : set
            Speed calculation method names.
        _crs : str
            Projection system of the given simulation.
        _copert_data_path : str
            Path to the copert data file.

        Returns
        -------
        None
        """

        # check
        if not simulation_outputs_directory or simulation_outputs_directory.strip() == "" :
            logger.error("Invalid or null path to simulation outputs directory.")
            raise ValueError("Invalid or null path to simulation outputs directory.")

        # assignment
        self._simulation_outputs_directory = simulation_outputs_directory
        self._methods = {"SPEED_MEAN","DISTANCE_DIFF"}
        self._crs = crs
        self._copert_data_path = copert_data_path
        self._copert_data = pd.read_csv(self._copert_data_path, sep=',')

        # load veh.csv file
        veh_data_path = f"{self._simulation_outputs_directory}/veh.csv"

        # preprocessing
        self._veh_data = pd.read_csv(veh_data_path, sep=';',engine="python",on_bad_lines="warn")
        self._veh_data["SPEED"] = self._veh_data["SPEED"].fillna(0)
        self._veh_data = self._veh_data.dropna(subset=["TIME","TYPE","POSITION"])
        self._veh_data = process_time_variable(self._veh_data, "TIME")
        self._veh_data = process_position(self._veh_data, self._crs)
        

        self._estimation = pd.DataFrame(columns=["ID","TYPE","TIME","PERIOD","POSITION","GEOMETRY","SPEED","DISTANCE","NOX","CO2"])
        # PERIOD in hour, SPEED in km/h, DISTANCE in km, NOX in, CO2 in

        logger.info(f"CopertEstimator object built with _simulation_outputs_directory : {self._simulation_outputs_directory}.")


    # configuration methods

    def save_copert(self, name="", path = ""):
        """
        Saves _estimation at the given path.

        Parameters
        ----------
        name : str, optional
            File name (default = "").
        path : str, optional
            Path to save _estimation (default = "").

        Returns
        -------
        None
        """

        # check

        if not name or name.strip() == "" :
            logger.error("Invalid or null name.")
            raise ValueError("Invalid or null name.")

        if not path or path.strip() == "" :
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        if self._estimation.empty : 
            logger.error("_estimation is empty.")
            raise ValueError("_estimation is empty.")

        os.makedirs(path, exist_ok=True)
        estimation_path = f"{path}_{name}.csv"
        self._estimation.to_csv(estimation_path, sep=';', index=False)
        cvnjklm

        logger.info(f"_estimation saved at path : {path}.")

    def load_copert(self, path = ""):
        """
        Loads _estimation from the given path.

        Parameters
        --------
        path : str
            Path to the file (default = "").
        """

        # check

        if not path or path.strip() == "" :
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        buffer = pd.read_csv(path, sep=';')

        if buffer.empty:
            logger.error("Empty dataframe.")
            raise ValueError("Empty dataframe.")

        if not list(buffer.columns) == ["ID","TYPE","TIME","PERIOD","POSITION","GEOMETRY","SPEED","DISTANCE","NOX","CO2"]:
            logger.error("Invalid columns.")
            raise ValueError("Invalid columns.")

        self._estimation = buffer.copy()
            
        logger.info(f"_estimation loaded from path : {path}.")


    # copert

    def copert_estimation(self, period="auto", method="SPEED_MEAN") :
        """
        Estimates Nox and CO2 emissions with Copert method and temporal aggregation.

        Parameters
        --------
        period : str, optionnal
            Temporal aggregation value, if "auto" aggregate by travel. (default:"auto").
        method : str, optionnal
            Speed calculation method name, SPEED uses the mean of the SPEED variable, DISTANCE calculates the relative distance value divided 
            the period (default:"SPEED").

        Returns
        -------
        None
        """

        # check
        if not method in self._methods : 
            logger.error("Invalid method.")
            raise ValueError("Invalid method.")

        buffer = self._veh_data.copy()
        buffer = buffer[["TIME","ID","TYPE","POSITION","GEOMETRY","DISTANCE","SPEED"]]

        buffer = buffer.sort_values("TIME") 
        buffer["SPEED"] = pd.to_numeric(buffer["SPEED"], errors="coerce")
        buffer["DISTANCE"] = pd.to_numeric(buffer["DISTANCE"], errors="coerce")
        
        # temporal aggregation
        if period == "auto" :
            buffer = (
                buffer.groupby(["ID", "TYPE"])
            .agg(
                SPEED_MEAN=("SPEED", "mean"),

                DISTANCE=("DISTANCE", lambda x: x.iloc[-1] - x.iloc[0]),

                POSITION=("POSITION", lambda x: " ".join(x.astype(str))),

                GEOMETRY=("GEOMETRY", lambda g: build_linestring(g)),

                TIME=("TIME", lambda x: x.min()),

                PERIOD=("TIME", lambda x: x.max() - x.min())
                )
                .reset_index()
            )
            buffer["PERIOD"] = buffer["PERIOD"].dt.total_seconds() / 3600

        else : 
            buffer["TIME"] = buffer["TIME"].dt.round(period)
            buffer = (
                buffer.groupby(["TIME","ID","TYPE"])  
                  .agg(
                    SPEED_MEAN=("SPEED", "mean"),

                    DISTANCE=("DISTANCE", lambda x: x.iloc[-1] - x.iloc[0]),

                    POSITION=("POSITION", lambda x: " ".join(x.astype(str))),

                    GEOMETRY=("GEOMETRY", lambda g: build_linestring(g))
                  ).reset_index()
            )
            buffer["PERIOD"] = freq_to_seconds(period)/ 3600

        
        # DISTANCE in meters, SPEED_MEAN in meters per second, conversion to km and km/h
        buffer["DISTANCE"] = buffer["DISTANCE"]/1000
        buffer["SPEED_MEAN"] = buffer["SPEED_MEAN"] * 3.6

        # speed calculation with DISTANCE_DIFF method
        buffer["DISTANCE_DIFF"] = buffer["DISTANCE"]/buffer["PERIOD"]
        

        # choose the speed calculation corresponding to method
        buffer = buffer.rename(columns={method:"SPEED"})

        buffer["SPEED"] = buffer["SPEED"].fillna(0)

        buffer = buffer[["ID","TYPE","TIME","PERIOD","POSITION","GEOMETRY","SPEED","DISTANCE"]]
        
        # Copert calculation
        buffer["NOX"] = buffer.apply(self.get_NOX_per_row, axis=1)
        buffer["CO2"] = buffer.apply(self.get_CO2_per_row, axis=1)

        buffer = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=self._crs) 

        self._estimation = buffer.copy()

        logger.info(f"Copert estimation done with period : {period} and method : {method}.")
        


    # Copert's calculation methods

    def get_NOX_per_row(self, row):
        """
        return the NOX emission values for the given row. 
        Can only be called by an estimation method in the Copert class.
    
        Parameters
        ----------
        row : Series
            row used for calculation.
            It must have the following columns : TYPE, SPEED, TRAVEL_TIME, COUNT
            
        Returns
        -------
        NOX value : float
            NOX value for the given row.
        """

        if round(row["SPEED"]) ==  0 : 
            return 0

        # conversion of SPEED values from float to string
        speed = str(round(row["SPEED"]))

        # vehicle type
        veh_type = row["TYPE"]

        # copert's calculation
        
        if veh_type == "Bus": 
            mask = (self._copert_data["type"]=="Buses") & (self._copert_data["pollutant"]=="NOX") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            if self._copert_data.loc[mask, speed ].size == 0 : logger.info(f"{row}")
            return self._copert_data.loc[mask , speed ].values[0] * row["DISTANCE"]

        if veh_type == "Car": 
            mask = (self._copert_data["type"]=="Passenger Cars") & (self._copert_data["pollutant"]=="NOX") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            if self._copert_data.loc[mask, speed ].size == 0 : logger.info(f"{row}")
            return self._copert_data.loc[mask, speed ].values[0] * row["DISTANCE"]

        if veh_type == "Metro": 
            return 0

        if veh_type == "Tram": 
            return 0


    def get_CO2_per_row(self, row):
        """
        return the CO2 emission values for the given row. 
        Can only be called by an estimation method in the Copert class.
    
        Parameters
        ----------
        row : Series
            row used for calculation.
            It must have the following columns : TYPE, SPEED, TRAVEL_TIME, COUNT
            
        Returns
        -------
        CO2 value : float
            CO2 value for the given row.
        """

        if round(row["SPEED"]) ==  0 : 
            return 0

        # conversion of SPEED values from float to string
        speed = str(round(row["SPEED"]))

        # vehicle type
        veh_type = row["TYPE"]

        # copert's calculation

        if veh_type == "Bus": 
            mask = (self._copert_data["type"]=="Buses") & (self._copert_data["pollutant"]=="CO2") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            if self._copert_data.loc[mask, speed ].size == 0 : logger.info(f"{row}")
            return self._copert_data.loc[mask , speed ].values[0] * row["DISTANCE"]

        if veh_type == "Car": 
            mask = (self._copert_data["type"]=="Passenger Cars") & (self._copert_data["pollutant"]=="CO2") & (self._copert_data["load"]==50) & (self._copert_data["slope"]==0)
            if self._copert_data.loc[mask, speed ].size == 0 : logger.info(f"{row}")
            return self._copert_data.loc[mask, speed ].values[0] * row["DISTANCE"]

        if veh_type == "Metro": 
            return 0

        if veh_type == "Tram": 
            return 0

    def get_nox(self, period=None, area=None, mobility_services=None):
        """
        Return NOX emissions from _estimation after mobility services 
        discrimination and spatio-temporal aggregation.

        Parameters
        ----------
        period : None or [str, str]
            Time period as strings "HH:MM:SS". Only the hour part is used.
            Example: ["08:00:00", "10:30:00"]
            If None → entire dataset.
        area : None or shapely Polygon
            Spatial filtering polygon.
        mobility_services : None or list[str]
            TYPE values to keep.

        Returns
        -------
        float
            Total NOX emissions after filters.
    """

        # check
        if self._estimation.empty:
            logger.info("_estimation is empty.")
            raise ValueError("_estimation is empty.")

        df = self._estimation.copy()

        if period is not None:
            if (
                not isinstance(period, (list, tuple))
                or len(period) != 2
                or not all(isinstance(x, str) for x in period)
            ):
                raise ValueError(
                    "period must be None or ['HH:MM:SS', 'HH:MM:SS']"
                )

            start_str, end_str = period

            # Convert strings to Python time objects
            try:
                start_t = pd.to_datetime(start_str).time()
                end_t   = pd.to_datetime(end_str).time()
            except Exception:
                raise ValueError("Invalid time format for period. Expected 'HH:MM:SS'")

            # Extract TIME as pure time (hours only)
            #df["HOUR"] = df["TIME"].dt.time

            # Case 1 — normal interval (08:00:00 → 12:00:00)
            #if start_t <= end_t:
                #mask = (df["HOUR"] >= start_t) & (df["HOUR"] <= end_t)
            # Case 2 — interval crossing midnight (23:00:00 → 02:00:00)
            #else:
                #mask = (df["HOUR"] >= start_t) | (df["HOUR"] <= end_t)

            #df = df[mask].drop(columns="HOUR")


            t_end = df["TIME"] + df["PERIOD"]

            if start_t <= end_t:
                mask = (
                    (df["TIME"].dt.time >= start_t) &
                    (t_end.dt.time <= end_t)
                )
            else:
                mask = (
                    (df["TIME"].dt.time >= start_t) |
                    (t_end.dt.time <= end_t)
                )
            df = df[mask]


            if df.empty:
                raise ValueError("No data found for the given hourly period.")

        if area is not None:
            if not isinstance(area, Polygon):
                raise ValueError("area must be a shapely Polygon")

            # .within works with LINESTRING (it checks if the whole line is inside)
            df = df[df["GEOMETRY"].within(area)]

            if df.empty:
                raise ValueError("No data found in the given spatial area.")

        if mobility_services is not None:
            if not (
                isinstance(mobility_services, (list, tuple))
                and all(isinstance(x, str) for x in mobility_services)
            ):
                raise ValueError("mobility_services must be None or list[str]")

            available = df["TYPE"].unique()

            if not any(ms in available for ms in mobility_services):
                raise ValueError(
                    f"No TYPE values match mobility_services {mobility_services}. "
                    f"Available types: {list(available)}"
                )

            df = df[df["TYPE"].isin(mobility_services)]

            if df.empty:
                raise ValueError("No data left after mobility_services filtering.")

        if "NOX" not in df.columns:
            raise ValueError("Column NOX is missing from _estimation.")

        total_nox = df["NOX"].sum()

        logger.info(
            f"NOX estimation done with period={period if period else 'all'}, area={area if area else 'all'}, "
            f"services={mobility_services}, Total NOX = {total_nox}"
        )

        return total_nox


    def get_co2(self, period=None, area=None, mobility_services=None):
        """
        Return CO2 emissions from _estimation after mobility services 
        discrimination and spatio-temporal aggregation.

        Parameters
        ----------
        period : None or [str, str]
            Time period as strings "HH:MM:SS". Only the hour part is used.
            Example: ["08:00:00", "10:30:00"]
            If None → entire dataset.
        area : None or shapely Polygon
            Spatial filtering polygon.
        mobility_services : None or list[str]
            TYPE values to keep.

        Returns
        -------
        float
            Total CO2 emissions after filters.
        """

        # check
        if self._estimation.empty:
            logger.info("_estimation is empty.")
            raise ValueError("_estimation is empty.")

        df = self._estimation.copy()

        if period is not None:
            if (
                not isinstance(period, (list, tuple))
                or len(period) != 2
                or not all(isinstance(x, str) for x in period)
            ):
                raise ValueError(
                    "period must be None or ['HH:MM:SS', 'HH:MM:SS']"
                )

            start_str, end_str = period

            # Convert strings to Python time objects
            try:
                start_t = pd.to_datetime(start_str).time()
                end_t   = pd.to_datetime(end_str).time()
            except Exception:
                raise ValueError("Invalid time format for period. Expected 'HH:MM:SS'")

            # Extract TIME as pure time (hours only)
            #df["HOUR"] = df["TIME"].dt.time

            # Case 1 — normal interval (08:00:00 → 12:00:00)
            #if start_t <= end_t:
                #mask = (df["HOUR"] >= start_t) & (df["HOUR"] <= end_t)
            # Case 2 — interval crossing midnight (23:00:00 → 02:00:00)
            #else:
                #mask = (df["HOUR"] >= start_t) | (df["HOUR"] <= end_t)

            #df = df[mask].drop(columns="HOUR")

            t_end = df["TIME"] + df["PERIOD"]

            if start_t <= end_t:
                mask = (
                    (df["TIME"].dt.time >= start_t) &
                    (t_end.dt.time <= end_t)
                )
            else:
                mask = (
                    (df["TIME"].dt.time >= start_t) |
                    (t_end.dt.time <= end_t)
                )
            df = df[mask]

            if df.empty:
                raise ValueError("No data found for the given hourly period.")

        if area is not None:
            if not isinstance(area, Polygon):
                raise ValueError("area must be a shapely Polygon")

            # .within works with LINESTRING (it checks if the whole line is inside)
            df = df[df["GEOMETRY"].within(area)]

            if df.empty:
                raise ValueError("No data found in the given spatial area.")

        if mobility_services is not None:
            if not (
                isinstance(mobility_services, (list, tuple))
                and all(isinstance(x, str) for x in mobility_services)
            ):
                raise ValueError("mobility_services must be None or list[str]")

            available = df["TYPE"].unique()

            if not any(ms in available for ms in mobility_services):
                raise ValueError(
                    f"No TYPE values match mobility_services {mobility_services}. "
                    f"Available types: {list(available)}"
                )

            df = df[df["TYPE"].isin(mobility_services)]

            if df.empty:
                raise ValueError("No data left after mobility_services filtering.")

        if "CO2" not in df.columns:
            raise ValueError("Column CO2 is missing from _estimation.")

        total_co2 = df["CO2"].sum()

        logger.info(
            f"CO2 estimation done with period={period if period else 'all'}, area={area if area else 'all'}, "
            f"services={mobility_services}, Total CO2 = {total_co2}"
        )

        return total_co2


    def get_mean_speed(self, period=None, area=None, mobility_services=None):
        """
        Get mean speed by period, area and mobility services. If None takes all.

        Parameters
        ----------
        period : None or [str, str]
            Time period as strings "HH:MM:SS". Only the hour part is used.
            Example: ["08:00:00", "10:30:00"]
            If None → entire dataset.
        area : None or shapely Polygon
            Spatial filtering polygon.
        mobility_services : None or list[str]
            TYPE values to keep.

        Returns
        -------
        float
            Mean speed after filters.
        """

        # check
        if self._estimation.empty:
            logger.info("_estimation is empty.")
            raise ValueError("_estimation is empty.")

        df = self._estimation.copy()

        if period is not None:
            if (
                not isinstance(period, (list, tuple))
                or len(period) != 2
                or not all(isinstance(x, str) for x in period)
            ):
                raise ValueError(
                    "period must be None or ['HH:MM:SS', 'HH:MM:SS']"
                )

            start_str, end_str = period

            # Convert strings to Python time objects
            try:
                start_t = pd.to_datetime(start_str).time()
                end_t   = pd.to_datetime(end_str).time()
            except Exception:
                raise ValueError("Invalid time format for period. Expected 'HH:MM:SS'")

            # Extract TIME as pure time (hours only)
            #df["HOUR"] = df["TIME"].dt.time

            # Case 1 — normal interval (08:00:00 → 12:00:00)
            #if start_t <= end_t:
                #mask = (df["HOUR"] >= start_t) & (df["HOUR"] <= end_t)
            # Case 2 — interval crossing midnight (23:00:00 → 02:00:00)
            #else:
                #mask = (df["HOUR"] >= start_t) | (df["HOUR"] <= end_t)

            #df = df[mask].drop(columns="HOUR")

            t_end = df["TIME"] + df["PERIOD"]

            if start_t <= end_t:
                mask = (
                    (df["TIME"].dt.time >= start_t) &
                    (t_end.dt.time <= end_t)
                )
            else:
                mask = (
                    (df["TIME"].dt.time >= start_t) |
                    (t_end.dt.time <= end_t)
                )
            df = df[mask]

            if df.empty:
                raise ValueError("No data found for the given hourly period.")

        if area is not None:
            if not isinstance(area, Polygon):
                raise ValueError("area must be a shapely Polygon")

            # .within works with LINESTRING (it checks if the whole line is inside)
            df = df[df["GEOMETRY"].within(area)]

            if df.empty:
                raise ValueError("No data found in the given spatial area.")

        if mobility_services is not None:
            if not (
                isinstance(mobility_services, (list, tuple))
                and all(isinstance(x, str) for x in mobility_services)
            ):
                raise ValueError("mobility_services must be None or list[str]")

            available = df["TYPE"].unique()

            if not any(ms in available for ms in mobility_services):
                raise ValueError(
                    f"No TYPE values match mobility_services {mobility_services}. "
                    f"Available types: {list(available)}"
                )

            df = df[df["TYPE"].isin(mobility_services)]

            if df.empty:
                raise ValueError("No data left after mobility_services filtering.")

        if "SPEED" not in df.columns:
            raise ValueError("Column SPEED is missing from _estimation.")

        mean_speed = df["SPEED"].mean()

        logger.info(
            f"Mean speed calculation done with period={period if period else 'all'}, area={area if area else 'all'}, "
            f"services={mobility_services}, mean speed = {mean_speed}"
        )

        return mean_speed

    def get_mean_distance(self, period=None, area=None, mobility_services=None):
        """
        Get mean distance by period, area and mobility services. If None takes all.

        Parameters
        ----------
        period : None or [str, str]
            Time period as strings "HH:MM:SS". Only the hour part is used.
            Example: ["08:00:00", "10:30:00"]
            If None → entire dataset.
        area : None or shapely Polygon
            Spatial filtering polygon.
        mobility_services : None or list[str]
            TYPE values to keep.

        Returns
        -------
        float
            Mean distance after filters.
        """

        # check
        if self._estimation.empty:
            logger.info("_estimation is empty.")
            raise ValueError("_estimation is empty.")

        df = self._estimation.copy()

        if period is not None:
            if (
                not isinstance(period, (list, tuple))
                or len(period) != 2
                or not all(isinstance(x, str) for x in period)
            ):
                raise ValueError(
                    "period must be None or ['HH:MM:SS', 'HH:MM:SS']"
                )

            start_str, end_str = period

            # Convert strings to Python time objects
            try:
                start_t = pd.to_datetime(start_str).time()
                end_t   = pd.to_datetime(end_str).time()
            except Exception:
                raise ValueError("Invalid time format for period. Expected 'HH:MM:SS'")

            # Extract TIME as pure time (hours only)
            #df["HOUR"] = df["TIME"].dt.time

            # Case 1 — normal interval (08:00:00 → 12:00:00)
            #if start_t <= end_t:
                #mask = (df["HOUR"] >= start_t) & (df["HOUR"] <= end_t)
            # Case 2 — interval crossing midnight (23:00:00 → 02:00:00)
            #else:
                #mask = (df["HOUR"] >= start_t) | (df["HOUR"] <= end_t)

            #df = df[mask].drop(columns="HOUR")

            t_end = df["TIME"] + df["PERIOD"]

            if start_t <= end_t:
                mask = (
                    (df["TIME"].dt.time >= start_t) &
                    (t_end.dt.time <= end_t)
                )
            else:
                mask = (
                    (df["TIME"].dt.time >= start_t) |
                    (t_end.dt.time <= end_t)
                )
            df = df[mask]

            if df.empty:
                raise ValueError("No data found for the given hourly period.")

        if area is not None:
            if not isinstance(area, Polygon):
                raise ValueError("area must be a shapely Polygon")

            # .within works with LINESTRING (it checks if the whole line is inside)
            df = df[df["GEOMETRY"].within(area)]

            if df.empty:
                raise ValueError("No data found in the given spatial area.")

        if mobility_services is not None:
            if not (
                isinstance(mobility_services, (list, tuple))
                and all(isinstance(x, str) for x in mobility_services)
            ):
                raise ValueError("mobility_services must be None or list[str]")

            available = df["TYPE"].unique()

            if not any(ms in available for ms in mobility_services):
                raise ValueError(
                    f"No TYPE values match mobility_services {mobility_services}. "
                    f"Available types: {list(available)}"
                )

            df = df[df["TYPE"].isin(mobility_services)]

            if df.empty:
                raise ValueError("No data left after mobility_services filtering.")

        if "DISTANCE" not in df.columns:
            raise ValueError("Column DISTANCE is missing from _estimation.")

        mean_distance = df["DISTANCE"].mean()

        logger.info(
            f"Mean distance calculation done with period={period if period else 'all'}, area={area if area else 'all'}, "
            f"services={mobility_services}, mean distance = {mean_distance}"
        )

        return mean_distance



    def copert_parameters_analysis(self, periods=["auto","3min","6min","10min"]):
        """
        Estimates emissions for all combinations of time interval in period and method in self._methods.
        Return a dataframe with column PERIOD, METHOD, NOX, CO2.

        Parameters
        ----------
        periods : list
            Time periods (default = ["3min","6min","10min"]).

        Returns
        -------
        DataFrame : pd.DataFrame
            dataframe with column PERIOD, METHOD, NOX, CO2.
        """

        methods = self._methods.copy()

        analysis_df = pd.DataFrame(columns=["PERIOD", "METHOD", "NOX", "CO2"])

        for period in periods:
            for method in methods:
                self.copert_estimation(period=period, method=method)
                if period == "auto" : 
                    row = {
                    "PERIOD": 0,
                    "METHOD": method,
                    "NOX":   self.get_nox(),
                    "CO2":   self.get_co2()
                    }
                else : 
                    row = {
                    "PERIOD": freq_to_seconds(period) / 3600,
                    "METHOD": method,
                    "NOX":   self.get_nox(),
                    "CO2":   self.get_co2()
                    }

                analysis_df = pd.concat([analysis_df, pd.DataFrame([row])], ignore_index=True)


        logger.info(f"Copert parameters analysis with periods : {periods} and methods : {self._methods}.")
        return analysis_df.copy()


    # getters

    def get_simulation_outputs_directory(self):
        return self._simulation_outputs_directory

    def get_veh_data(self):
        return self._veh_data

    def get_crs(self):
        return self._crs

    def get_copert_data_path(self):
        return self._copert_data_path

    def get_copert_data(self):
        return self._copert_data.copy()

    def get_estimation(self):
        return self._estimation.copy()

        