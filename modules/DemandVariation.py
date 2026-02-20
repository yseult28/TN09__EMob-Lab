

# dependencies

import os

import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import truncnorm
from scipy.stats import beta

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
    return df

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

def list_of_lists_to_string(list_of_lists):
    """
    Converts a list of lists into a single string.

    Each sublist becomes a section separated by '_'.
    Each element within a sublist is also separated by '_'.
    If a sublist is None, the string 'None' is used.

    Parameters
    ----------
    list_of_lists : list
        A list containing sublists or None.

    Returns
    -------
    str
        The resulting string according to the rules above.
    """
    if list_of_lists is None:
        return "None"

    sections = []
    for sublist in list_of_lists:
        if sublist is None:
            sections.append("None")
        else:
            # convert each element to str and join with '_'
            sections.append("-".join(str(x) for x in sublist))
    
    # join all sections with '_'
    return "_".join(sections)

class DemandVariation():
    """
    Class used to produce variations of a MnMS demand file.
    The variations included only influence the demand, not the network definition.
    A DemandVariation object is initialized with the relative path to the original demand file targeted.
    It can be changed with the method change_original_demand_file(path:str).
    To produce variations, call the method corresponding to the wanted variation and add parameters such as the number of variations to produce,
    arguments relative to the method and the path to the directory to save them.
    If you only need one variation, you can use the method apply_variation and add all the parameters for the wanted variation.
    The method apply_variation produces a variation for a given df and is used by other methods in the class to produce total headcount variations 
    or cluster variations.
    Every variation name follow the same pattern : 
    Variation_<variation number>__<variation_type>__<method and law>__<parameters>__<argument>.csv
    example : Variation_6__Total-Ratio__WeightedSampling-Uniform__None__1.1.csv

    Attributes
    ---------
    _original_demand_path : str
        Relative path to the original demand file.
    _original_demand_df : pandas.DataFrame
        original demand dataframe.
    _types : set
        set of variation types included in the class.
    _methods : set
        set of variation method names included in the class.
    _laws : set
        set of probability distributions included in the class.
    """

    # Constructor
    
    def __init__(self, original_demand_path=""):
        """
        DemandVariation constructor.

        Parameters
        ----------
        original_demand_path : str
            Relative path to the original demand file.
            
        Returns
        -------
        None
        """

        # check
        if not original_demand_path or original_demand_path.strip() == "":
            logger.error("Invalid or null original demand path.")
            raise ValueError("Invalid or null original demand path.")

        # assignment
        self._original_demand_path = original_demand_path # Relative path to the original demand file.
        self._types = {"Ratio", "Quantity"}
        self._methods = {"WeightedSampling", "Density"}
        self._laws = {"Uniform", "Normal"}

        # loading the dataframe
        self._original_demand_df = pd.read_csv(self._original_demand_path, sep=';')

        # preprocessing
        self._original_demand_df = process_time_variable(self._original_demand_df, "DEPARTURE")
        
        logger.info(f"DemandVariation object built with _original_demand_path :{self._original_demand_path}.")


    
    # Configuration methods

    def change_original_demand_file(self, new_original_demand_path=""):
        """
        Change the original demand relative path and load the new original demand.

        Parameters
        ----------
        new_original_demand_path : str
            Relative path to the new original demand file.

        Returns
        ------
        None
        """

        # check
        if not new_original_demand_path or new_original_demand_path.strip() == "":
            logger.error("Invalid or null  new original demand path.")
            raise ValueError("Invalid or null new original demand path.")

        # assignment
        self._original_demand_path = new_original_demand_path # Relative path to the original demand file.
        

        # loading the dataframe
        self._original_demand_df = pd.read_csv(self._original_demand_path, sep=';')

        # preprocessing
        self._original_demand_df = process_time_variable(self._original_demand_df, "DEPARTURE")
        
        logger.info(f"Original demand path changed : {self._original_demand_path}.")



    # Variations methods

    # level 1

    def apply_variation(self, variation_type="Ratio",variation_parameter=[1, "WeightedSampling", "Uniform", None, 42], target_df=pd.DataFrame()):
        """
        Apply variation to a given dataframe.

        Parameters
        ----------
        variation_type : str, optional
            The way the given dataframe is modified (default = "Ratio").
        variation_parameter : list, optional
            Variation parameters with format [target quantity or ratio, type, law, law paramaters, random state] (default = [1, "WeightedSampling", "Uniform", None, 42]).
        target_df : pandas.DataFrame
            Dataframe to modify.
            
        Returns
        -------
        pandas.DataFrame
            Modified dataframe.
        """

        # check
        if not variation_type in self._types :
            logger.error("Invalid variation_type.")
            raise ValueError("Invalid variation_type.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")

        if variation_type == "Ratio" : 
            return self.apply_ratio_variation(variation_parameter, target_df)
        if variation_type == "Quantity" : 
            return self.apply_quantity_variation(variation_parameter, target_df)

    # level 2

    def apply_ratio_variation(self, variation_parameters, target_df):
        """
        Apply ratio variation to a given dataframe.

        Parameters
        ----------
        variation_parameter : list, optional
            Variation parameters with format [target ratio, type, law, law paramaters, random state] (default = [1, "WeightedSampling", "Uniform", None, 42]).
        target_df : pandas.DataFrame
            Dataframe to modify.
            
        Returns
        -------
        pandas.DataFrame
            Modified dataframe.
        """
        target_ratio, method, law, parameters, random_state = variation_parameters
        
        # check
        if not method in self._methods : 
            logger.error("Invalid method.")
            raise ValueError("Invalid method.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")
            
        if method == "WeightedSampling" :
            return self.apply_ratio_weightedsampling_variation(target_ratio, law, parameters, random_state, target_df)
        if method == "Density" :
            return self.apply_ratio_density_variation(target_ratio, law, parameters, random_state, target_df)


    def apply_quantity_variation(self, variation_parameters, target_df):
        """
        Apply ratio variation to a given dataframe.

        Parameters
        ----------
        variation_parameter : list, optional
            Variation parameters with format [target quantity, type, law, law paramaters, random state] (default = [1, "WeightedSampling", "Uniform", None, 42]).
        target_df : pandas.DataFrame
            Dataframe to modify.
            
        Returns
        -------
        pandas.DataFrame
            Modified dataframe.
        """
        target_quantity, method, law, parameters, random_state = variation_parameters
        
        # check
        if not method in self._methods : 
            logger.error("Invalid method.")
            raise ValueError("Invalid method.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")
            
        if method == "WeightedSampling" :
            return self.apply_quantity_weightedsampling_variation(target_quantity, law, parameters, random_state, target_df)
        if method == "Density" :
            return self.apply_quantity_density_variation(target_quantity, law, parameters, random_state, target_df)

    # level 3

    def apply_ratio_weightedsampling_variation(self, target_ratio, law, parameters, random_state, target_df):
        """
        Apply ratio weightedsampling variation to a given dataframe.

        Parameters
        ----------
        target_ratio : int
            Target ratio.
        law : str
            Probability law.
        parameters : str
            Probability law parameters.
        random_state : int
            Random state.
        target_df : pandas.DataFrame
            Dataframe to modify.
        Return
        ------
        pandas.DataFrame
        """    
        
        # check
        if not law in self._laws : 
            logger.error("Invalid law.")
            raise ValueError("Invalid law.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")

        if law == "Uniform" :
            return self.apply_ratio_weightedsampling_uniform_variation(target_ratio, parameters, random_state, target_df)
        if law == "Normal" :
            return self.apply_ratio_weightedsampling_normal_variation(target_ratio, parameters, random_state, target_df)

    def apply_ratio_density_variation(self, target_ratio, law, parameters, random_state, target_df):
        """
        Apply ratio density variation to a given dataframe.

        Parameters
        ----------
        target_ratio : int
            Target ratio.
        law : str
            Probability law.
        parameters : str
            Probability law parameters.
        random_state : int
            Random state.
        target_df : pandas.DataFrame
            Dataframe to modify.
            
        Returns
        -------
        pandas.DataFrame
        """    
        
        # check
        if not law in self._laws : 
            logger.error("Invalid law.")
            raise ValueError("Invalid law.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")

        if law == "Uniform" :
            return self.apply_ratio_density_uniform_variation(target_ratio, parameters, random_state, target_df)
        if law == "Normal" :
            return self.apply_ratio_density_normal_variation(target_ratio, parameters, random_state, target_df)


    def apply_quantity_weightedsampling_variation(self, target_quantity, law, parameters, random_state, target_df):
        """
        Apply quantity weightedsampling variation to a given dataframe.

        Parameters
        ----------
        target_quantity : int
            Target quantity.
        law : str
            Probability law.
        parameters : str
            Probability law parameters.
        random_state : int
            Random state.
        target_df : pandas.DataFrame
            Dataframe to modify.
            
        Returns
        -------
        pandas.DataFrame
        """  
        
        # check
        if not law in self._laws : 
            logger.error("Invalid law.")
            raise ValueError("Invalid law.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")

        if law == "Uniform" :
            return self.apply_quantity_weightedsampling_uniform_variation(target_quantity, parameters, random_state, target_df)
        if law == "Normal" :
            return self.apply_quantity_weightedsampling_normal_variation(target_quantity, parameters, random_state, target_df)

    def apply_quantity_density_variation(self, target_quantity, law, parameters, random_state, target_df):
        """
        Apply quantity density variation to a given dataframe.

        Parameters
        ----------
        target_quantity : int
            Target quantity.
        law : str
            Probability law.
        parameters : str
            Probability law parameters.
        random_state : int
            Random state.
        target_df : pandas.DataFrame
            Dataframe to modify.
            
        Returns
        -------
        pandas.DataFrame
        """  
        
        # check
        if not law in self._laws : 
            logger.error("Invalid law.")
            raise ValueError("Invalid law.")

        if target_df is None or target_df.empty : 
            logger.error("Invalid target_df.")
            raise ValueError("Invalid target_df.")

        if law == "Uniform" :
            return self.apply_quantity_density_uniform_variation(target_quantity, parameters, random_state, target_df)
        if law == "Normal" :
            return self.apply_quantity_density_normal_variation(target_quantity, parameters, random_state, target_df)


    # level 4

    def apply_ratio_weightedsampling_uniform_variation(self, target_ratio, parameters, random_state, target_df):
        """
        Apply a ratio-based variation using uniform weighted sampling
        restricted to a given time period.

        Parameters
        ----------
        target_ratio : float
            Global ratio (e.g., 1.2 → +20%, 0.8 → -20%) applied to TOTAL row count.
        parameters : list
            Either empty list [] → full DEPARTURE time span is used,
            or a list with one element: [ [t_start, t_end] ]  
            where t_start, t_end are datetime.time or str "HH:MM".
        random_state : int
            Seed for sampling reproducibility.
        target_df : pd.DataFrame
            Must contain at least the column DEPARTURE (datetime).

        Returns
        -------
        variation_df : pd.DataFrame
        """

        variation_df = target_df.copy()

        # check
        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain a DEPARTURE column (datetime).")

   
        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 1):
                raise ValueError("parameters must be [] or [[t_start, t_end]].")

            period = parameters[0]

            if not (isinstance(period, list) and len(period) == 2):
                raise ValueError("parameters[0] must be a list [t_start, t_end].")

            def parse_time(t):
                if isinstance(t, dt.time):
                    return t
                elif isinstance(t, str):
                    return dt.datetime.strptime(t, "%H:%M").time()
                else:
                    raise ValueError("Time must be datetime.time or 'HH:MM' string.")

            t_start = parse_time(period[0])
            t_end = parse_time(period[1])

        else:
            # Use full extent of the DEPARTURE column
            t_start = variation_df["DEPARTURE"].dt.time.min()
            t_end   = variation_df["DEPARTURE"].dt.time.max()
            parameters = [[t_start, t_end]]  # standardize structure for later

        # extract data inside period

        df_period = variation_df[
            variation_df["DEPARTURE"].dt.time.between(t_start, t_end)
        ]

        if df_period.empty:
            raise ValueError("No rows exist in target_df within the given time period.")

        # compute global target size

        original_total = len(variation_df)
        final_total = int(original_total * target_ratio)
        delta = final_total - original_total

        if delta == 0:
            # No change → just reassign IDs and return
            variation_df["ID"] = [i for i in range(variation_df.shape[0])]
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()

        rng = np.random.default_rng(random_state)


        # apply uniform weighted sampling in the TARGET PERIOD

        if delta > 0:
            # duplication
            sampled = df_period.sample(
                n=delta,
                replace=True,
                random_state=random_state
            )
            variation_df = pd.concat([variation_df, sampled], ignore_index=True)

        else:
            # deletion
            n_remove = -delta

            if n_remove >= len(df_period):
                raise ValueError(
                    f"Cannot remove {n_remove} rows: period only contains {len(df_period)} rows."
                )

            to_remove = df_period.sample(
                n=n_remove,
                replace=False,
                random_state=random_state
            ).index

            variation_df = variation_df.drop(index=to_remove).reset_index(drop=True)

        # reassign ID = 0,1,2,...,N-1

        variation_df["ID"] = [i for i in range(variation_df.shape[0])]
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()

    def apply_ratio_weightedsampling_normal_variation(self, target_ratio, parameters, random_state, target_df):
        """
        Apply a ratio-based variation using time-based Normal distribution weighting.

        Parameters
        ----------
        target_ratio : float
            Global row-count ratio (e.g., 1.2 = +20%).
        parameters : list
            Either `[]` meaning:
                - mean_time = center of DEPARTURE times
                - variance  = std of times (in minutes)

            Or `[mean_time, variance]` :
                - mean_time : datetime.time or "HH:MM"
                - variance  : float (in minutes)
        random_state : int
            Seed for reproducibility.
        target_df : pd.DataFrame
            Must contain a DEPARTURE column (datetime).

        Returns
        -------
        variation_df : pd.DataFrame
        """

        variation_df = target_df.copy()

        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain a DEPARTURE column.")

        # parameters parsing

        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("mean_time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 2):
                raise ValueError("parameters must be [] or [mean_time, variance_minutes].")

            mean_time = parse_time(parameters[0])
            variance_minutes = float(parameters[1])
            if variance_minutes <= 0:
                 raise ValueError("variance must be > 0")
        else:
            # AUTO-ESTIMATION depuis target_df
            dep_times = variation_df["DEPARTURE"].dt.time

            # convert times to float hours
            t_float = (
                dep_times.apply(lambda t: t.hour + t.minute/60 + t.second/3600)
            )

            mean_hour = t_float.mean()
            std_hour = t_float.std()

            # conversion to datetime.time
            mean_seconds = int(mean_hour * 3600)
            mean_time = dt.time(
                hour=mean_seconds // 3600,
                minute=(mean_seconds % 3600) // 60,
                second=(mean_seconds % 60)
            )

            variance_minutes = std_hour * 60

            parameters = [mean_time, variance_minutes]
    
        # compute final global target size

        original_total = len(variation_df)
        final_total = int(original_total * target_ratio)
        delta = final_total - original_total

        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()

        rng = np.random.default_rng(random_state)

        # build Normal weights on the entire time axis

        # convert DEPARTURE to minutes of day
        def time_to_minutes(t):
            return t.hour * 60 + t.minute + t.second / 60

        minutes_col = variation_df["DEPARTURE"].dt.time.apply(time_to_minutes)

        mean_minutes = time_to_minutes(mean_time)

        # weights from normal PDF
        weights = np.exp(-0.5 * ((minutes_col - mean_minutes) / variance_minutes)**2)

        # Avoid zeros
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()

        # apply sampling
        if delta > 0:
            # duplication
            sampled = variation_df.sample(
                n=delta,
                replace=True,
                weights=weights,
                random_state=random_state
            )
            variation_df = pd.concat([variation_df, sampled], ignore_index=True)

        else:
            # deletion
            n_remove = -delta

            if n_remove >= len(variation_df):
                raise ValueError("Cannot remove more rows than available.")

            to_remove = variation_df.sample(
                n=n_remove,
                replace=False,
                weights=weights,
                random_state=random_state
            ).index

            variation_df = variation_df.drop(index=to_remove).reset_index(drop=True)

        # reassign IDs
        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()



    def apply_quantity_weightedsampling_uniform_variation(self, delta_quantity, parameters, random_state, target_df):
        """
        Apply a quantity-based variation using uniform weighted sampling
        restricted to a given time period.

        Parameters
        ----------
            delta_quantity : int
            Positive → duplicate rows
            Negative → delete rows
        parameters : list
            []  → full DEPARTURE time span is used,
            [[t_start, t_end]] → time interval only.
        random_state : int
            Seed for reproducibility.
        target_df : pd.DataFrame
            Must contain DEPARTURE (datetime).

        Returns
        -------
         variation_df : pd.DataFrame
        """

        variation_df = target_df.copy()

        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain a DEPARTURE column (datetime).")


        # parameters parsing (same as in ratio version)
        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("Time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 1):
                raise ValueError("parameters must be [] or [[start,end]].")

            period = parameters[0]

            if not (isinstance(period, list) and len(period) == 2):
                raise ValueError("parameters[0] must be [t_start, t_end].")

            t_start = parse_time(period[0])
            t_end   = parse_time(period[1])

        else:
            # full time span automatically
            t_start = variation_df["DEPARTURE"].dt.time.min()
            t_end   = variation_df["DEPARTURE"].dt.time.max()
            parameters = [[t_start, t_end]]

        # extract period subset
        df_period = variation_df[
            variation_df["DEPARTURE"].dt.time.between(t_start, t_end)
        ]

        if df_period.empty:
            raise ValueError("No rows exist within the given time period.")


        # if delta = 0 → nothing to do
        delta = int(delta_quantity)

        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()


        # sampling
        if delta > 0:
            # duplication
            sampled = df_period.sample(
                n=delta,
                replace=True,
                random_state=random_state
            )
            variation_df = pd.concat([variation_df, sampled], ignore_index=True)

        else:
            # deletion
            n_remove = -delta

            if n_remove >= len(df_period):
                raise ValueError(
                    f"Cannot remove {n_remove} rows: period only contains {len(df_period)} rows."
                )

            to_remove = df_period.sample(
                n=n_remove,
                replace=False,
                random_state=random_state
            ).index

            variation_df = variation_df.drop(index=to_remove).reset_index(drop=True)

        # renumber IDs

        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()
     
    def apply_quantity_weightedsampling_normal_variation(self, delta_quantity, parameters, random_state, target_df):
        """
        Apply a quantity-based variation using a time-based Normal distribution
        weighting on the entire DEPARTURE span.

        Parameters
        ----------
        delta_quantity : int
            Positive → duplicate rows
            Negative → delete rows
        parameters : list
            []                  → auto mean + variance
            [mean_time, variance_minutes]
        random_state : int
            Sampling seed.
        target_df : pd.DataFrame
            Must contain DEPARTURE (datetime).

        Returns
        -------
        variation_df : pd.DataFrame
        """

        variation_df = target_df.copy()

        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain a DEPARTURE column.")

    # manage parameters (mean_time, variance_minutes)
        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("mean_time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 2):
                raise ValueError("parameters must be [] or [mean_time, variance_minutes].")

            mean_time = parse_time(parameters[0])
            variance_minutes = float(parameters[1])
            if variance_minutes <= 0:
                raise ValueError("variance must be > 0")

        else:
            # auto-derive mean + variance from data
            dep_times = variation_df["DEPARTURE"].dt.time
            t_float = dep_times.apply(lambda t: t.hour + t.minute/60 + t.second/3600)

            mean_hour = t_float.mean()
            std_hour = t_float.std()

            # convert to datetime.time
            mean_seconds = int(mean_hour * 3600)
            mean_time = dt.time(
                hour=mean_seconds // 3600,
                minute=(mean_seconds % 3600) // 60,
                second=(mean_seconds % 60)
            )

            variance_minutes = std_hour * 60
            parameters = [mean_time, variance_minutes]

        # if delta = 0 → nothing
        delta = int(delta_quantity)
        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()

        rng = np.random.default_rng(random_state)

        # build normal weights on entire DF
        def time_to_minutes(t):
            return t.hour * 60 + t.minute + t.second / 60

        minutes_col = variation_df["DEPARTURE"].dt.time.apply(time_to_minutes)
        mean_minutes = time_to_minutes(mean_time)

        weights = np.exp(-0.5 * ((minutes_col - mean_minutes) / variance_minutes)**2)
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()

        # apply sampling
        if delta > 0:
            # duplication
            sampled = variation_df.sample(
                n=delta,
                replace=True,
                weights=weights,
                random_state=random_state
            )
            variation_df = pd.concat([variation_df, sampled], ignore_index=True)

        else:
            # deletion
            n_remove = -delta

            if n_remove >= len(variation_df):
                raise ValueError("Cannot remove more rows than available.")

            to_remove = variation_df.sample(
                n=n_remove,
                replace=False,
                weights=weights,
                random_state=random_state
            ).index
    
            variation_df = variation_df.drop(index=to_remove).reset_index(drop=True)

        # renumber IDs
        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()

    def apply_ratio_density_uniform_variation(self, target_ratio, parameters, random_state, target_df):
        """
        Apply a ratio-based variation using uniform density on DEPARTURE times.

        Parameters
        ----------
        target_ratio : float
            Global row-count ratio (e.g., 1.2 = +20%).
        parameters : list
            [] → full DEPARTURE span is used
            [[t_start, t_end]] → interval to restrict uniform variation
        random_state : int
            Seed for reproducibility
        target_df : pd.DataFrame
            Must contain DEPARTURE column

        Returns
        -------
        variation_df : pd.DataFrame
        """
        variation_df = target_df.copy()
        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain DEPARTURE column.")

        rng = np.random.default_rng(random_state)

        # parse time interval
        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("Time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 1):
                raise ValueError("parameters must be [] or [[t_start, t_end]].")
            t_start, t_end = [parse_time(x) for x in parameters[0]]
        else:
            t_start = variation_df["DEPARTURE"].dt.time.min()
            t_end   = variation_df["DEPARTURE"].dt.time.max()
            parameters = [[t_start, t_end]]

        # convert times to minutes
        def time_to_minutes(t):
            return t.hour*60 + t.minute + t.second/60

        start_min, end_min = time_to_minutes(t_start), time_to_minutes(t_end)

        # extract period
        df_period = variation_df[
            variation_df["DEPARTURE"].dt.time.between(t_start, t_end)
        ]

        if df_period.empty:
            raise ValueError("No rows in target interval.")

        # compute delta rows
        original_total = len(variation_df)
        final_total = int(original_total * target_ratio)
        delta = final_total - original_total

        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()

        if delta > 0:
            # duplication
            sampled = df_period.sample(n=delta, replace=True, random_state=random_state).reset_index(drop=True)
            new_minutes = rng.uniform(start_min, end_min, size=delta)
            new_times = [dt.time(int(m)//60, int(m)%60, int((m%1)*60)) for m in new_minutes]
            sampled["DEPARTURE"] = new_times
            variation_df = pd.concat([variation_df, sampled], ignore_index=True)
        else:
            # deletion
            n_remove = -delta
            if n_remove >= len(df_period):
                raise ValueError(f"Cannot remove {n_remove} rows: only {len(df_period)} in interval.")
            remove_idx = df_period.sample(n=n_remove, replace=False, random_state=random_state).index
            variation_df = variation_df.drop(index=remove_idx).reset_index(drop=True)

        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()

    def apply_ratio_density_normal_variation(self, target_ratio, parameters, random_state, target_df):
        """
        Apply a ratio-based variation using Gaussian density sampling on DEPARTURE times.

        Parameters
        ----------
        target_ratio : float
            Global row-count ratio (e.g., 1.2 = +20%).
        parameters : list
            [] → auto mean + variance from DEPARTURE column
            [mean_time, variance_minutes] → mean_time: datetime.time or 'HH:MM', variance_minutes: float
        random_state : int
            Seed for reproducibility
        target_df : pd.DataFrame
            Must contain DEPARTURE (datetime)

        Returns
        -------
        variation_df : pd.DataFrame
        """
        variation_df = target_df.copy()

        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain a DEPARTURE column.")

        # parse parameters
        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("mean_time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 2):
                raise ValueError("parameters must be [] or [mean_time, variance_minutes].")
            mean_time = parse_time(parameters[0])
            variance_minutes = float(parameters[1])
            if variance_minutes <= 0:
                raise ValueError("variance_minutes must be > 0")
        else:
            # auto compute mean and variance
            dep_times = variation_df["DEPARTURE"].dt.time
            t_float = dep_times.apply(lambda t: t.hour + t.minute/60 + t.second/3600)
            mean_hour = t_float.mean()
            std_hour = t_float.std()
            mean_seconds = int(mean_hour*3600)
            mean_time = dt.time(hour=mean_seconds//3600,
                                minute=(mean_seconds%3600)//60,
                                second=(mean_seconds%60))
            variance_minutes = std_hour * 60
            parameters = [mean_time, variance_minutes]

        # compute delta rows
        original_total = len(variation_df)
        final_total = int(original_total * target_ratio)
        delta = final_total - original_total

        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()

        rng = np.random.default_rng(random_state)

        # convert DEPARTURE to minutes
        def time_to_minutes(t):
            return t.hour*60 + t.minute + t.second/60

        minutes_col = variation_df["DEPARTURE"].dt.time.apply(time_to_minutes)
        mean_minutes = time_to_minutes(mean_time)

        # duplication
        if delta > 0:
            new_minutes = rng.normal(loc=mean_minutes, scale=variance_minutes, size=delta)
            # clip to 0-1439 (minutes in a day)
            new_minutes = np.clip(new_minutes, 0, 24*60-1)
            new_times = [dt.time(int(m)//60, int(m)%60, int((m%1)*60)) for m in new_minutes]
            # duplicate random rows and set DEPARTURE to new_times
            sampled_rows = variation_df.sample(n=delta, replace=True, random_state=random_state).reset_index(drop=True)
            sampled_rows["DEPARTURE"] = new_times
            variation_df = pd.concat([variation_df, sampled_rows], ignore_index=True)

        else:
            # deletion: remove closest rows to Gaussian peak
            n_remove = -delta
            distances = (minutes_col - mean_minutes)**2
            remove_idx = distances.nlargest(n_remove).index  # farthest from peak
            variation_df = variation_df.drop(index=remove_idx).reset_index(drop=True)

        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()

    def apply_quantity_density_uniform_variation(self, delta_quantity, parameters, random_state, target_df):
        """
        Apply a quantity-based variation using uniform density on DEPARTURE times.

        Parameters
        ----------
        delta_quantity : int
            Positive → duplicate rows, Negative → delete rows
        parameters : list
            [] → full DEPARTURE span
            [[t_start, t_end]] → interval to restrict uniform variation
        random_state : int
            Seed for reproducibility
        target_df : pd.DataFrame
            Must contain DEPARTURE column

        Returns
        -------
        variation_df : pd.DataFrame
        """
        variation_df = target_df.copy()
        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain DEPARTURE column.")

        rng = np.random.default_rng(random_state)

        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("Time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 1):
                raise ValueError("parameters must be [] or [[t_start, t_end]].")
            t_start, t_end = [parse_time(x) for x in parameters[0]]
        else:
            t_start = variation_df["DEPARTURE"].dt.time.min()
            t_end   = variation_df["DEPARTURE"].dt.time.max()
            parameters = [[t_start, t_end]]

        def time_to_minutes(t):
            return t.hour*60 + t.minute + t.second/60

        start_min, end_min = time_to_minutes(t_start), time_to_minutes(t_end)

        df_period = variation_df[
            variation_df["DEPARTURE"].dt.time.between(t_start, t_end)
        ]

        if df_period.empty:
            raise ValueError("No rows in target interval.")

        delta = int(delta_quantity)
        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            variations_df = restore_time_variable(variation_df, "DEPARTURE")
            return variation_df.copy()

        if delta > 0:
            sampled = df_period.sample(n=delta, replace=True, random_state=random_state).reset_index(drop=True)
            new_minutes = rng.uniform(start_min, end_min, size=delta)
            new_times = [dt.time(int(m)//60, int(m)%60, int((m%1)*60)) for m in new_minutes]
            sampled["DEPARTURE"] = new_times
            variation_df = pd.concat([variation_df, sampled], ignore_index=True)
        else:
            n_remove = -delta
            if n_remove >= len(df_period):
                raise ValueError(f"Cannot remove {n_remove} rows: only {len(df_period)} in interval.")
            remove_idx = df_period.sample(n=n_remove, replace=False, random_state=random_state).index
            variation_df = variation_df.drop(index=remove_idx).reset_index(drop=True)

        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()

    def apply_quantity_density_normal_variation(self, delta_quantity, parameters, random_state, target_df):
        """
        Apply a quantity-based variation using Gaussian density sampling on DEPARTURE times.

        Parameters
        ----------
        delta_quantity : int
            Positive → duplicate rows, Negative → remove rows
        parameters : list
            [] → auto mean + variance
            [mean_time, variance_minutes]
        random_state : int
            Seed for reproducibility
        target_df : pd.DataFrame
            Must contain DEPARTURE column

        Returns
        -------
        variation_df : pd.DataFrame
        """
        variation_df = target_df.copy()

        if "DEPARTURE" not in variation_df.columns:
            raise ValueError("target_df must contain a DEPARTURE column.")

        # parse parameters
        def parse_time(t):
            if isinstance(t, dt.time):
                return t
            elif isinstance(t, str):
                return dt.datetime.strptime(t, "%H:%M").time()
            else:
                raise ValueError("mean_time must be datetime.time or 'HH:MM' string.")

        if parameters:
            if not (isinstance(parameters, list) and len(parameters) == 2):
                raise ValueError("parameters must be [] or [mean_time, variance_minutes].")
            mean_time = parse_time(parameters[0])
            variance_minutes = float(parameters[1])
            if variance_minutes <= 0:
                raise ValueError("variance_minutes must be > 0")
        else:
            dep_times = variation_df["DEPARTURE"].dt.time
            t_float = dep_times.apply(lambda t: t.hour + t.minute/60 + t.second/3600)
            mean_hour = t_float.mean()
            std_hour = t_float.std()
            mean_seconds = int(mean_hour*3600)
            mean_time = dt.time(hour=mean_seconds//3600,
                                minute=(mean_seconds%3600)//60,
                                second=(mean_seconds%60))
            variance_minutes = std_hour * 60
            parameters = [mean_time, variance_minutes]

        delta = int(delta_quantity)
        if delta == 0:
            variation_df["ID"] = list(range(len(variation_df)))
            return variation_df.copy()

        rng = np.random.default_rng(random_state)
        def time_to_minutes(t):
            return t.hour*60 + t.minute + t.second/60
        minutes_col = variation_df["DEPARTURE"].dt.time.apply(time_to_minutes)
        mean_minutes = time_to_minutes(mean_time)

        if delta > 0:
            new_minutes = rng.normal(loc=mean_minutes, scale=variance_minutes, size=delta)
            new_minutes = np.clip(new_minutes, 0, 24*60-1)
            new_times = [dt.time(int(m)//60, int(m)%60, int((m%1)*60)) for m in new_minutes]
            sampled_rows = variation_df.sample(n=delta, replace=True, random_state=random_state).reset_index(drop=True)
            sampled_rows["DEPARTURE"] = new_times
            variation_df = pd.concat([variation_df, sampled_rows], ignore_index=True)
        else:
            n_remove = -delta
            distances = (minutes_col - mean_minutes)**2
            remove_idx = distances.nlargest(n_remove).index
            variation_df = variation_df.drop(index=remove_idx).reset_index(drop=True)

        variation_df["ID"] = list(range(len(variation_df)))
        variations_df = restore_time_variable(variation_df, "DEPARTURE")
        return variation_df.copy()


    # Total headcount variations

    def apply_total_ratio_variation(self, origin="Unknown", method="WeightedSampling", law="Uniform", parameters=None, interval=[1,1,0.1], number_per_ratio=1, random_state=42, path="", car_only=False):
        """
        Creates total ratio variations.

        Parameters
        ----------
        origin : str, optional
            Origin of the source dataframe (default = "Unknown").
        method : str, optional
            Variation method (default = "WeightedSampling").
        law : str, optional
            Variation law (default = "Uniform").
        parameters : lsit, optional
            Law parameters (default = None).
        interval : list, optional
            Target ratios or quantities (default = [1,1,0.1]).
        number_per_ratio : int, optional
            Number of variation per target ratio (default = [1,1,0.1]).
        random_state : int, optional
            Random state (default = 42).
        path : str, optional
            Path to the source dataframe (default = "").
        car_only : bool, optional
            (default = False).

        Returns
        -------
        None
        """

        # check
        if not method in self._methods:
            logger.error("Invalid method.")
            raise ValueError("Invalid method.")
        if not law in self._laws:
            logger.error("Invalid law.")
            raise ValueError("invalid law.")


        if number_per_ratio < 1 or not isinstance(number_per_ratio, int) :
            logger.error("Invalid number_per_ratio.")
            raise ValueError("invalid number_per_ratio.")

        if not path or path.strip()=="":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        os.makedirs(path, exist_ok=True)

        # total ratio range
        total_ratios = np.arange(interval[0], interval[1] + interval[2] / 10, interval[2])

        variation_counter = 0
        for total_ratio in total_ratios :
            for i in range(number_per_ratio) : 
                variation_counter += 1
                rs = random_state + i
                logger.info(f"Creating variation {variation_counter} : total ratio : {total_ratio:.2f}, duplicata : {i}, random state {rs}.")

                variation_df = self._original_demand_df.copy()

                target_size = (variation_df.shape[0] * total_ratio).round()

                variation_df = self.apply_variation("Ratio",[total_ratio, method, law, parameters, random_state], variation_df)
                variation_df["ID"] = [i for i in range(variation_df.shape[0])]

                variation_df = process_time_variable(variation_df, "DEPARTURE")
                variation_df = variation_df.sort_values(by="DEPARTURE")
                variation_df = restore_time_variable(variation_df, "DEPARTURE")

                if car_only:
                    variation_df["MOBILITY SERVICES"] = "CAR"

                parameters_string = list_of_lists_to_string(parameters)
                
                variation_name = (f"{origin}"
                                 f"__Variation_{variation_counter}" 
                                 f"__Total_Ratio" 
                                 f"__{method}_{law}"
                                 f"__{parameters_string}"
                                 f"__{total_ratio:.2f}"
                                 f"__1_{total_ratio:.2f}")


                variation_path = f"{path}/{variation_name}.csv"

                variation_df.to_csv(variation_path, sep=';', index=False)


                logger.info(f"Variation {variation_counter} saved at {path}, ({len(variation_df)} rows).")
 
        logger.info(f"{variation_counter} total ratio variations, method : {method}, law : {law}, parameters : {parameters}, interval : {interval}, number_per_ratio : {number_per_ratio}, random_state : {random_state}, saved at path : {path}.")

    def apply_total_quantity_variation(self, origin="Unknown", method="WeightedSampling", law="Uniform", parameters=None, interval=[0,0,0.1], number_per_quantity=1, random_state=42, path="", car_only=False):
        """
        Creates total ratio variations.

        Parameters
        ----------
        origin : str, optional
            Origin of the source dataframe (default = "Unknown").
        method : str, optional
            Variation method (default = "WeightedSampling").
        law : str, optional
            Variation law (default = "Uniform").
        parameters : lsit, optional
            Law parameters (default = None).
        interval : list, optional
            Target ratios or quantities (default = [1,1,0.1]).
        number_per_quantity : int, optional
            Number of variation per target quantity (default = [1,1,0.1]).
        random_state : int, optional
            Random state (default = 42).
        path : str, optional
            Path to the source dataframe (default = "").
        car_only : bool, optional
            (default = False).

        Returns
        -------
        None
        """

        # check
        if not method in self._methods:
            logger.error("Invalid method.")
            raise ValueError("Invalid method.")
        if not law in self._laws:
            logger.error("Invalid law.")
            raise ValueError("invalid law.")


        if number_per_quantity < 1 or not isinstance(number_per_quantity, int) :
            logger.error("Invalid number_per_ratio.")
            raise ValueError("invalid number_per_ratio.")

        if not path or path.strip()=="":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        os.makedirs(path, exist_ok=True)

        # total ratio range
        total_quantities = np.arange(interval[0], interval[1] + interval[2] / 10, interval[2])

        variation_counter = 0
        for total_quantity in total_quantities :
            for i in range(number_per_quantity) : 
                variation_counter += 1
                rs = random_state + i
                logger.info(f"Creating variation {variation_counter} : total quantity : {total_quantity:.2f}, duplicata : {i}, random state {rs}.")

                variation_df = self._original_demand_df.copy()

                target_size = (variation_df.shape[0] * total_quantity).round()

                variation_df = self.apply_variation("Quantity",[total_quantity, method, law, parameters, random_state], variation_df)
                variation_df["ID"] = [i for i in range(variation_df.shape[0])]

                variation_df = process_time_variable(variation_df, "DEPARTURE")
                variation_df = variation_df.sort_values(by="DEPARTURE")
                variation_df = restore_time_variable(variation_df, "DEPARTURE")

                
                if car_only:
                    variation_df["MOBILITY SERVICES"] = "CAR"
                parameters_string = list_of_lists_to_string(parameters)
                
                variation_name = (f"{origin}"
                                 f"Variation_{variation_counter}" 
                                 f"__Total_Quantity" 
                                 f"__{method}_{law}"
                                 f"__{parameters_string}"
                                 f"__{total_ratio:.2f}"
                                 f"__1_{total_ratio:.2f}")



                variation_path = f"{path}/{variation_name}.csv"

                variation_df.to_csv(variation_path, sep=';', index=False)


                logger.info(f"Variation {variation_counter} saved at {path}, ({len(variation_df)} rows).")
 
        logger.info(f"{variation_counter} total quantity variations, method : {method}, law : {law}, parameters : {parameters}, interval : {interval}, number_per_quantity : {number_per_quantity}, random_state : {random_state}, saved at path : {path}.")

    def apply_classification_ratio_variation(
            self,
            origin="Unknown",
            method="WeightedSampling",
            law="Uniform",
            parameters=None,
            total_ratio=1,
            clusters=[],
            interval=[1,1,0.1],
            number_per_ratio=1,
            random_state=42,
            path="",
            car_only=False
        ):
        """
        Creates classification ratio variations on selected clusters. For each ratio value and cluster in clusters, a variation is produced and            saved at path. The subset of individuals associated with the target cluster is tranformed to have it count equal to its original count             time the target cluster, the remaining subset is then reshaped to target a global count equal to original global count time total_ratio.

        Parameters
        ----------
        origin : str
            label for the original demand.
        method : str
            Method used to produce variations.
        lax : str
            Theorical distribution used by the method.
        parameters : list
            Parameters of the given law.
        total_ratio : float
            Total ratio targeted.
        clusters : list
            Clusters targeted.
        interval : list
            Interval for the ratio values. pattern : np.arange(interval[0], interval[1] + interval[2] / 10, interval[2]).
        number_per_ratio : int
            Number of variations produced by cluster ratio couple.
        random_state : int
            Random state.
        path : str
            path to save the variations.
        car_only : bool
            If true, change only lines with Car or Personal Car as obility services (doesn't work).

        Returns
        -------
        None
        """

        # ------------------------
        # Checks
        # ------------------------
        if method not in self._methods:
            raise ValueError("Invalid method.")
        if law not in self._laws:
            raise ValueError("Invalid law.")
        if number_per_ratio < 1 or not isinstance(number_per_ratio, int):
            raise ValueError("Invalid number_per_ratio.")
        if not path or path.strip() == "":
            raise ValueError("Invalid or null path.")
        if not clusters:
            raise ValueError("Clusters list must not be empty.")

        os.makedirs(path, exist_ok=True)

        # Ratios to apply to target clusters
        ratios = np.arange(interval[0], interval[1] + interval[2] / 10, interval[2])

        original_df = self._original_demand_df.copy()
        original_size = original_df.shape[0]

        # store original cluster sizes
        cluster_original_sizes = original_df.groupby("CLUSTER_ID").size().to_dict()

        variation_counter = 0

        # ------------------------
        # Main loop
        # ------------------------
        for target_cluster in clusters:
            for ratio in ratios:
                for dup in range(number_per_ratio):

                    variation_counter += 1
                    rs = random_state + dup

                    logger.info(
                        f"[Variation {variation_counter}] "
                        f"Cluster={target_cluster} | TargetRatio={ratio:.2f} | dup={dup}"
                    )

                    df = original_df.copy()

                    # -------------------------------
                    # SPLIT DF IN TWO PARTS
                    # -------------------------------
                    df_target = df[df["CLUSTER_ID"] == target_cluster].copy()
                    df_others = df[df["CLUSTER_ID"] != target_cluster].copy()

                    # -------------------------------
                    # APPLY VARIATION TO TARGET CLUSTER
                    # -------------------------------
                    df_target_var = self.apply_variation(
                        "Ratio",
                        [ratio, method, law, parameters, rs],
                        df_target
                    )

                    logger.info(f"Original count : {len(df_target)}, new count : {len(df_target_var)}.")

                    size_target_final = df_target_var.shape[0]

                    # -------------------------------
                    # CALCULATE COMPENSATORY RATIO FOR THE WHOLE "OTHERS" BLOCK
                    # -------------------------------
                    target_total = original_size * total_ratio

                    size_others_original = df_others.shape[0]
                    remaining_needed = target_total - size_target_final

                    if remaining_needed < 0:
                        raise ValueError(
                            f"Target cluster ratio too large (cluster {target_cluster})"
                        )

                    ratio_comp = (
                        remaining_needed / size_others_original
                        if size_others_original > 0 else 1
                    )

                    # -------------------------------
                    # APPLY VARIATION ON ALL OTHER CLUSTERS AS A SINGLE BLOCK
                    # -------------------------------
                    df_others_var = self.apply_variation(
                        "Ratio",
                        [ratio_comp, method, law, parameters, rs],
                        df_others
                    )

                    # -------------------------------
                    # MERGE EVERYTHING
                    # -------------------------------
                    final_df = pd.concat([df_target_var, df_others_var], ignore_index=True)

                    # -------------------------------
                    # COMPUTE TRUE RATIOS PER CLUSTER
                    # -------------------------------
                    final_cluster_sizes = final_df.groupby("CLUSTER_ID").size().to_dict()

                    true_cluster_ratios = {
                        cid: final_cluster_sizes.get(cid, 0) / cluster_original_sizes[cid]
                        for cid in cluster_original_sizes
                    }

                    final_df["CLUSTER_RATIO"] = final_df["CLUSTER_ID"].apply(
                        lambda cid: true_cluster_ratios[cid]
                    )

                    # -------------------------------
                    # CLEAN OUTPUT
                    # -------------------------------
                    final_df = final_df.drop(columns=["CLUSTER_ID","CLUSTER_RATIO","SERVICE"])
                    final_df["ID"] = range(final_df.shape[0])

                    final_df = process_time_variable(final_df, "DEPARTURE")
                    final_df = final_df.sort_values(by="DEPARTURE")
                    final_df = restore_time_variable(final_df, "DEPARTURE")

                    if car_only:
                        final_df["SERVICE"] = "PersonalVehicle"

                    parameters_string = list_of_lists_to_string(parameters)

                    variation_name = (
                        f"{origin}"
                        f"__Variation_{variation_counter}"
                        f"__Classification_Quantity" 
                        f"__{method}_{law}"
                        f"__{parameters_string}"
                        f"__{total_ratio:.2f}"
                        f"__{target_cluster}_{ratio:.2f}"
                    )

                    out_file = f"{path}/{variation_name}.csv"
                    final_df.to_csv(out_file, sep=";", index=False)

                    logger.info(
                        f"Saved variation {variation_counter} | rows={final_df.shape[0]} | {out_file}"
                    )

        logger.info(
            f"{variation_counter} total variations created and saved in {path}"
        )

    def apply_classification_quantity_variation(
            self,
            origin="Unknown",
            method="WeightedSampling",
            law="Uniform",
            parameters=None,
            total_ratio=1,
            interval=[0, 0, 1],
            number_per_quantity=1,
            clusters=[],
            random_state=42,
            path="",
            car_only=False
        ):
        """
        Same as the ratio one but with quantity as parameters.
        """

        # ------------------------
        # Checks
        # ------------------------
        if method not in self._methods:
            raise ValueError("Invalid method.")
        if law not in self._laws:
            raise ValueError("Invalid law.")
        if number_per_quantity < 1 or not isinstance(number_per_quantity, int):
            raise ValueError("Invalid number_per_quantity.")
        if not path or path.strip() == "":
            raise ValueError("Invalid or null path.")
        if not clusters:
            raise ValueError("Clusters list must not be empty.")

        os.makedirs(path, exist_ok=True)

        # Absolute quantities (deltas) applied to target cluster
        target_quantities = np.arange(
            interval[0],
            interval[1] + interval[2],
            interval[2],
            dtype=int
        )

        original_df = self._original_demand_df.copy()
        original_size = original_df.shape[0]

        # Store original cluster sizes
        cluster_original_sizes = original_df.groupby("CLUSTER_ID").size().to_dict()

        variation_counter = 0

        # ------------------------
        # Main loop
        # ------------------------
        for target_cluster in clusters:
            for target_quantity in target_quantities:
                for dup in range(number_per_quantity):

                    variation_counter += 1
                    rs = random_state + dup

                    logger.info(
                        f"[Variation {variation_counter}] "
                        f"Cluster={target_cluster} | "
                        f"TargetQuantity={target_quantity:+d} | dup={dup}"
                    )

                    df = original_df.copy()

                    # -------------------------------
                    # SPLIT DATAFRAME
                    # -------------------------------
                    df_target = df[df["CLUSTER_ID"] == target_cluster].copy()
                    df_others = df[df["CLUSTER_ID"] != target_cluster].copy()

                    size_target_original = len(df_target)
                    size_others_original = len(df_others)

                    # -------------------------------
                    # APPLY ABSOLUTE QUANTITY TO TARGET CLUSTER
                    # -------------------------------
                    df_target_var = self.apply_variation(
                        "Quantity",
                        [target_quantity, method, law, parameters, rs],
                        df_target
                    )

                    size_target_final = df_target_var.shape[0]

                    if size_target_final < 0:
                        raise ValueError(
                            f"Target quantity {target_quantity} removes too many elements "
                            f"(cluster {target_cluster})"
                        )

                    logger.info(
                        f"Target cluster {target_cluster}: "
                        f"{size_target_original} → {size_target_final}"
                    )

                    # -------------------------------
                    # TOTAL FINAL SIZE (VIA total_ratio)
                    # -------------------------------
                    target_total = int(round(original_size * total_ratio))

                    # -------------------------------
                    # DELTA TO APPLY TO OTHERS (ABSOLUTE)
                    # -------------------------------
                    delta_others = (
                        target_total
                        - size_target_final
                    - size_others_original
                    )

                    # -------------------------------
                    # APPLY ABSOLUTE QUANTITY TO OTHERS (BLOCK)
                    # -------------------------------
                    df_others_var = self.apply_variation(
                        "Quantity",
                        [delta_others, method, law, parameters, rs],
                        df_others
                    )

                    # -------------------------------
                    # MERGE
                    # -------------------------------
                    final_df = pd.concat(
                        [df_target_var, df_others_var],
                        ignore_index=True
                    )

                    # -------------------------------
                    # COMPUTE TRUE FINAL CLUSTER QUANTITIES
                    # -------------------------------
                    final_cluster_sizes = final_df.groupby("CLUSTER_ID").size().to_dict()

                    true_cluster_quantities = {
                        cid: final_cluster_sizes.get(cid, 0)
                        for cid in cluster_original_sizes
                    }

                    final_df["CLUSTER_QUANTITY"] = final_df["CLUSTER_ID"].apply(
                        lambda cid: true_cluster_quantities[cid]
                    )

                    # -------------------------------
                    # CLEAN OUTPUT
                    # -------------------------------
                    final_df = final_df.drop(
                        columns=["CLUSTER_ID", "CLUSTER_QUANTITY", "SERVICE"],
                        errors="ignore"
                    )

                    final_df["ID"] = range(final_df.shape[0])

                    final_df = process_time_variable(final_df, "DEPARTURE")
                    final_df = final_df.sort_values(by="DEPARTURE")
                    final_df = restore_time_variable(final_df, "DEPARTURE")

                    if car_only:
                        final_df["SERVICE"] = "PersonalVehicle"

                    parameters_string = list_of_lists_to_string(parameters)

                    variation_name = (
                        f"{origin}"
                        f"__Variation_{variation_counter}"
                        f"__Classification_Quantity"
                        f"__{method}_{law}"
                        f"__{parameters_string}"
                        f"__{total_ratio:.2f}"
                        f"__{target_cluster}_{target_quantity:+d}"
                    )
    
                    out_file = f"{path}/{variation_name}.csv"
                    final_df.to_csv(out_file, sep=";", index=False)

                    logger.info(
                        f"Saved variation {variation_counter} | "
                        f"rows={final_df.shape[0]} | {out_file}"
                    )

        logger.info(
            f"{variation_counter} total classification quantity variations saved in {path}"
        )





    def apply_uniform_distribution_quantile(self, target_df=pd.DataFrame()):
        """
        Transform target_df temporal distribution to a Uniform distribution
        using quantiles (deterministic, order-preserving).
        
        Parameters
        ----------
        target_df : pandas.DataFrame()
            dataframe to transform.

        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """
        if target_df.empty:
            raise ValueError("Null dataframe.")
        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")

        df = target_df.copy()
        df = process_time_variable(df, "DEPARTURE")

        # Sort to preserve temporal order
        df = df.sort_values(by="DEPARTURE").reset_index(drop=True)

        times = df["DEPARTURE"]
        t_min = times.min()
        t_max = times.max()

        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        n = len(df)

        # Deterministic quantiles
        quantiles = (np.arange(n) + 0.5) / n

        # Uniform inverse CDF
        new_seconds = quantiles * total_seconds

        df["DEPARTURE"] = [
            t_min + pd.to_timedelta(s, unit="s") for s in new_seconds
        ]

        df = restore_time_variable(df, "DEPARTURE")
        return df

    def apply_uniform_distributions(self,demand_path="", directory=""):
        """
        Transform the demand_path dataframe temporal distribution to a Uniform distribution
        using quantiles (deterministic, order-preserving) and saves at directory.

        Parameters
        ----------
        demand_path : str
            Path to the dataframe.
        directory : str
            Path to save the transformed dataframe.

        Returns
        -------
        None
        """
        # check
        if not demand_path or demand_path.strip()=="":
            logger.info("Invalid demand path.")
            raise ValueError("Invalid demand path.")
        if not directory or directory.strip()=="":
            logger.info("Invalid directory.")
            raise ValueError("Invalid directory.")

        demand = pd.read_csv(demand_path, sep=';')
        
        os.makedirs(directory, exist_ok=True)

        logger.info(f"uniform variation.")
        buffer = demand.copy()
        var = self.apply_uniform_distribution_quantile(target_df=buffer)
        file_name = f"Temporal __Uniform.csv"
        full_path = f"{directory}/{file_name}"
        var.to_csv(full_path, sep=';', index=False)

        logger.info(f"Uniform variationsproduced and savec at directory : {directory}.")



    def apply_uniform_distribution(self, target_df=pd.DataFrame()):
        """
        Transforms target_df temporal distribution to be uniform
        without changing the total number of rows.
        The transformation is applied on the DEPARTURE column only.
        
        Parameters
        ----------
        target_df : pandas.DataFrame()
            dataframe to transform.

        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """

        # check
        if target_df.empty:
            logger.info("Null dataframe.")
            raise ValueError("Null dataframe.")

        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")

        df = target_df.copy()

        # Ensure DEPARTURE is datetime
        df = process_time_variable(df, "DEPARTURE")

        # Convert times to seconds since start of interval
        times = df["DEPARTURE"]

        t_min = times.min()
        t_max = times.max()

        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        seconds = (times - t_min).dt.total_seconds().to_numpy()

        # Empirical ranks → uniform distribution in [0, 1]
        ranks = pd.Series(seconds).rank(method="average")
        u = (ranks - 0.5) / len(seconds)

        # Map uniform quantiles to uniform time interval
        new_seconds = u * total_seconds

        # Convert back to datetime
        df["DEPARTURE"] = [
            t_min + pd.to_timedelta(s, unit="s") for s in new_seconds
        ]

        # Restore original string format
        df = restore_time_variable(df, "DEPARTURE")

        return df


    def apply_normal_distribution(self, parameters=[],target_df=pd.DataFrame()):
        """
        Transforms target_df temporal distribution to be normal
        without changing the total number of rows.

        Parameters
        ----------
        parameters : list
            [mean, std]
            mean     : float in [0, 1], relative position in the time interval
            std : float, std in seconds
        target_df : pandas.DataFrame
            dataframe to transform.

        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """

        # check
        if target_df.empty:
            logger.info("Null dataframe.")
            raise ValueError("Null dataframe.")

        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")

        if not isinstance(parameters, list) or len(parameters) != 2:
            raise ValueError("parameters must be [mean, std].")

        mean_rel = float(parameters[0])
        std_sec = float(parameters[1])

        if not (0.0 <= mean_rel <= 1.0):
            raise ValueError("mean must be in [0, 1].")

        if std_sec <= 0:
            raise ValueError("std must be > 0 (in seconds).")

        #std_sec = np.sqrt(variance_sec)

        df = target_df.copy()

        # Ensure DEPARTURE is datetime
        df = process_time_variable(df, "DEPARTURE")

        times = df["DEPARTURE"]

        # Define time interval
        t_min = times.min()
        t_max = times.max()
        total_seconds = (t_max - t_min).total_seconds()

        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        # Convert to seconds since interval start
        seconds = (times - t_min).dt.total_seconds().to_numpy()

        # Empirical CDF (quantiles)
        ranks = pd.Series(seconds).rank(method="average")
        u = (ranks - 0.5) / len(seconds)

        # Target normal parameters (absolute seconds)
        mean_sec = mean_rel * total_seconds

        # Quantile mapping to target normal distribution
        new_seconds = norm.ppf(u, loc=mean_sec, scale=std_sec)

        # Clip to valid interval
        new_seconds = np.clip(new_seconds, 0, total_seconds)

        # Convert back to datetime
        df["DEPARTURE"] = [
            t_min + pd.to_timedelta(s, unit="s") for s in new_seconds
        ]

        # Restore original string format
        df = restore_time_variable(df, "DEPARTURE")

        return df

    

    def apply_normal_distribution2(self, parameters=[], target_df=pd.DataFrame()):
        """
        Transform target_df temporal distribution to a Normal (Gaussian) law
        restricted to the interval [t_min, t_max] without changing the number of rows.

        Parameters
        ----------
        parameters : list
            [mean, std]
            mean     : float in [0, 1], relative position in the time interval
            std : float, std in seconds
        target_df : pandas.DataFrame
            dataframe to transform.

        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """
        if target_df.empty:
            raise ValueError("Null dataframe.")
        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise ValueError("parameters must be [mean, std].")

        mean_rel = float(parameters[0])
        std_sec = float(parameters[1])

        if not (0.0 <= mean_rel <= 1.0):
            raise ValueError("mean must be in [0, 1].")
        if std_sec <= 0:
            raise ValueError("std must be > 0 (in seconds).")

        df = target_df.copy()
        df = process_time_variable(df, "DEPARTURE")
        times = df["DEPARTURE"]

        t_min = times.min()
        t_max = times.max()
        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        n = len(df)
        mean_sec = mean_rel * total_seconds

        # Truncated normal bounds in std units
        a, b = (0 - mean_sec) / std_sec, (total_seconds - mean_sec) / std_sec

        # Sample truncated normal
        new_seconds = truncnorm.rvs(a, b, loc=mean_sec, scale=std_sec, size=n)

        # Convert back to datetime
        df["DEPARTURE"] = [t_min + pd.to_timedelta(s, unit="s") for s in new_seconds]

        df = df.sort_values(by="DEPARTURE", ascending=True)

        # Restore original string format
        df = restore_time_variable(df, "DEPARTURE")
        return df





    def apply_normal_distribution_quantile(self, parameters=[], target_df=pd.DataFrame()):
        """
        Transform target_df temporal distribution to a truncated Normal (Gaussian)
        using quantiles (deterministic, order-preserving).

        Parameters
        ----------
        parameters : list
            [mean, std]
            mean     : float in [0, 1], relative position in the time interval
            std : float, std in seconds
        target_df : pandas.DataFrame
            dataframe to transform.

        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """

        from scipy.stats import truncnorm
        import pandas as pd
        import numpy as np
    
        if target_df.empty:
            raise ValueError("Null dataframe.")
        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise ValueError("parameters must be [mean, std].")

        mean_rel = float(parameters[0])
        std_sec = float(parameters[1])

        if not (0.0 <= mean_rel <= 1.0):
            raise ValueError("mean must be in [0, 1].")
        if std_sec <= 0:
            raise ValueError("std must be > 0 (in seconds).")

        df = target_df.copy()
        df = process_time_variable(df, "DEPARTURE")

        # Sort to preserve temporal order
        df = df.sort_values(by="DEPARTURE").reset_index(drop=True)

        times = df["DEPARTURE"]
        t_min = times.min()
        t_max = times.max()

        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        n = len(df)
        mean_sec = mean_rel * total_seconds

        # Truncated normal bounds (in std units)
        a = (0 - mean_sec) / std_sec
        b = (total_seconds - mean_sec) / std_sec

        # Deterministic quantiles
        quantiles = (np.arange(n) + 0.5) / n

        # Inverse CDF (ppf)
        new_seconds = truncnorm.ppf(
            quantiles,
            a, b,
            loc=mean_sec,
            scale=std_sec
        )

        df["DEPARTURE"] = [
            t_min + pd.to_timedelta(s, unit="s") for s in new_seconds
        ]

        df = restore_time_variable(df, "DEPARTURE")
        return df


    def apply_normal_distributions(self, mean_interval=[], std_interval=[], demand_path="", directory=""):
        """
        Produce normal variations from demand_path demand with parameters in mean_interval et std_interval. The variations are saved at                    directory.

        Parameters
        ----------
        mean_interval : list
            Parameters to create a range of mean values. pattern : np.arange(mean_interval[0], mean_interval[1], mean_interval[2]).
        std_interval : list
            Parameters to create a range of std values. pattern : np.arange(std_interval[0], std_interval[1], std_interval[2]).
        demand_path : str
            Path the target dataframe.
        directory : str
            Path to save the variations.

        Returns
        -------
        None
        """
        # check
        if len(mean_interval)!=3:
            logger.info("Invalid mean parameters.")
            raise ValueError("Invalid mean parameters.")
        if len(std_interval)!=3:
            logger.info("Invalid std parameters.")
            raise ValueError("Invalid std parameters.")
        if not demand_path or demand_path.strip()=="":
            logger.info("Invalid demand path.")
            raise ValueError("Invalid demand path.")
        if not directory or directory.strip()=="":
            logger.info("Invalid directory.")
            raise ValueError("Invalid directory.")


        mean_range = np.arange(mean_interval[0], mean_interval[1], mean_interval[2])
        std_range = np.arange(std_interval[0], std_interval[1], std_interval[2])

        demand = pd.read_csv(demand_path, sep=';')
        
        os.makedirs(directory, exist_ok=True)

        for mean in mean_range : 
            for std in std_range : 
                logger.info(f"normal variation, mean : {mean}, std : {std}.")
                buffer = demand.copy()
                var = self.apply_normal_distribution2(parameters=[mean, std], target_df=buffer)
                file_name = f"Temporal __Normal__{mean:.2f}_{std:.2f}.csv"
                full_path = f"{directory}/{file_name}"
                var.to_csv(full_path, sep=';', index=False)

        logger.info(f"variations produced with mean_interval : {mean_interval}, std_interval : {std_interval} and directory : {directory}.")
                
                



    def apply_beta_distribution_ab(self, parameters=[], target_df=pd.DataFrame(),
                               fond_=0.0005):
        """
        Transform the temporal distribution of target_df according to a Beta distribution,
        with a constant minimal baseline before and after the peak, while preserving the order of individuals.

        Parameters
        ----------
        parameters : list
            [alpha, beta]
            alpha : float
            beta : float
        target : pandas.DataFrame
            dataframe to transform.
        fond_ : float
            constant minimal baseline.

        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """
        import numpy as np
        from scipy.stats import beta

        if target_df.empty:
            raise ValueError("Null dataframe.")
        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise ValueError("parameters must be [alpha, beta].")

        alpha, beta_param = parameters
        if alpha <= 0 or beta_param <= 0:
            raise ValueError("alpha and beta must be > 0.")

        df = target_df.copy()
        df = process_time_variable(df, "DEPARTURE")
        times = df["DEPARTURE"]

        t_min = times.min()
        t_max = times.max()
        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        n = len(df)

        # Tirage dense pour construire la densité combinée
        x_grid = np.linspace(0, 1, 1000)

        # Calcul du mode si alpha>1 et beta>1
        mode = (alpha - 1) / (alpha + beta_param - 2) if alpha > 1 and beta_param > 1 else 0.5

        # Densité Beta
        beta_density = beta.pdf(x_grid, alpha, beta_param)
        beta_density /= beta_density.sum()  # normalisation

        # Fond minimal avant et après le pic
        fond = np.where(x_grid < mode, fond_, fond_)

        # Combinaison Beta + fond
        combined_density = beta_density + fond
        combined_density /= combined_density.sum()  # normalisation totale

        # Calcul CDF pour mapping des quantiles
        cdf = np.cumsum(combined_density)
        cdf /= cdf[-1]

        # Conserver l'ordre : map les quantiles uniformes selon l'ordre original
        quantiles = np.linspace(0, 1, n)
        x_samples_ordered = np.interp(quantiles, cdf, x_grid)

        # Conversion en secondes
        new_seconds = x_samples_ordered * total_seconds
        df["DEPARTURE"] = [t_min + pd.to_timedelta(s, unit="s") for s in new_seconds]

        df = df.sort_values(by="DEPARTURE", ascending=True)


        # Restaurer format original
        df = restore_time_variable(df, "DEPARTURE")
        return df



    def apply_beta_distribution_mc(self, parameters=[], target_df=pd.DataFrame()):
        """
        Transform target_df temporal distribution to a Beta law using
        mean / concentration parametrization.

        Parameters
        ----------
        parameters : list
            parameters = [mean_rel, concentration]
            mean_rel : float
            concentration : float
        target : pandas.DataFrame
             dataframe to transform.
        Returns
        -------
        df : pandas.DataFrame
            transformed dataframe.
        """

        if target_df.empty:
            raise ValueError("Null dataframe.")
        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise ValueError("parameters must be [mean_rel, concentration].")

        mean_rel = float(parameters[0])
        concentration = float(parameters[1])

        if not (0.0 < mean_rel < 1.0):
            raise ValueError("mean_rel must be in (0, 1).")
        if concentration <= 0:
            raise ValueError("concentration must be > 0.")

        # Convert to alpha / beta
        alpha = mean_rel * concentration
        beta_param = (1.0 - mean_rel) * concentration

        return self.apply_beta_distribution_ab(
            parameters=[alpha, beta_param],
            target_df=target_df
        )

    def apply_beta_distributions_ab(self, alpha_interval=[], beta_interval=[],
                                demand_path="", directory="",
                                fond=0.0005):
        """
        Produce beta variations from demand_path dataframe with alpha beta parameters value in alpha_interval and beta interval and
        with a constant minimal baseline before and after the peak, while preserving the order of individuals. The variations are saved at                 directory.

        Parameters
        ----------
        alpha_interval : list
            Parameters for alpha range. pattern : np.arange(alpha_interval[0], alpha_interval[1], alpha_interval[2]).
        beta_interval : list
            Parameters for beta range. pattern : np.arange(beta_interval[0], beta_interval[1], beta_interval[2])
        demand_path : str
            Original dataframe path.
        directory : str
            Path to save variations
        fond_ : float
            constant minimal baseline.

        Returns
        -------
        None
        """
        if len(alpha_interval) != 3 or len(beta_interval) != 3:
            raise ValueError("Invalid alpha/beta intervals.")
    
        alpha_range = np.arange(alpha_interval[0], alpha_interval[1], alpha_interval[2])
        beta_range = np.arange(beta_interval[0], beta_interval[1], beta_interval[2])

        demand = pd.read_csv(demand_path, sep=';')
        os.makedirs(directory, exist_ok=True)

        for alpha in alpha_range:
            for beta_param in beta_range:
                logger.info(f"beta variation, alpha={alpha}, beta={beta_param}")
                buffer = demand.copy()
                var = self.apply_beta_distribution_ab(
                    parameters=[alpha, beta_param],
                    target_df=buffer,
                    fond_=fond
                )
                file_name = f"Temporal__Beta__{alpha:.2f}_{beta_param:.2f}_{fond}.csv"
                var.to_csv(f"{directory}/{file_name}", sep=';', index=False)


    def apply_beta_distributions_mc(self, mean_interval=[], concentration_interval=[],
                                demand_path="", directory="", name=""):
        """
        Produce beta variations from demand_path dataframe with mean concentratoon parameters value in mean_interval and concentration_interval and
        with a constant minimal baseline before and after the peak, while preserving the order of individuals. The variations are saved at                 directory.

        Parameters
        ----------
        mean_interval : list
            Parameters for mean range. pattern : np.arange(mean_interval[0], mean_interval[1], mean_interval[2]).
        concentration_interval : list
            Parameters for concentration range. pattern : np.arange(
                concentration_interval[0], concentration_interval[1], concentration_interval[2]
        )
        demand_path : str
            Original dataframe path.
        directory : str
            Path to save variations
        fond_ : float
            constant minimal baseline.

        Returns
        -------
        None
        """

        # Vérifications d'inputs
        if len(mean_interval) != 3:
            raise ValueError("Invalid mean_interval (must be [start, stop, step]).")
        if len(concentration_interval) != 3:
            raise ValueError("Invalid concentration_interval (must be [start, stop, step]).")
        if not demand_path or not directory or not name:
            raise ValueError("Invalid input paths or name.")

        # Création des grilles
        mean_range = np.arange(mean_interval[0], mean_interval[1], mean_interval[2])
        concentration_range = np.arange(
                concentration_interval[0], concentration_interval[1], concentration_interval[2]
        )

        # Lecture du fichier source
        demand = pd.read_csv(demand_path, sep=';')
        os.makedirs(directory, exist_ok=True)

        # Boucle sur toutes les combinaisons
        for mean_rel in mean_range:
            for concentration in concentration_range:
                logger.info(f"beta variation, mean_rel={mean_rel}, concentration={concentration}")
                buffer = demand.copy()
                var = self.apply_beta_distribution_mc(
                    parameters=[mean_rel, concentration],
                    target_df=buffer
                )
                # Nom du fichier
                file_name = f"Temporal__Beta__mean{mean_rel:.2f}__conc{concentration:.2f}.csv"
                var.to_csv(f"{directory}/{file_name}", sep=';')

        logger.info(
            f"Beta variations produced with mean_interval: {mean_interval}, "
            f"concentration_interval: {concentration_interval}, directory: {directory}"
        )









        


    from scipy.stats import beta

    def apply_beta_distribution_ab2(
        self,
        parameters=[],
        target_df=pd.DataFrame(),
        fond_before=0.05,
        fond_after=0.02
    ):
        """
        Déforme la distribution temporelle selon une loi Beta PURE.
        - Pas de cassure
        - Ordre conservé
        - Effectif total strictement conservé
        - Les fonds n'influencent PAS la CDF
        """

        import numpy as np
        import pandas as pd
        from scipy.stats import beta

        if target_df.empty:
            raise ValueError("Null dataframe.")
        if "DEPARTURE" not in target_df.columns:
            raise ValueError("DEPARTURE column not found.")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise ValueError("parameters must be [alpha, beta].")

        alpha, beta_param = parameters
        if alpha <= 0 or beta_param <= 0:
            raise ValueError("alpha and beta must be > 0.")

        df = target_df.copy()
        df = process_time_variable(df, "DEPARTURE")
        times = df["DEPARTURE"]

        t_min = times.min()
        t_max = times.max()
        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Invalid temporal interval.")

        n = len(df)
    
        # Axe continu
        x_grid = np.linspace(0, 1, 2000)

        # Beta PURE
        beta_density = beta.pdf(x_grid, alpha, beta_param)
        beta_density /= beta_density.sum()

        # CDF lisse
        cdf = np.cumsum(beta_density)
        cdf /= cdf[-1]

        # Mapping des individus (effectif conservé)
        quantiles = np.linspace(0, 1, n)
        x_samples_ordered = np.interp(quantiles, cdf, x_grid)

        # Conversion temporelle
        new_seconds = x_samples_ordered * total_seconds
        df["DEPARTURE"] = [t_min + pd.to_timedelta(s, unit="s") for s in new_seconds]

        df = restore_time_variable(df, "DEPARTURE")
        return df



        
    def apply_beta_distributions_ab2(self, alpha_interval=[], beta_interval=[],
                                demand_path="", directory="", name="",
                                fond_before=0.05, fond_after=0.02):

        if len(alpha_interval) != 3 or len(beta_interval) != 3:
            raise ValueError("Invalid alpha/beta intervals.")
    
        alpha_range = np.arange(alpha_interval[0], alpha_interval[1], alpha_interval[2])
        beta_range = np.arange(beta_interval[0], beta_interval[1], beta_interval[2])

        demand = pd.read_csv(demand_path, sep=';')
        os.makedirs(directory, exist_ok=True)

        for alpha in alpha_range:
            for beta_param in beta_range:
                logger.info(f"beta variation, alpha={alpha}, beta={beta_param}")
                buffer = demand.copy()
                var = self.apply_beta_distribution_ab2(
                    parameters=[alpha, beta_param],
                    target_df=buffer,
                    fond_before=fond_before,
                    fond_after=fond_after
                )
                file_name = f"Temporal__Beta__{alpha:.2f}_{beta_param:.2f}_{fond_before}_{fond_after}.csv"
                var.to_csv(f"{directory}/{file_name}", sep=';', index=False)

    def reshape_departures(self, df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    
        df_out = df.copy()
        col = parameters["column"]
    
        # Sauvegarder index original
        df_out["_original_index"] = df_out.index
    
        # Convertir horaire en secondes
        df_out[col] = pd.to_timedelta(df_out[col]).dt.total_seconds()
    
        T0 = pd.to_timedelta(parameters["T0"]).total_seconds()
        Td = pd.to_timedelta(parameters["Td"]).total_seconds()
        Tf = pd.to_timedelta(parameters["Tf"]).total_seconds()
        T1 = pd.to_timedelta(parameters["T1"]).total_seconds()
    
        p_before = parameters["p_before"]
        p_after = parameters["p_after"]
        alpha = parameters["alpha"]
    
        if p_before + p_after >= 1:
            raise ValueError("p_before + p_after must be < 1")
    
        p_main = 1 - p_before - p_after
    
        # Trier pour conserver l'ordre relatif
        df_out = df_out.sort_values(by=col)
    
        N = len(df_out)
        u = np.linspace(0, 1, N)
    
        total_before = Td - T0
        total_main = Tf - Td
        total_after = T1 - Tf
    
        new_seconds = np.zeros(N)
    
        # Zones vectorisées
        mask_before = u < p_before
        mask_main = (u >= p_before) & (u <= p_before + p_main)
        mask_after = u > p_before + p_main
    
        # Avant
        if p_before > 0:
            v = u[mask_before] / p_before
            new_seconds[mask_before] = v * total_before
    
        # Principal
        v = (u[mask_main] - p_before) / p_main
        new_seconds[mask_main] = total_before + (v ** alpha) * total_main
    
        # Après
        if p_after > 0:
            v = (u[mask_after] - (p_before + p_main)) / p_after
            new_seconds[mask_after] = total_before + total_main + v * total_after
    
        # Ajouter T0
        final_seconds = T0 + new_seconds
    
        # Conversion en H:MM:SS
        final_seconds = np.round(final_seconds).astype(int)
        hours = final_seconds // 3600
        minutes = (final_seconds % 3600) // 60
        seconds = final_seconds % 60
    
        df_out[col] = [
            f"{h:02d}:{m:02d}:{s:02d}"
            for h, m, s in zip(hours, minutes, seconds)
        ]
    
        # Remettre l'ordre original
        df_out = df_out.sort_values("_original_index").drop(columns="_original_index")
        
        return df_out





    # getters

    def get_original_demand_path(self):
        return self._original_demand_path

    def get_original_demand_df(self):
        return self._original_demand_df

    def get_types(self):
        return self._types

    def get_methods(self):
        return self._methods

    def get_laws(self):
        return self._laws








    