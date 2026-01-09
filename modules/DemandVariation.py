# dependencies

import os

import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import poisson

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
        Return
        ------
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
        Return
        ------
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
        Return
        ------
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
        Creates classification ratio variations on selected clusters.
    
        DIFFERENCE WITH PREVIOUS VERSION :
        ---------------------------------
        The target cluster is modified with its own ratio.
        All OTHER clusters are treated AS ONE SINGLE BLOCK:
            - A single compensatory ratio is applied to the union of all other clusters,
              instead of cluster-by-cluster compensation.
    
        The final file contains:
            - CLUSTER_RATIO: true final multiplicative ratio per cluster.
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
        Creates classification quantity variations on selected clusters.

        Quantity semantics:
        -------------------
        - target_quantity is an ABSOLUTE delta (integer, positive or negative)
        - final_target_size = original_target_size + target_quantity
        - all other clusters are treated as ONE SINGLE BLOCK
        - total_ratio controls the final total demand size
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

    