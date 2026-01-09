

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



class NaiveVariation():

    # Constructor
    
    def __init__(self, path=""):
        """
        Variation's constructor.
        loads the original demand.
    
        Parameters
        ----------
        path : string
            Path to the original demand.
        """

        # check
        if not path or path.strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        # assignment
        self._path = path # path to the demand file
        self._original_demand = pd.read_csv(path, sep=';') # original demand dataframe, the separator must be ';'

        # preprocessing
        self._original_demand["DEPARTURE"] = self._original_demand["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._original_demand["DEPARTURE"] = pd.to_datetime(self._original_demand["DEPARTURE"], format='%H:%M:%S')
        self._original_demand["DEPARTURE"] = self._original_demand["DEPARTURE"].dt.round("min")
        
        self._variation = self._original_demand # variation of the demand, al variation are done and saved on this dataframe
        self._variation_name = "" # the name of _variation. It describes what was done to obtain it

        # other attributes will be added for the clustering part

        logger.info("Variation initialized.")



    # Configuration's methods
    
    def change_demand(self, path=""):
        """
        Changes the original demand.
    
        Parameters
        ----------
        path : string
            Path to the original demand.
        """

        # check
        if not path or path.strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        # assignment
        self._path = path # path to the demand file
        self._original_demand = pd.read_csv(path, sep=';') # original demand dataframe, the separator must be ';'

        # preprocessing
        self._original_demand["DEPARTURE"] = self._original_demand["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._original_demand["DEPARTURE"] = pd.to_datetime(self._original_demand["DEPARTURE"], format='%H:%M:%S')
        self._original_demand["DEPARTURE"] = self._original_demand["DEPARTURE"].dt.round("min")
        
        self._variation = self._original_demand # variation of the demand, al variation are done and saved on this dataframe
        self._variation_name = "" # the name of _variation. It describes what was done to obtain it

        # other attributes will be added for the clustering part

        logger.info("demand file changed.")


    def reset(self):
        """
        Resets variation to the the original demand.
    
        No parameters
        """

        # check
        if self._original_demand.empty : 
            logger.info("_original_demand is empty.")
            raise ValueError("_original_demand is empty.")

        self._variation = self._original_demand
        self._variation_name = ""

        logger.info("_variation resets to _original_demand.")

    
    def save_variation(self, path, name=""):
        """
        saves variation in path with name
    
        Parameters
        ----------
        path : string
            Path to save.
        name : string
            name of the to be saved file
        """

        # check
        if not path or path.strip() == "": 
            logger.error("Invalid or null path")
            raise ValueError("Invalid or null path")
        if self._variation.empty: 
            logger.error("_variation is empty.")
            raise ValueError("_variation is empty.")
        if not self._variation_name or self._variation_name.strip() == "":  
            logger.error("_variation_name is null : no modifications were done.")
            raise ValueError("_variation_name is null : no modifications were done.")

        if not name or name.strip() == "" : name = self._variation_name
        
        full_path = os.path.join(path,f"{name}.csv")

        self._variation["DEPARTURE"] = self._variation["DEPARTURE"].dt.strftime("%H:%M:%S")
        
        self._variation.to_csv(full_path, sep=';', index=False)

        self._variation["DEPARTURE"] = self._variation["DEPARTURE"].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
        self._variation["DEPARTURE"] = pd.to_datetime(self._variation["DEPARTURE"], format='%H:%M:%S')
        self._variation["DEPARTURE"] = self._variation["DEPARTURE"].dt.round("min")

        logger.info(f"{name} saved in {path}.")


    def help(self):
        print("help")


    
    # variation

    # random sampling

    def uniform_weightedsampling_variations(self, ratio, period=None, random_state=42):
        """
        Apply a variation to the `_variation` DataFrame using uniform random sampling
        (equal probability for all records) within a specified time period.
        The total number of records in the selected period is modified according to
        the given ratio (duplication or deletion).

        Parameters
        ----------
        ratio : float
            Scaling factor for the total number of records in the selected period.
            (e.g., 1.2 = +20%, 0.8 = -20%)
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
            Example: ("16:00", "18:00") or (time(16,0), time(18,0)).
            If None, applies to the entire observed period.
        random_state : int, optional
            Random seed for reproducibility.

        Notes
        -----
        - This method uses **uniform random sampling**, meaning all records in the 
          selected period have equal chance of being duplicated or deleted.
        """

        # basic checks
        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if ratio == 0:
            raise ValueError("null ratio, invalid variation.")
        if len(self._variation_name) > 0:
            raise ValueError(
                "Apply a variation on an already changed demand is not allowed. "
                "Use reset() if you want to return to the original demand."
            )

        def time_to_float(t):
            """Convert a datetime.time to decimal hours."""
            return t.hour + t.minute / 60 + t.second / 3600

        # prepare time columns
        self._variation["TIME_ONLY"] = self._variation["DEPARTURE"].dt.time
        self._variation["TIME_FLOAT"] = self._variation["TIME_ONLY"].apply(time_to_float)

        # define the time period 
        if period is None:
            start_float = self._variation["TIME_FLOAT"].min()
            end_float = self._variation["TIME_FLOAT"].max()
        else:
            if len(period) != 2:
                raise ValueError("period format is invalid (should be 2 elements).")

            def parse_time(t):
                if isinstance(t, str):
                    d = dt.datetime.strptime(t, "%H:%M")
                    return d.hour + d.minute / 60
                elif isinstance(t, time):
                    return t.hour + t.minute / 60 + t.second / 3600
                else:
                    raise ValueError("period elements must be str HH:MM or datetime.time")

            start_float, end_float = parse_time(period[0]), parse_time(period[1])

        # filter data within the period 
        buffer = self._variation[
            (self._variation["TIME_FLOAT"] >= start_float)
            & (self._variation["TIME_FLOAT"] <= end_float)
        ].copy()

        if buffer.empty:
            logger.error("No data for the given time period.")
            raise ValueError("No data for the given time period.")

        # compute total change 
        total_original = len(self._variation)
        total_target = int(total_original * ratio)
        delta = total_target - total_original

        logger.info(
            f"Total records: {total_original} → {total_target} "
            f"({'+' + str(delta) if delta > 0 else delta} rows)"
        )
    
        # apply uniform random variation
        if delta == 0:
            logger.info("No modification (ratio = 1).")
            return

        if delta > 0:
            # uniform duplication
            sampled = buffer.sample(n=delta, replace=True, random_state=random_state)
            max_id = self._variation["ID"].max()
            sampled = sampled.copy()
            sampled["ID"] = range(max_id + 1, max_id + 1 + len(sampled))
            df_new = pd.concat([self._variation, sampled], ignore_index=True)
            df_new = df_new.sort_values(by="DEPARTURE").reset_index(drop=True)
            operation = f"+{delta} uniform duplications"
        else:
            # uniform deletion
            n_remove = -delta
            if n_remove >= len(buffer):
                raise ValueError("No modification, ratio too low.")
            drop_idx = buffer.sample(n=n_remove, replace=False, random_state=random_state).index
            df_new = self._variation.drop(index=drop_idx).reset_index(drop=True)
            operation = f"-{n_remove} uniform deletions"

        # update the DataFrame and naming 
        str_period = (
            f"{int(start_float):02d}h{int((start_float % 1) * 60):02d}-"
            f"{int(end_float):02d}h{int((end_float % 1) * 60):02d}"
        )
        new_name = f"WeightedSampling-Uniform_None_{ratio:.2f}_{str_period}__None"
        df_new = df_new.drop(columns=["TIME_ONLY","TIME_FLOAT"])
        if "SERVICE" in df_new.columns : df_new = df_new.drop(columns=["SERVICE"])
        self._variation = df_new.copy()
        self._variation_name = new_name

        logger.info(
            f"_variation modified using Uniform random sampling with ratio={ratio}, "
            f"period={str_period}. ({operation})"
        )


    def normal_weightedsampling_variations(self, ratio, mean=None, std=None, period=None, random_state=42):
        """
        Apply a variation to the `_variation` DataFrame using random sampling
        weighted by a Normal (Gaussian) distribution, within a specified time period.
        The total number of records in the selected period is modified according to
        the given ratio (duplication or deletion).

        Parameters
        ----------
        ratio : float
            Scaling factor for the total number of records in the selected period.
            (e.g., 1.2 = +20%, 0.8 = -20%)
        mean : str | datetime.time | None, optional
            Mean (center) of the Normal distribution in hours. 
            If None, defaults to the midpoint of the selected period.
            Example: "17:00" or time(17, 0)
        std : float | None, optional
            Standard deviation (in hours). Controls how wide the weighting is
            around the mean. If None, defaults to one quarter of the period width.
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
            Example: ("16:00", "18:00") or (time(16,0), time(18,0)).
            If None, applies to the entire observed period.
        random_state : int, optional
            Random seed for reproducibility.

        Notes
        -----
        - This method uses random sampling **weighted by a Normal distribution**.
            - Duplications or deletions are applied only to the records within the period.
        """

        # basic checks
        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if ratio == 0:
            raise ValueError("null ratio, invalid variation.")
        if len(self._variation_name) > 0:
            raise ValueError(
                "Apply a variation on an already changed demand is not allowed. "
                "Use reset() if you want to return to the original demand."
            )

        def time_to_float(t):
            """Convert a datetime.time to decimal hours."""
            return t.hour + t.minute / 60 + t.second / 3600

        # prepare time columns
        self._variation["TIME_ONLY"] = self._variation["DEPARTURE"].dt.time
        self._variation["TIME_FLOAT"] = self._variation["TIME_ONLY"].apply(time_to_float)

        # define the time period 
        if period is None:
            start_float = self._variation["TIME_FLOAT"].min()
            end_float = self._variation["TIME_FLOAT"].max()
        else:
            if len(period) != 2:
                raise ValueError("period format is invalid (should be 2 elements).")

            def parse_time(t):
                if isinstance(t, str):
                    d = dt.datetime.strptime(t, "%H:%M")
                    return d.hour + d.minute / 60
                elif isinstance(t, time):
                    return t.hour + t.minute / 60 + t.second / 3600
                else:
                    raise ValueError("period elements must be str HH:MM or datetime.time")

            start_float, end_float = parse_time(period[0]), parse_time(period[1])

        # filter data within the period 
        buffer = self._variation[
            (self._variation["TIME_FLOAT"] >= start_float)
            & (self._variation["TIME_FLOAT"] <= end_float)
        ].copy()

        if buffer.empty:
            logger.error("No data for the given time period.")
            raise ValueError("No data for the given time period.")

        # compute hours and normal weights 
        h_start, h_end = start_float, end_float
        if mean is None:
            mu = (h_start + h_end) / 2
        else:
            if isinstance(mean, str):
                d = dt.datetime.strptime(mean, "%H:%M")
                mu = d.hour + d.minute / 60
            elif isinstance(mean, time):
                mu = mean.hour + mean.minute / 60 + mean.second / 3600
            else:
                raise ValueError("mean must be a string HH:MM, datetime.time, or None")

        if std is None:
            sigma = (h_end - h_start) / 4
        elif not isinstance(std, (int, float)):
            raise ValueError("std must be a numeric value (float or int).")
        else:
            sigma = std

        x = buffer["TIME_FLOAT"].values
        probs = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        probs /= probs.sum()  # normalize probabilities

        # compute total change 
        total_original = len(self._variation)
        total_target = int(total_original * ratio)
        delta = total_target - total_original

        logger.info(
            f"Total records: {total_original} → {total_target} "
            f"({'+' + str(delta) if delta > 0 else delta} rows)"
        )

        # apply the variation (Normal-weighted random sampling)
        if delta == 0:
            logger.info("No modification (ratio = 1).")
            return

        if delta > 0:
            # weighted duplication
            sampled = buffer.sample(
                n=delta, replace=True, weights=probs, random_state=random_state
            )
            max_id = self._variation["ID"].max()
            sampled = sampled.copy()
            sampled["ID"] = range(max_id + 1, max_id + 1 + len(sampled))
            df_new = pd.concat([self._variation, sampled], ignore_index=True)
            df_new = df_new.sort_values(by="DEPARTURE").reset_index(drop=True)
            operation = f"+{delta} normal-weighted duplications"
        else:
            # weighted deletion
            n_remove = -delta
            if n_remove >= len(buffer):
                raise ValueError("No modification, ratio too low.")
            drop_idx = buffer.sample(
                n=n_remove, replace=False, weights=probs, random_state=random_state
            ).index
            df_new = self._variation.drop(index=drop_idx).reset_index(drop=True)
            operation = f"-{n_remove} normal-weighted deletions"

        # update the DataFrame and naming 
        str_period = (
            f"{int(start_float):02d}h{int((start_float % 1) * 60):02d}-"
            f"{int(end_float):02d}h{int((end_float % 1) * 60):02d}"
        )
        new_name = f"WeightedSampling-Normal_{mu:.2f}-{sigma:.2f}_{ratio:.2f}_{str_period}__None"
        df_new = df_new.drop(columns=["TIME_ONLY","TIME_FLOAT"])
        if "SERVICE" in df_new.columns : df_new = df_new.drop(columns=["SERVICE"])
        self._variation = df_new.copy()
        self._variation_name = new_name

        logger.info(
            f"_variation modified using Normal-weighted random sampling with ratio={ratio:.2f}, "
            f"mean={mu:.2f}h, std={sigma:.2f}h, period={str_period}. ({operation})"
        )


    def poisson_weightedsampling_variations(self, ratio, lam=None, period=None, peak_time=None, random_state=42):
        """
        Apply a variation to the `_variation` DataFrame using Poisson-distributed 
        random sampling within a specified time period.
    
        Parameters
        ----------
        ratio : float
            Scaling factor for the total number of records in the selected period.
            (e.g., 1.2 = +20%, 0.8 = -20%)
        lam : float, optional
            Expected value (λ) of the Poisson distribution in minutes.
            If None, defaults to the mean of the relative time in minutes.
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
        peak_time : float, optional
            Desired peak hour in decimal format (e.g., 16.75 for 16:45). 
            Overrides lam if provided.
        random_state : int, optional
            Random seed for reproducibility.
        """

        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if ratio == 0:
            raise ValueError("null ratio, invalid variation.")
        if len(self._variation_name) > 0:
            raise ValueError(
                "Apply a Poisson variation on an already changed demand is not allowed. "
                "Use reset() if you want to return to the original demand."
            )

        np.random.seed(random_state)

        def time_to_float(t):
            return t.hour + t.minute / 60 + t.second / 3600

        self._variation["TIME_ONLY"] = self._variation["DEPARTURE"].dt.time
        self._variation["TIME_FLOAT"] = self._variation["TIME_ONLY"].apply(time_to_float)

        # define period
        if period is None:
            start_float = self._variation["TIME_FLOAT"].min()
            end_float = self._variation["TIME_FLOAT"].max()
        else:
            if len(period) != 2:
                raise ValueError("period format is invalid (should be 2 elements).")

            def parse_time(t):
                if isinstance(t, str):
                    d = dt.datetime.strptime(t, "%H:%M")
                    return d.hour + d.minute / 60
                elif isinstance(t, dt.time):
                    return t.hour + t.minute / 60 + t.second / 3600
                else:
                    raise ValueError("period elements must be str HH:MM or datetime.time")

            start_float, end_float = parse_time(period[0]), parse_time(period[1])

        buffer = self._variation[
            (self._variation["TIME_FLOAT"] >= start_float) & 
            (self._variation["TIME_FLOAT"] <= end_float)
        ].copy()
    
        if buffer.empty:
            raise ValueError("No data for the given time period.")

        # relative time in minutes 
        rel_minutes = np.round((buffer["TIME_FLOAT"] - start_float) * 60).astype(int)

        # determine lambda 
        if peak_time is not None:
            lam = int((peak_time - start_float) * 60)
        elif lam is None:
            lam = int(rel_minutes.mean())

        if lam <= 0:
            raise ValueError("λ must be positive.")

        # poisson probability 
        density = poisson.pmf(rel_minutes, mu=lam)
        if np.all(density == 0):
            raise ValueError("All Poisson densities are zero — check lam/peak_time and period.")
        density /= density.sum()

        # compute delta based on ratio 
        total_original = len(self._variation)
        total_target = int(total_original * ratio)
        delta = total_target - total_original
    
        if delta == 0:
            return
        
        # apply variation 
        if delta > 0:
            sampled = buffer.sample(n=delta, replace=True, weights=density, random_state=random_state)
            max_id = self._variation["ID"].max()
            sampled = sampled.copy()
            sampled["ID"] = range(max_id + 1, max_id + 1 + len(sampled))
            df_new = pd.concat([self._variation, sampled], ignore_index=True)
            df_new = df_new.sort_values(by="DEPARTURE").reset_index(drop=True)
        else:
            n_remove = -delta
            if n_remove >= len(buffer):
                raise ValueError("No modification, ratio too low.")
            drop_idx = buffer.sample(n=n_remove, replace=False, weights=density, random_state=random_state).index
            df_new = self._variation.drop(index=drop_idx).reset_index(drop=True)

        # update DataFrame and name 
        str_period = f"{int(start_float):02d}h{int((start_float%1)*60):02d}-{int(end_float):02d}h{int((end_float%1)*60):02d}"
        peak_time_float = start_float + lam / 60
        peak_hours = int(peak_time_float)
        peak_minutes = int(round((peak_time_float % 1) * 60))
        peak_str = f"{peak_hours:02d}h{peak_minutes:02d}"
        new_name = f"WeightedSampling-Poisson_{peak_str}_{ratio:.2f}_{str_period}__None"
        df_new = df_new.drop(columns=["TIME_ONLY","TIME_FLOAT"])
        if "SERVICE" in df_new.columns : df_new = df_new.drop(columns=["SERVICE"])
        self._variation = df_new.copy()
        self._variation_name = new_name



    # variation by probability density

    def uniform_density_variation(self, ratio, period=None, random_state=42):
        """
        Apply a variation to the `_variation` DataFrame based on a Uniform
        probability density, within a specified time period.

        Parameters
        ----------
        ratio : float
            Scaling factor for the total number of records in the selected period.
            (e.g., 1.2 = +20%, 0.8 = -20%)
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
            Example: ("16:00", "18:00") or (time(16,0), time(18,0))
        random_state : int, optional
            Random seed.
        """

        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if ratio == 0:
            raise ValueError("null ratio, invalid variation.")
        if len(self._variation_name) > 0:
            raise ValueError(
                "Apply a density variation on an already changed demand is not allowed. "
                "Use reset() if you want to return to the original demand."
            )

        def time_to_float(t):
            return t.hour + t.minute / 60 + t.second / 3600

        self._variation["TIME_ONLY"] = self._variation["DEPARTURE"].dt.time
        self._variation["TIME_FLOAT"] = self._variation["TIME_ONLY"].apply(time_to_float)

        # define the time period
        if period is None:
            start_float = self._variation["TIME_FLOAT"].min()
            end_float = self._variation["TIME_FLOAT"].max()
        else:
            if len(period) != 2:
                raise ValueError("period format is invalid (should be 2 elements).")

            def parse_time(t):
                if isinstance(t, str):
                    d = dt.datetime.strptime(t, "%H:%M")
                    return d.hour + d.minute / 60
                elif isinstance(t, time):
                    return t.hour + t.minute / 60 + t.second / 3600
                else:
                    raise ValueError("period elements must be str HH:MM or datetime.time")

            start_float, end_float = parse_time(period[0]), parse_time(period[1])

        # filter data within the period 
        buffer = self._variation[
            (self._variation["TIME_FLOAT"] >= start_float)
            & (self._variation["TIME_FLOAT"] <= end_float)
        ].copy()

        if buffer.empty:
            logger.error("No data for the given time period.")
            raise ValueError("No data for the given time period.")

        # compute uniform density (equal probability)
        density = np.ones(len(buffer)) / len(buffer)

        # compute total change based on ratio 
        total_original = len(self._variation)
        total_target = int(total_original * ratio)
        delta = total_target - total_original

        logger.info(
            f"Total records: {total_original} → {total_target} "
            f"({'+' + str(delta) if delta > 0 else delta} rows)"
        )

        # apply the variation
        if delta == 0:
            logger.info("No modification (ratio = 1).")
            return

        if delta > 0:
            # weighted duplication (all equal weights)
            sampled = buffer.sample(n=delta, replace=True, weights=density, random_state=random_state)
            max_id = self._variation["ID"].max()
            sampled = sampled.copy()
            sampled["ID"] = range(max_id + 1, max_id + 1 + len(sampled))
            df_new = pd.concat([self._variation, sampled], ignore_index=True)
            df_new = df_new.sort_values(by="DEPARTURE").reset_index(drop=True)
        else:
            # weighted deletion (all equal weights)
            n_remove = -delta
            if n_remove >= len(buffer):
                raise ValueError("No modification, ratio too low.")
            drop_idx = buffer.sample(n=n_remove, replace=False, weights=density, random_state=random_state).index
            df_new = self._variation.drop(index=drop_idx).reset_index(drop=True)

        # update DataFrame and naming
        str_period = (
            f"{int(start_float):02d}h{int((start_float % 1) * 60):02d}-"
            f"{int(end_float):02d}h{int((end_float % 1) * 60):02d}"
        )
        new_name = f"Density-Uniform_None_{ratio:.2f}_{str_period}__None"
        df_new = df_new.drop(columns=["TIME_ONLY","TIME_FLOAT"])
        if "SERVICE" in df_new.columns : df_new = df_new.drop(columns=["SERVICE"])
        self._variation = df_new.copy()
        self._variation_name = new_name
    
        logger.info(
            f"_variation modified using Uniform law with ratio={ratio:.2f}, period={str_period}."
        )


    def normal_density_variation(self, ratio, mean=None, std=None, period=None, random_state=42):
        """
        Apply a variation to the `_variation` DataFrame based on a Normal (Gaussian)
        probability density, within a specified time period.

        Parameters
        ----------
        ratio : float
            Scaling factor for the total number of records in the selected period.
            (e.g., 1.2 = +20%, 0.8 = -20%)
        mean : str | datetime.time | None, optional
            Center of the normal distribution (in hours). If None, defaults to
            the mean of the observed times within the period.
            Example: "17:00" or time(17, 0)
        std : float | None, optional
            Standard deviation of the normal distribution (in hours).
            Controls how wide the weighting is around the mean.
            If None, defaults to one quarter of the total period width.
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
            Example: ("16:00", "18:00") or (time(16,0), time(18,0))
         random_state : int, optionnal
            random seed.
        """

        # basic checks 
        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if ratio == 0:
            raise ValueError("null ratio, invalid variation.")
        if len(self._variation_name) > 0:
            raise ValueError("Apply a density variation on an already changed demand is not allowed, use reset if you want to return to the original demand.")

        def time_to_float(t):
            return t.hour + t.minute / 60 + t.second / 3600

        self._variation["TIME_ONLY"] = self._variation["DEPARTURE"].dt.time
        self._variation["TIME_FLOAT"] = self._variation["TIME_ONLY"].apply(time_to_float)

        # define the time period
        if period is None:
            start_float = self._variation["TIME_FLOAT"].min()
            end_float = self._variation["TIME_FLOAT"].max()
        else:
            if len(period) != 2:
                raise ValueError("period format is invalid (should be 2 elements).")

            def parse_time(t):
                if isinstance(t, str):
                    d = dt.datetime.strptime(t, "%H:%M")
                    return d.hour + d.minute / 60
                elif isinstance(t, time):
                    return t.hour + t.minute / 60 + t.second / 3600
                else:
                    raise ValueError("period elements must be str HH:MM or datetime.time")

            start_float, end_float = parse_time(period[0]), parse_time(period[1])

        # filter data within the period 
        buffer = self._variation[
            (self._variation["TIME_FLOAT"] >= start_float)
            & (self._variation["TIME_FLOAT"] <= end_float)
        ].copy()

        if buffer.empty:
            logger.error("No data for the given time period.")
            raise ValueError("No data for the given time period.")

        # compute relative hours (in decimal hours) 
        hours = buffer["TIME_FLOAT"] - start_float

        # determine mean (center) 
        if mean is None:
            mean_rel = hours.mean()
        else:
            if isinstance(mean, str):
                d = dt.datetime.strptime(mean, "%H:%M")
                mean_rel = d.hour + d.minute / 60 - start_float
            elif isinstance(mean, time):
                mean_rel = mean.hour + mean.minute / 60 + mean.second / 3600 - start_float
            else:
                raise ValueError("mean must be a string HH:MM, datetime.time, or None")

        # determine standard deviation 
        if std is None:
            std = (end_float - start_float) / 4
        elif not isinstance(std, (int, float)):
            raise ValueError("std must be a numeric value (float or int).")

        # compute Normal (Gaussian) density 
        density = norm.pdf(hours, loc=mean_rel, scale=std)
        density /= density.sum()  # normalize to sum = 1

        # compute total change based on ratio 
        total_original = len(self._variation)
        total_target = int(total_original * ratio)
        delta = total_target - total_original

        logger.info(
            f"Total records: {total_original} → {total_target} "
            f"({'+' + str(delta) if delta > 0 else delta} rows)"
        )

        # apply the variation 
        if delta == 0:
            logger.info("No modification (ratio = 1).")
            return

        if delta > 0:
            # weighted duplication
            sampled = buffer.sample(n=delta, replace=True, weights=density, random_state=random_state)
            max_id = self._variation["ID"].max()
            sampled = sampled.copy()
            sampled["ID"] = range(max_id + 1, max_id + 1 + len(sampled))
            df_new = pd.concat([self._variation, sampled], ignore_index=True)
            df_new = df_new.sort_values(by="DEPARTURE").reset_index(drop=True)
        else:
            # weighted deletion
            n_remove = -delta
            if n_remove >= len(buffer):
                raise ValueError("No modification, ratio too low.")
            drop_idx = buffer.sample(
                n=n_remove, replace=False, weights=density, random_state=42
            ).index
            df_new = self._variation.drop(index=drop_idx).reset_index(drop=True)

        # update DataFrame and naming
        str_period = (
            f"{int(start_float):02d}h{int((start_float % 1) * 60):02d}-"
            f"{int(end_float):02d}h{int((end_float % 1) * 60):02d}"
        )
        new_name = f"Density-Normal_{mean_rel:.2f}-{std:.2f}_{ratio:.2f}_{str_period}__None"
        df_new = df_new.drop(columns=["TIME_ONLY","TIME_FLOAT"])
        if "SERVICE" in df_new.columns : df_new = df_new.drop(columns=["SERVICE"])
        self._variation = df_new.copy()
        self._variation_name = new_name

        logger.info(
                f"_variation modified using Normal law with ratio={ratio}, "
                f"mean={mean_rel:.2f}h, std={std:.2f}h, period={str_period}."
            )


    def poisson_density_variation(self, ratio, lam=None, period=None, peak_time=None, random_state=42):
        """
        Apply a variation to the `_variation` DataFrame based on a Poisson
        distribution within a specified time period. Works on minutes for finer granularity.
    
        Parameters
        ----------
        ratio : float
            Scaling factor for the total number of records in the selected period.
            (e.g., 1.2 = +20%, 0.8 = -20%)
        lam : float, optional
            Lambda (λ) of the Poisson distribution in minutes.
            If None, defaults to the mean of the relative time in minutes.
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
            Example: ("16:00", "18:00") or (time(16,0), time(18,0))
        peak_time : float, optional
            Desired peak hour in decimal format (e.g., 16.75 for 16:45). 
            Overrides lam if provided.
        random_state : int
            Random seed for reproducibility.
        """


        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if ratio == 0:
            raise ValueError("null ratio, invalid variation.")
        if len(self._variation_name) > 0:
            raise ValueError("Apply a density variation on an already changed demand is not allowed, use reset if you want to return to the original demand.")

        def time_to_float(t):
            return t.hour + t.minute / 60 + t.second / 3600

        self._variation["TIME_ONLY"] = self._variation["DEPARTURE"].dt.time
        self._variation["TIME_FLOAT"] = self._variation["TIME_ONLY"].apply(time_to_float)

        # define period 
        if period is None:
            start_float = self._variation["TIME_FLOAT"].min()
            end_float = self._variation["TIME_FLOAT"].max()
        else:
            if len(period) != 2:
                raise ValueError("period format is invalid (should be 2 elements).")
    
            def parse_time(t):
                if isinstance(t, str):
                    d = dt.datetime.strptime(t, "%H:%M")
                    return d.hour + d.minute / 60
                elif isinstance(t, dt.time):
                    return t.hour + t.minute / 60 + t.second / 3600
                else:
                    raise ValueError("period elements must be str HH:MM or datetime.time")

            start_float, end_float = parse_time(period[0]), parse_time(period[1])

        # filter data within period
        buffer = self._variation[
            (self._variation["TIME_FLOAT"] >= start_float) & 
            (self._variation["TIME_FLOAT"] <= end_float)
        ].copy()
    
        if buffer.empty:
            logger.error("No data for the given time period.")
            raise ValueError("No data for the given time period.")

        # compute relative time in minutes
        rel_minutes = np.round((buffer["TIME_FLOAT"] - start_float) * 60).astype(int)

        # determine lambda
        if peak_time is not None:
            lam = int((peak_time - start_float) * 60)  # λ in minutes for desired peak
        elif lam is None:
            lam = int(rel_minutes.mean())  # default to mean in minutes

        # poisson probability
        density = poisson.pmf(rel_minutes, mu=lam)
        if np.all(density == 0):
            raise ValueError("All Poisson densities are zero — check lam/peak_time and period.")
        density /= density.sum()

        # compute delta based on ratio 
        total_original = len(self._variation)
        total_target = int(total_original * ratio)
        delta = total_target - total_original

        logger.info(f"Total: {total_original} → {total_target} ({'+'+str(delta) if delta>0 else delta} rows)")

        if delta == 0:
            logger.info("No modification (ratio = 1).")
            return

        # --- Apply variation ---
        if delta > 0:
            sampled = buffer.sample(n=delta, replace=True, weights=density, random_state=random_state)
            max_id = self._variation["ID"].max()
            sampled = sampled.copy()
            sampled["ID"] = range(max_id + 1, max_id + 1 + len(sampled))
            df_new = pd.concat([self._variation, sampled], ignore_index=True)
            df_new = df_new.sort_values(by="DEPARTURE").reset_index(drop=True)
        else:
            n_remove = -delta
            if n_remove >= len(buffer):
                raise ValueError("No modification, ratio too low.")
            drop_idx = buffer.sample(n=n_remove, replace=False, weights=density, random_state=random_state).index
            df_new = self._variation.drop(index=drop_idx).reset_index(drop=True)

        # --- Update DataFrame and name ---
        str_period = f"{int(start_float):02d}h{int((start_float%1)*60):02d}-{int(end_float):02d}h{int((end_float%1)*60):02d}"
        peak_time_float = start_float + lam / 60  # heures décimales
        peak_hours = int(peak_time_float)
        peak_minutes = int(round((peak_time_float % 1) * 60))
        peak_str = f"{peak_hours:02d}h{peak_minutes:02d}"
        new_name = f"Density-Poisson_{peak_str}_{ratio:.2f}_{str_period}__None"
        df_new = df_new.drop(columns=["TIME_ONLY","TIME_FLOAT"])
        if "SERVICE" in df_new.columns : df_new = df_new.drop(columns=["SERVICE"])
        self._variation = df_new.copy()
        self._variation_name = new_name

        logger.info(f"_variation modified using Poisson law with ratio={ratio}, λ={lam}min, period={str_period}.")



    # events

    # strike

    def publictransport_strike_variation(self, strike_list, period = None):
        """
        Apply a public transport strike on the _variation dataframe.
        In reality, it only notifies the change by formatting the name, the real change becomes effective
        during the simulation.
    
        Parameters
        ----------
        strike_list : list
            list of public transport in strike in upper case
        period : tuple(str|datetime.time, str|datetime.time), optional
            Start and end times of the period on which to apply the variation.
            Example: ("16:00", "18:00") or (time(16,0), time(18,0)).
        """

        # check
        if self._variation.empty:
            logger.error("_variation is empty.")
            raise ValueError("_variation is empty.")
        if len(strike_list) == 0 :
            logger.error("strike list is empty.")
            raise ValueError("strike list is empty.")
        if len(self._variation_name) > 0 and self._variation_name[-4:] != "None":
            logger.error("An event has already been applied to the current variation, resets if you want to apply this one.")
            raise ValueError("An event has already been applied to the current variation.")

        # available means
        available_means = {"CAR", "METRO", "BUS","TRAM"}
        strike_set = set(strike_list)
        restricted_means = available_means - strike_set
        restricted_means = list(restricted_means)
        restricted_means = list(map(str.upper, restricted_means))

        available_means = list(map(str.upper, available_means))
        available_means = np.array(list(available_means))
        available_means = " ".join(map(str, available_means))

        # if no time period, takes the whole time period of the original demand
        if not period : 
            period = []
            period.append(self._original_demand["DEPARTURE"].min().time())
            period.append(self._original_demand["DEPARTURE"].max().time())

        if len(self._variation.loc[(period[0] <= self._variation["DEPARTURE"].dt.time) & (self._variation["DEPARTURE"].dt.time <= period[1])]) == 0:
            logger.error("Invalid time period.")
            raise ValueError("Invalid time period.")

        # name
        
        strike_str = "Strike-PublicTransport_"
        strike_str += "-".join(strike_list)
        strike_str += f"_{period[0].hour:02d}h{period[0].minute:02d}-{period[1].hour:02d}h{period[1].minute:02d}"

            
        if len(self._variation_name) == 0 : self._variation_name = "None__" + strike_str
        else : 
            self._variation_name =  self._variation_name[:-4]
            self._variation_name += strike_str

        logger.info(f"strike applied on {strike_list}")
        
        

    
        

    def apply_events(self, events=[]):

       # check
        if self._variation.empty:
            raise ValueError("_variation is empty.")
        if len(events) == 0:
            raise ValueError("No events.")
            

        
            
    # getters

    def get_path(self):
        # check
        if not self._path or self._path.strip() == "": raise ValueError("Invalid or null path.")
            
        return self._path

        
    def get_original_demand(self):
        # check
        if self._original_demand.empty: raise ValueError("_original_demand is empty.")
            
        return self._original_demand


    def get_variation(self):
        # check
        if self._variation.empty: logger.info("_variation is empty.")
            
        return self._variation


    def get_variation_name(self):
        # check
        if  not self._variation_name or self._variation_name.strip() == "" : raise ValueError("_variation_name is empty.")

        return self._variation_name



    # display 

    def display_variation(self):
        # check
        if self._variation.empty: logger.info("_variation is empty.")

        buffer = self._variation.groupby("DEPARTURE").count().reset_index()

        fig, ax = plt.subplots(1,1,figsize=(12,5))
        ax.bar(buffer["DEPARTURE"],buffer["ID"], width=pd.Timedelta(seconds=60),align='center',edgecolor='black')
        ax.set_title(self._variation_name)
        ax.set_xlabel("temps")
        ax.set_ylabel("nombre de départs")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=20)) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(True)

        plt.tight_layout()
        plt.show()
        


    