# tout tester, fonction de variations des parametres pour chaqye type de densité


# dependencies
import os

import json

import random

import numpy as np
import pandas as pd

import logging

# log
logger = logging.getLogger(__name__)


# helper functions

def generate_indices(df_group, k, law="Uniform", parameters=None, with_replacement=False):
    """
    Generates row indices according to a probability law.
    """
    n = len(df_group)
    if n == 0:
        return np.array([])

    if law == "Uniform":
        return np.random.choice(n, size=k, replace=with_replacement)

    elif law == "Normal":
        mu, sigma = parameters if parameters else (n / 2, n / 6)
        idx = np.clip(np.random.normal(mu, sigma, k).astype(int), 0, n - 1)
        return idx

    elif law == "Poisson":
        lam = parameters if parameters else n / 2
        idx = np.clip(np.random.poisson(lam, k), 0, n - 1)
        return idx

    else:
        raise ValueError(f"Unsupported law '{law}'")


def generate_weights(n, law="Uniform", parameters=None):
    """
    Generates normalized sampling weights according to a probability law.
    """
    if law == "Uniform":
        weights = np.ones(n)
    elif law == "Normal":
        mu, sigma = parameters if parameters else (n / 2, n / 6)
        x = np.arange(n)
        weights = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    elif law == "Poisson":
        lam = parameters if parameters else n / 2
        x = np.arange(n)
        weights = np.exp(-lam) * (lam ** x) / np.maximum(1, np.array([np.math.factorial(int(i)) for i in x]))
    else:
        raise ValueError(f"Unsupported law '{law}'")

    weights = weights / np.sum(weights)
    return weights


def apply_density_variation(df_group, ratio, law="Uniform", parameters=None):
    """
    Modifies the group size (density) according to a target ratio.
    """
    n = len(df_group)
    target_n = max(1, int(n * ratio))

    if target_n == n:
        return df_group.copy()
    elif target_n < n:
        indices = generate_indices(df_group, target_n, law, parameters)
        return df_group.iloc[indices]
    else:
        additional_n = target_n - n
        indices = generate_indices(df_group, additional_n, law, parameters, with_replacement=True)
        duplicated = df_group.iloc[indices]
        return pd.concat([df_group, duplicated], ignore_index=True)


def apply_weighted_sampling(df_group, ratio, law="Uniform", parameters=None):
    """
    Applies weighted random sampling based on a distribution law.
    Automatically chooses replace=False for under-sampling, replace=True for over-sampling.

    Parameters
    ----------
    df_group : pd.DataFrame
        The DataFrame group to sample from.
    ratio : float
        Proportion of rows to sample (e.g., 0.5 means 50% of n).
    law : str
        Probability law for weighting: "Uniform", "Normal", "Poisson".
    parameters : tuple, float, or None
        Parameters for the distribution law.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame with weighted selection.
    """
    n = len(df_group)
    if n == 0:
        return df_group.copy()  # nothing to sample

    target_n = max(1, int(n * ratio))
    weights = generate_weights(n, law, parameters)

    if len(weights) != n:
        raise ValueError("Length of weights does not match number of rows in df_group")

    # Automatically choose replace
    replace = target_n > n

    sampled_indices = np.random.choice(df_group.index, size=target_n, replace=replace, p=weights)
    return df_group.loc[sampled_indices].copy()

def apply_density_variation_quantity(df_group, n_target, law="Uniform", parameters=None):
    """
    Modifies group size according to a target *absolute* quantity (n_target).
    """
    n = len(df_group)
    n_target = max(1, int(n_target))  # safety

    if n_target == n:
        return df_group.copy()

    elif n_target < n:
        # Under-sampling
        indices = generate_indices(df_group, n_target, law, parameters)
        return df_group.iloc[indices]

    else:
        # Over-sampling
        additional = n_target - n
        indices = generate_indices(df_group, additional, law, parameters, with_replacement=True)
        duplicated = df_group.iloc[indices]
        return pd.concat([df_group, duplicated], ignore_index=True)

def apply_weighted_sampling_quantity(df_group, n_target, law="Uniform", parameters=None):
    """
    Applies weighted random sampling based on a probability law.
    Sampling is now based on an absolute number n_target (not a ratio).
    """
    n = len(df_group)
    if n == 0:
        return df_group.copy()

    n_target = max(1, int(n_target))  # safety

    weights = generate_weights(n, law, parameters)
    if len(weights) != n:
        raise ValueError("Length of weights does not match number of rows in df_group")

    # replace=True if oversampling
    replace = n_target > n

    sampled = np.random.choice(df_group.index, size=n_target, replace=replace, p=weights)
    return df_group.loc[sampled].copy()



class ClusteringVariation:

    # constructor
    def __init__(self, demand_path="", labels_path=""):
        """
        ClusteringVariation constructor.

        Parameters
        ----------
        demand_path : string
            Path to the original demand file.
        labels_path : string
            Path to the labels.
        """

        # check
        if not demand_path or demand_path.strip() == "":
            logger.error("Invalid or null demand_path.")
            raise ValueError("Invalid or null demand_path.")
        if not labels_path or labels_path.strip() == "":
            logger.error("Invalid or null labels_path.")
            raise ValueError("Invalid or null labels_path.")

        # assignment
        self._demand_path = demand_path
        self._labels_path = labels_path

        self._demand = pd.read_csv(demand_path, sep=';')
        self._labels = pd.read_csv(labels_path, sep=';')

        # check
        if self._demand.empty:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")
        if self._labels.empty:
            logger.error("_labels is empty.")
            raise ValueError("_labels is empty.")
        if self._demand.shape[0] != self._labels.shape[0]:
            logger.error("Invalid _labels.")
            raise ValueError("Invalid _labels.")

        self._demand["LABEL"] = self._labels

        # variation parameters
        
        self._populations_list = [] # A list containing every population subgroups, example : ["Student", "Executive", "Manual"]. By default self._populations_list = ["Default"]
        self._populations_variations = {} # a dict containing all population  subgroups with the way they vary (density and its parameters, ratio interval, transports), example : {"Student":{"Density":"Uniform", "Parameters":None, "Ratio":[1,1], "Transport":["CAR", "METRO", "TRAM", "BUS"]}, ect ...}, by default self._populations_variations = {"Default":{"Method":"Density", "Law":"Uniform", "Parameters":"None", "Ratio":[1,1], "Transport":["CAR", "METRO", "TRAM", "BUS"]}}, Parameters is a list
        self._populations_distributions = [] # A list where each element represent a cluster. Each element is a dict with the identifiers of the population subgroups as keys and their proportion in the cluster as the value. Each cluster must have at least one group, the sum of the proportions of all population subgroups in a cluster must be equal to one. By default self._populations_distributions = [{"Default":1}] for each cluster.
        
        self._profile_list = [] # A list containing all every profile, example : ["season", "holyday"]. By default, self._profile_list = []
        self._profile_parameters = {} # A dict with the possible values of each profile, example {"holyday":{"holyday":0.3, "not holyday":0.7}}, ect ...}. By default self._profile_parameters = {}
        
        self._profile_variations_on_populations = {} # A dict with the profile identifier as the key and a dict representing its influence on the population subgroups ratio as the value, example : {"holyday":{"student":[-0.8,-0.8], "Executive":[-0.5,-0.5], "Manual":[-0.35,-0.25]}, ect ...}. By default, self._profile_variations_on_populations = {} 
        self._profile_variations_on_events = {} # A dict with the profile identifier as the key and a dict representing its influence on the events probability as the value, example { "holyday" : {"Strike-PublicTransport":0, "Accident-Road":0, "Work-Road":-0.5}}. By default, self._profile_variations_on_events = {}
        
        self._events_list = [] # A list with every events. this list can not be modifiy. By default self._events_list = ["PublicTransportStrike"].
        self._events_parameterers = {} # A dict with the possible values of each profile, example {"PublicTransortStrike":{"METRO":0.3, "METRO_TRAM":0.7, ect ...}}, ect ...}. By default self._profile_parameters = {"PublicTransortStrike":{"METRO":0.04, "METRO_TRAM":0.004, "METRO_TRAM_BUS":0.001,"METRO_BUS":0.02,"TRAM":0.004,"BUS":0.04"TRAM_BUS":0.01,"None":0.9}}
        
        self.load_default_parameters()

        logger.info("ClusteringVariation initialized.")

    # Configuration methods

    def load_default_parameters(self):
        """
        Initializes the _parameters variables with default values. By default, there is only one population subgroup, "Default", with a uniform density and  no possible variation (ratio between 1 and 1). There are no profile and two events : "Strike-PublicTransport" and "Work-Road".

        No parameters
        """

        self._populations_list = ["Default"]
        self._populations_variations = {"Default":{"Method":"Density","Law":"Uniform", "Parameters":None, "Ratio":[1,1], "Transport":["CAR", "METRO", "TRAM", "BUS"]}}
        for i in range(self._labels.shape[0]):
            self._populations_distributions.append({"Default":1})
       
        
        self._profile_list = [] 
        self._profile_parameters = {} 
        self._profile_variations_on_populations = {} 
        self._profile_variations_on_events = {} 
        
        self._events_list = ["PublicTransportStrike"] 
        self._events_parameters = {"PublicTransportStrike":{"METRO":0.04, "METRO_TRAM":0.004, "METRO_TRAM_BUS":0.001,"METRO_BUS":0.02,"TRAM":0.004,"BUS":0.04, "TRAM_BUS":0.01,"None":0.9}}
            
        logger.info("default parameters loaded.")

        
    def load_parameters(self, populations_list, populations_variations, populations_distributions, profile_list, profile_parameters, profile_variations_on_populations, profile_variations_on_events, events_parameters):
        """
        Loads parameters.

        Parameters
        ----------
        subgroups_variations_parameters : dict

        subgroups_distribution_parameters : list

        day_variations_parameters : dict

        day_distribution_parameters : dict

        events_distribution_parameters : dict
        
        """

        # check 
        if set(populations_list) != set(populations_variations.keys()):
            logger.error("Invalid population subgroups.")
            raise ValueError("Invalid population subgroups.")
        for group, subdict in profile_variations_on_populations.items():
            if set(populations_list) != set(subdict.keys()):
                logger.error("Invalid population subgroups.")
                raise ValueError("Invalid population subgroups.")
        for i, dist in enumerate(populations_distributions):
            if set(populations_list) != set(dist.keys()):
                logger.error("Invalid population subgroups.")
                raise ValueError("Invalid population subgroups.")

            
        if set(profile_list) != set(profile_parameters.keys()):
            logger.error("Invalidprofile.")
            raise ValueError("Invalid profile.")
        profile_parameters_set = {key for subdict in profile_parameters.values() for key in subdict.keys()}
        logger.info(profile_parameters_set)
        if profile_parameters_set != set(profile_variations_on_events.keys()):
            logger.error("Invalid profile.")
            raise ValueError("Invalid profile.")

        logger.info(events_parameters.keys())
        if set(self._events_list) != set(events_parameters.keys()):
            logger.error("Invalid events.")
            raise ValueError("Invalid events.")

        for profile, events in profile_variations_on_events.items(): 
            if set(self._events_list) != set(events.keys()):
                logger.error(f"Invalid events in {profile}.")
                raise ValueError(f"Invalid events in {profile}.")
                
            for event_name, variations in events.items():
                if set(variations.keys()) != set(events_parameters[event_name].keys()):
                    logger.error(f"Invalid keys for event '{event_name}' in season '{season}'.")
                    raise ValueError(f"Invalid keys for event '{event_name}' in season '{season}'.")

        for event_name, params in events_parameters.items():
            total = sum(params.values())
            if abs(total - 1.0) > 1e-8:
                logger.error(f"Sum of values for event '{event_name}' != 1 (got {total}).")
                raise ValueError(f"Sum of values for event '{event_name}' != 1 (got {total}).")




        for i, dist in enumerate(populations_distributions):
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-8:  # tolérance pour les flottants
                logger.error("Invalid population distributions.")
                raise ValueError("Invalid population distributions.")
        for key, subdict in profile_parameters.items():
            total = sum(subdict.values())
            if abs(total - 1.0) > 1e-8:
                logger.error("Invalid profile parameters.")
                raise ValueError("Invalid profile parameters.")
            



        # other check ?

        # assignment
        self._populations_list = populations_list
        self._populations_variations = populations_variations
        self._populations_distributions = populations_distributions
        
        self._profile_list = profile_list 
        self._profile_parameters = profile_parameters
        self._profile_variations_on_populations = profile_variations_on_populations
        self._profile_variations_on_events = profile_variations_on_events
        
        self._events_parameters = events_parameters

        logger.info("paramaters loaded")


    def save_parameters(self, path):
        """
        Saves all internal parameters of the module into a text file named 'parameters.txt'.

        Parameters
        ----------
        path : str
            The directory where the file 'parameters.txt' will be saved.
        """

    

        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "parameters.txt")

        with open(file_path, "w") as f:
            f.write("===== Clustering Variation Parameters =====\n\n")

            f.write("=== Populations List ===\n")
            f.write(f"{self._populations_list}\n\n")

            f.write("=== Populations Variations ===\n")
            f.write(json.dumps(self._populations_variations, indent=4))
            f.write("\n\n")

            f.write("=== Populations Distributions ===\n")
            f.write(json.dumps(self._populations_distributions, indent=4))
            f.write("\n\n")

            f.write("=== Profile List ===\n")
            f.write(f"{self._profile_list}\n\n")

            f.write("=== Profile Parameters ===\n")
            f.write(json.dumps(self._profile_parameters, indent=4))
            f.write("\n\n")

            f.write("=== Profile Variations on Populations ===\n")
            f.write(json.dumps(self._profile_variations_on_populations, indent=4))
            f.write("\n\n")

            f.write("=== Profile Variations on Events ===\n")
            f.write(json.dumps(self._profile_variations_on_events, indent=4))
            f.write("\n\n")
 
            f.write("=== Events List ===\n")
            f.write(f"{self._events_list}\n\n")

            f.write("=== Events Parameters ===\n")
            f.write(json.dumps(self._events_parameters, indent=4))  # anciennement events_distributions
            f.write("\n")

        logger.info(f"Parameters saved to {file_path}")


    def save_variable_csvs(self, path):
        """
        Saves the 9 main variables of the class as separate CSVs in a subfolder called 'parameters'.
        """

        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        # 1. populations_list
        df_pop_list = pd.DataFrame({"Population": self._populations_list})
        df_pop_list.to_csv(os.path.join(path, "populations_list.csv"), index=False, sep=';')

        # 2. populations_variations
        pop_var_processed = {}
        for k, v in self._populations_variations.items():
            pop_var_processed[k] = {
                key: "_".join(map(str, val)) if isinstance(val, list) else val
                for key, val in v.items()
            }
        df_pop_var = pd.DataFrame.from_dict(pop_var_processed, orient="index")
        df_pop_var.to_csv(os.path.join(path, "populations_variations.csv"), sep=';')

        # 3. populations_distributions
        df_pop_dist = pd.DataFrame(self._populations_distributions)
        df_pop_dist.to_csv(os.path.join(path, "populations_distributions.csv"), index=False, sep=';')

        # 4. profile_list
        df_profile_list = pd.DataFrame({"Profile": self._profile_list})
        df_profile_list.to_csv(os.path.join(path, "profile_list.csv"), index=False, sep=';')

        # 5. profile_parameters
        profile_param_processed = {}
        for k, v in self._profile_parameters.items():
            profile_param_processed[k] = {
                key: "_".join(map(str, val)) if isinstance(val, list) else val
                for key, val in v.items()
            }
        df_profile_param = pd.DataFrame.from_dict(profile_param_processed, orient="index")
        df_profile_param.to_csv(os.path.join(path, "profile_parameters.csv"), sep=';')

        # 6. profile_variations_on_populations
        profile_pop_var_processed = {}
        for k, v in self._profile_variations_on_populations.items():
            profile_pop_var_processed[k] = {
                key: "_".join(map(str, val)) if isinstance(val, list) else val
                for key, val in v.items()
            }
        df_profile_pop_var = pd.DataFrame.from_dict(profile_pop_var_processed, orient="index")
        df_profile_pop_var.to_csv(os.path.join(path, "profile_variations_on_populations.csv"), sep=';')

        # 7. profile_variations_on_events
        profile_event_var_processed = {}
        for k, v in self._profile_variations_on_events.items():
            profile_event_var_processed[k] = {
                key: "_".join(map(str, val)) if isinstance(val, list) else val
                for key, val in v.items()
            }
        df_profile_event_var = pd.DataFrame.from_dict(profile_event_var_processed, orient="index")
        df_profile_event_var.to_csv(os.path.join(path, "profile_variations_on_events.csv"), sep=';')

        # 8. events_list
        df_events_list = pd.DataFrame({"Event": self._events_list})
        df_events_list.to_csv(os.path.join(path, "events_list.csv"), index=False, sep=';')

        events_param_processed = {}
        for k, v in self._events_parameters.items():
            events_param_processed[k] = {
                subkey: "_".join(map(str, val)) if isinstance(val, list) else val
                for subkey, val in v.items()
            }
        df_events_param = pd.DataFrame.from_dict(events_param_processed, orient="index")
        df_events_param.to_csv(os.path.join(path, "events_parameters.csv"), sep=';')


        logger.info(f"All 9 variables saved in {path}.")

        

    # variations

    def variations(self, path, n):
        """
        Creates variation and saves them at path.

        parameters
        ----------
        path : string
            The path to the directory to save the variation.
        n : int
            the number of variation.
        """

        # check
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise valueError("Invalid or null path.")
        if not n:
            logger.error("Null n.")
            raise ValueError("Null n.")

        # useful functions

        def draw_from_weighted_category(data, category):
            """
            Randomly selects one element from a weighted category.

            Parameters
            ----------
            data : dict
                Dictionary structured as:
                {
                    "category1": {"A": 0.2, "B": 0.5, "C": 0.3},
                    "category2": {"X": 0.7, "Y": 0.3}
                }
                Each sub-dictionary represents a weighted distribution where
                the sum of all proportions must equal 1.
            category : str
                The category name from which to draw an element.

            Returns
            -------
            str
                Randomly selected element according to its weight.

            Raises
            ------
            ValueError
                If the category is missing, proportions don’t sum to 1,
                or any weight is not in the [0, 1] range.
            """
            if category not in data:
                raise ValueError(f"Category '{category}' not found in data.")

            weights_dict = data[category]

            # Check that proportions sum to 1
            if abs(sum(weights_dict.values()) - 1) > 1e-9:
                raise ValueError(f"Proportions in category '{category}' do not sum to 1.")

            # Check that all weights are between 0 and 1
            for k, v in weights_dict.items():
                if not (0 <= v <= 1):
                    raise ValueError(f"Proportion for '{k}' in '{category}' must be between 0 and 1.")

            # Perform weighted random draw
            elements = list(weights_dict.keys())
            weights = list(weights_dict.values())
            choice = random.choices(elements, weights=weights, k=1)[0]

            return choice

        def sample_independent_events(event_probs):
            """
            Simulates a set of independent events.

            Each key in `event_probs` is considered an independent event that occurs
            with its given probability. Returns a list of the events that occurred.

            Parameters
            ----------
            event_probs : dict
                A dictionary where keys are event names (str) and values are
                probabilities of occurrence (floats between 0 and 1).

            Returns
            -------
            list
                A list of events that occurred during this simulation.
            """
            occurred = []
            for event, p in event_probs.items():
                if not 0 <= p <= 1:
                    raise ValueError(f"Probability for event '{event}' must be between 0 and 1.")
                # Simulate a Bernoulli trial: occurs with probability p
                if random.random() < p:
                    occurred.append(event)
            return occurred

        os.makedirs(path, exist_ok=True)

        # variations
        for i in range(n):
            logger.info(f"{i+1}e variation.")

            # day profile
            profile = []
            for category in self._profile_list:
                profile.append(draw_from_weighted_category(self._profile_parameters,category))

            # population profile
            population_profile = self._populations_variations.copy()
            for group in population_profile.keys():

                # the density and its parameters does not change

                # each profile can potentially modify every population subgroup
                for category in profile :
                    population_profile[group]["Ratio"][0] += self._profile_variations_on_populations[category][group][0]
                    population_profile[group]["Ratio"][1] += self._profile_variations_on_populations[category][group][1]

            for group in population_profile.keys():
                if population_profile[group]["Ratio"][0] < 0 : population_profile[group]["Ratio"][0] = 0

                if population_profile[group]["Ratio"][1] < 0 : population_profile[group]["Ratio"][1] = 0

            # event profile
            events_parameters = self._events_parameters.copy()
            # each profile modify the odds
            for category in profile :
                for event in events_parameters.keys():
                    for spec in events_parameters[event].keys():
                        events_parameters[event][spec] += self._profile_variations_on_events[category][event][spec]

            for event_name, params in events_parameters.items():
                total = sum(params.values())
                if abs(total - 1.0) > 1e-8:
                    logger.error(f"Sum of values for event '{event_name}' != 1 (got {total}).")
                    raise ValueError(f"Sum of values for event '{event_name}' != 1 (got {total}).")

            events = []
            for event in self._events_list:
                events.append(draw_from_weighted_category(events_parameters, event))

            # Step 4: apply variations on clusters
            df = self._demand.copy()
            clusters = sorted(df["LABEL"].unique())
            logger.info(clusters)
            all_cluster_results = []

            for cluster_id in clusters:
                df_cluster = df[df["LABEL"] == cluster_id].drop(columns=["LABEL"])
                cluster_dist = self._populations_distributions[cluster_id]
                cluster_size = len(df_cluster)

                # Randomly assign population to each line according to cluster proportions
                population_labels = np.random.choice(
                    list(cluster_dist.keys()),
                    size=cluster_size,
                    p=list(cluster_dist.values())
                )
                df_cluster["POPULATION"] = population_labels

                new_cluster_df_list = []

                # For each population in this cluster
                for pop_name, proportion in cluster_dist.items():
                    df_pop = df_cluster[df_cluster["POPULATION"] == pop_name]
                    if df_pop.empty:
                        continue

                    # with the modifies variations
                    params = population_profile[pop_name]
                    ratio_min, ratio_max = params["Ratio"]
                    ratio = np.random.uniform(ratio_min, ratio_max)

                    method = params.get("Method", "Density")
                    law = params.get("Law", "Uniform")
                    law_params = params.get("Parameters", None)

                    # Choose variation type
                    if method == "Density":
                        df_pop_new = apply_density_variation(df_pop, ratio, law, law_params)
                    elif method == "WeightedSampling":
                        df_pop_new = apply_weighted_sampling(df_pop, ratio, law, law_params)
                    else:
                        raise ValueError(f"Unknown variation method '{method}' for population '{pop_name}'")

                    # Get allowed transports for this population group
                    allowed_transports = params.get("Transport", ["CAR", "METRO", "TRAM", "BUS"])

                    df_pop_new = df_pop_new.copy()

                    # Assign MOBILITY SERVICES column based on allowed transports
                    df_pop_new["MOBILITY SERVICES"] = [" ".join(allowed_transports)] * len(df_pop_new)

                    new_cluster_df_list.append(df_pop_new)

                df_cluster_new = pd.concat(new_cluster_df_list, ignore_index=True)
                all_cluster_results.append(df_cluster_new)

            # Combine all clusters
            df_variation = pd.concat(all_cluster_results, ignore_index=True)

            # Rebuild unique IDs
            df_variation["ID"] = range(len(df_variation))

            # Save variation
            if "POPULATION" in df_variation.columns:
                df_variation = df_variation.drop(columns=["POPULATION"])
            # Sort by DEPARTURE
            if "DEPARTURE" in df_variation.columns:
                df_variation = df_variation.sort_values(by="DEPARTURE").reset_index(drop=True)

            # Generate string for profile: concatenate the actual profile values used in this variation
            #profile_str = "_".join([str(value) for value in profile])

            # Generate fine-grained binary string for profile
            profile_str_parts = []

            for profile_name in self._profile_list:
                # profile_values contains the sub-categories drawn for this profile in this variation
                # Exemple : profile_values = ["summer"] pour "season"
                profile_values = [value for cat, value in zip(self._profile_list, profile) if cat == profile_name]
    
                # Get all possible sub-categories for this profile
                subcategories = list(self._profile_parameters[profile_name].keys())
    
                # Create 0/1 for each sub-category
                profile_str_parts.extend(["1" if subcat in profile_values else "0" for subcat in subcategories])

            profile_str = "_".join(profile_str_parts)


            # Generate fine-grained binary string for events
            event_str_parts = []

            for event_name in self._events_list:
                # event_values contains the sub-categories drawn for this profile in this variation
                # Exemple : event_values = ["summer"] pour "season"
                event_values = [value for cat, value in zip(self._events_list, events) if cat == event_name]
    
                # Get all possible sub-categories for this profile
                subcategories = list(self._events_parameters[event_name].keys())
    
                # Create 0/1 for each sub-category
                event_str_parts.extend(["1" if subcat in event_values else "0" for subcat in subcategories])

            event_str = "_".join(event_str_parts)

            # Build output filename
            output_path = f"{path}/variation_{i+1}__{profile_str}__{event_str}.csv"
                                                                                   
            df_variation.to_csv(output_path, sep=';', index=False)
            logger.info(f"Variation {i+1}: {len(df_variation)} rows (original demand: {len(self._demand)} rows).")
            logger.info(f"Variation {i+1} saved at {output_path}.")

        parameters_path = os.path.join(path, "parameters")
        os.makedirs(parameters_path, exist_ok=True)
        self.save_parameters(parameters_path)
        self.save_variable_csvs(parameters_path)
        logger.info(f"Parameters saved at {parameters_path}.")
        logger.info(f"Variations produced.")



    def variations_clusters_ratio(self, path, ratio_tot, ratio_cluster, methode):
        """
        Creates variations of the demand DataFrame based on total and per-cluster ratios.

        Parameters
        ----------
        path : str
            Directory path where variations will be saved.
        ratio_tot : list [start, end, step]
            Defines the range of total ratios to apply (e.g., [0.8, 1.2, 0.1]).
        ratio_cluster : list [start, end, step]
            Defines the range of ratios to apply per cluster.
        methode : str
            String in the form "type-law" (e.g., "WeightedSampling-normal" or "Density-uniform").
        """

        # === Checks =============================================================
        if not path or path.strip() == "":
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")

        os.makedirs(path, exist_ok=True)

        if not isinstance(ratio_tot, (list, tuple)) or len(ratio_tot) != 3:
            raise ValueError("ratio_tot must be a list [start, end, step].")
        if not isinstance(ratio_cluster, (list, tuple)) or len(ratio_cluster) != 3:
            raise ValueError("ratio_cluster must be a list [start, end, step].")
        if not isinstance(methode, str) or "-" not in methode:
            raise ValueError("methode must be a string like 'WeightedSampling-normal' or 'Density-uniform'.")

        method_type, law = methode.split("-")
        if method_type not in ["WeightedSampling", "Density"]:
            raise ValueError("Unknown method type. Must be 'WeightedSampling' or 'Density'.")

        # === Generate ratio ranges =============================================
        total_ratios = np.arange(ratio_tot[0], ratio_tot[1] + ratio_tot[2] / 10, ratio_tot[2])
        cluster_ratios = np.arange(ratio_cluster[0], ratio_cluster[1] + ratio_cluster[2] / 10, ratio_cluster[2])

        df = self._demand.copy()
        clusters = sorted(df["LABEL"].unique())

        logger.info(f"Clusters found: {clusters}")

        # === Iterate through all combinations ==================================
        variant_counter = 1
        for tot_ratio in total_ratios:
            for cl_ratio in cluster_ratios:
                for cluster_id in clusters:

                    logger.info(f"Creating variation {variant_counter}: total={tot_ratio}, cluster={cluster_id} ratio={cl_ratio}")

                    df_copy = df.copy()

                    # 1️⃣ Calculate initial cluster sizes
                    cluster_sizes = df_copy["LABEL"].value_counts().to_dict()
                    total_original = len(df_copy)

                    # 2️⃣ Adjust the selected cluster
                    target_cluster_size = int(cluster_sizes[cluster_id] * cl_ratio)
                    cluster_variants = {}

                    for cid in clusters:
                        df_cluster = df_copy[df_copy["LABEL"] == cid]

                        if cid == cluster_id:
                            # Cluster ciblé → appliquer le ratio spécifique
                            target_size = target_cluster_size
                        else:
                            # Clusters restants : taille initiale (ajustée plus tard)
                            target_size = len(df_cluster)

                        cluster_variants[cid] = {"df": df_cluster, "target_size": target_size}

                    # 3️⃣ Ajustement global pour respecter le ratio total
                    total_target = int(total_original * tot_ratio)
                    current_total = sum(v["target_size"] for v in cluster_variants.values())
                    difference = total_target - current_total

                    if difference != 0:
                        other_clusters = [c for c in clusters if c != cluster_id]
                        if other_clusters:
                            per_cluster_adjust = difference // len(other_clusters)
                            for c in other_clusters:
                                cluster_variants[c]["target_size"] = max(
                                    1, cluster_variants[c]["target_size"] + per_cluster_adjust
                                )

                    # 4️⃣ Appliquer les méthodes de variation
                    new_clusters = []
                    for cid, info in cluster_variants.items():
                        df_cluster = info["df"]
                        n = len(df_cluster)
                        ratio = info["target_size"] / n if n > 0 else 1.0

                        if method_type == "WeightedSampling":
                            df_new = apply_weighted_sampling(df_cluster, ratio, law)
                        else:
                            df_new = apply_density_variation(df_cluster, ratio, law)

                        new_clusters.append(df_new)

                    # 5️⃣ Concaténer
                    df_variant = pd.concat(new_clusters, ignore_index=True)

                    # 6️⃣ Réassigner des IDs uniques
                    df_variant["ID"] = range(len(df_variant))

                    # 7️⃣ Tri et sauvegarde
                    if "DEPARTURE" in df_variant.columns:
                        df_variant = df_variant.sort_values(by="DEPARTURE").reset_index(drop=True)

                    # 8️⃣ Nom du fichier
                    ratios_str = "_".join([
                        f"{cid}-{round(cluster_variants[cid]['target_size'] / len(df[df['LABEL'] == cid]), 3)}"
                        for cid in clusters
                    ])
                    ratio_cluster_str = f"{cluster_id}-{round(cl_ratio,3)}"
                    filename = f"variation_{variant_counter}__{method_type}-{law}__tot-{round(tot_ratio,3)}__{ratio_cluster_str}__{ratios_str}.csv"
                    output_path = os.path.join(path, filename)

                    df_variant.to_csv(output_path, sep=';', index=False)
                    logger.info(f"Variation {variant_counter} saved at {output_path} ({len(df_variant)} rows).")

                    variant_counter += 1

        logger.info(f"All {variant_counter - 1} variations saved successfully at {path}.")



    def variations_clusters_quantities(self, path, qty_tot, qty_cluster, methode):
        """
        Creates variations of the demand DataFrame based on total and per-cluster quantity offsets.

        Parameters
        ----------
        path : str
            Directory path where variations will be saved.
        qty_tot : list [start, end, step]
            Offsets added to the total quantity (e.g. [-10, 10, 5]).
        qty_cluster : list [start, end, step]
            Offsets added to the target cluster quantity (e.g. [-1, 1, 1]).
        methode : str
            Method in the form "type-law" (e.g., "WeightedSampling-normal").
        """

        # === Checks =============================================================
        if not path or path.strip() == "":
            raise ValueError("Invalid or null path.")

        os.makedirs(path, exist_ok=True)

        if not isinstance(qty_tot, (list, tuple)) or len(qty_tot) != 3:
            raise ValueError("qty_tot must be a list [start, end, step].")
        if not isinstance(qty_cluster, (list, tuple)) or len(qty_cluster) != 3:
            raise ValueError("qty_cluster must be a list [start, end, step].")
        if not isinstance(methode, str) or "-" not in methode:
            raise ValueError("methode must be a string like 'WeightedSampling-normal'.")

        method_type, law = methode.split("-")
        if method_type not in ["WeightedSampling", "Density"]:
            raise ValueError("Unknown method type. Must be 'WeightedSampling' or 'Density'.")

        # === Create offset ranges (inclusive) ===================================
        def arange_inclusive(start, end, step):
            return np.arange(start, end + step/10, step, dtype=int)

        offset_tot_range = arange_inclusive(*qty_tot)
        offset_cluster_range = arange_inclusive(*qty_cluster)

        # === Load data ===========================================================
        df = self._demand.copy()
        clusters = sorted(df["LABEL"].unique())

        original_cluster_sizes = df["LABEL"].value_counts().to_dict()
        original_total = len(df)

        variant_counter = 1

        # === Iterate through all variations =====================================
        for offset_tot in offset_tot_range:
            for offset_cl in offset_cluster_range:
                for target_cluster in clusters:

                    # 1️⃣ Determine target sizes per cluster ------------------------
                    cluster_variants = {}
                    no_change = True
                    for cid in clusters:
                        base_size = original_cluster_sizes[cid]

                        if cid == target_cluster:
                            # Apply the offset to the selected cluster
                            target_size = max(1, base_size + offset_cl)
                        else:
                            # Other clusters unchanged (adjusted later)
                            target_size = base_size

                        # Vérifier si un changement existe
                        if target_size != base_size:
                            no_change = False   


                        cluster_variants[cid] = {
                            "df": df[df["LABEL"] == cid],
                            "target_size": target_size
                        }

                    if no_change and offset_tot == 0:    ### NEW → pas de modif cluster + pas de modif total
                        continue

                    # 2️⃣ Adjust to satisfy total offset ----------------------------
                    total_target = max(1, original_total + offset_tot)
                    current_total = sum(v["target_size"] for v in cluster_variants.values())
                    difference = total_target - current_total

                    if difference != 0:
                        other_clusters = [c for c in clusters if c != target_cluster]
                        if other_clusters:
                            adjust = difference // len(other_clusters)
                            for cid in other_clusters:
                                cluster_variants[cid]["target_size"] = max(
                                    1,
                                    cluster_variants[cid]["target_size"] + adjust
                                )

                    # 3️⃣ Apply variation method to each cluster --------------------
                    new_clusters = []
                    for cid, info in cluster_variants.items():
                        df_cluster = info["df"]
                        n_target = info["target_size"]

                        if method_type == "WeightedSampling":
                            df_new = apply_weighted_sampling_quantity(df_cluster, n_target=n_target, law=law)
                        else:
                            df_new = apply_density_variation_quantity(df_cluster, n_target=n_target, law=law)

                        new_clusters.append(df_new)

                    # 4️⃣ Concatenate all clusters ---------------------------------
                    df_variant = pd.concat(new_clusters, ignore_index=True)

                    # 5️⃣ Reassign unique IDs --------------------------------------
                    df_variant["ID"] = range(len(df_variant))

                    # 6️⃣ Sort by DEPARTURE if present ------------------------------
                    if "DEPARTURE" in df_variant.columns:
                        df_variant = df_variant.sort_values(by="DEPARTURE").reset_index(drop=True)

                    # 7️⃣ Save file --------------------------------------------------
                    quantities_str = "_".join(
                        f"{cid}-{cluster_variants[cid]['target_size']}"
                        for cid in clusters
                    )
                    quantity_cluster_str = f"{target_cluster}-{round(offset_cl,3)}"
                    filename = (
                        f"variation_{variant_counter}"
                        f"__{method_type}-{law}"
                        f"__tot-{total_target}"
                        f"__{quantity_cluster_str}"
                        f"__{quantities_str}.csv"
                    )

                    output_path = os.path.join(path, filename)
                    df_variant.to_csv(output_path, sep=';', index=False)

                    print(f"[OK] Variation {variant_counter} saved → {filename}")

                    variant_counter += 1

        print(f"\nAll {variant_counter - 1} variations saved successfully at: {path}\n")

            



    # getters

    def get_demand_path(self):
        #check
        if not self._demand_path or self._demand_path.strip() == "":
            logger.error("Invalid or null _demand_path.")
            raise ValueError("Invalid or null _demand_path.")
        return self._demand_path

    def get_labels_path(self):
        #check
        if not self._labels_path or self._labels_path.strip() == "":
            logger.error("Invalid or null _labels_path.")
            raise ValueError("Invalid or null _labels_path.")
        return self._demand_path

    def get_demand(self):
        # check
        if self._demand.empty:
            logger.error("_demand is empty.")
            raise ValueError("_demand is empty.")
        return self._demand

    def get_labels(self):
        # check
        if self._labels.empty:
            logger.error("_labels is empty.")
            raise ValueError("_labels is empty.")
        return self._labels

    def get_populations_list(self):
        return self._populations_list

    def get_populations_variations(self):
        return self._populations_variations

    def get_populations_distributions(self):
        return self._populations_distributions

    def get_profile_list(self):
        return self._profile_list

    def get_profile_parameters(self):
        return self._profile_parameters

    def get_profile_variations_on_populations(self):
        return self._profile_variations_on_populations

    def get_events_list(self):
        return self._events_list

    def get_events_distributions(self):
        return self._events_distributions


    
        