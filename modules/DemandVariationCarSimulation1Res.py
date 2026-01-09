# dependencies

import os
from pathlib import Path

from mnms.mobility_service.on_demand import OnDemandDepotMobilityService
from mnms.simulation import Supervisor
from mnms.demand import CSVDemandManager
from mnms.flow.MFD import Reservoir, MFDFlowMotor
from mnms.log import attach_log_file, LOGLEVEL, get_logger, set_all_mnms_logger_level, set_mnms_logger_level
from mnms.time import Time, Dt
from mnms.io.graph import load_graph, load_odlayer, save_odlayer
from mnms.travel_decision.logit import LogitDecisionModel, ModeCentricLogitDecisionModel
from mnms.travel_decision.dummy import DummyDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_bbox_origin_destination_layer, generate_matching_origin_destination_layer
from mnms.generation.layers import generate_bbox_origin_destination_layer
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.mobility_service.public_transport import PublicTransportMobilityService
from mnms.io.graph import save_transit_link_odlayer, load_transit_links
from mnms.tools.render import draw_roads

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

import logging


logger = logging.getLogger(__name__)


class DemandVariationCarSimulation1Res:
    """
    MnMS simulation for given variations.

    Attributes
    ----------
    _input_path : str 
        Path to directory file containing the demand files.
    _inputs : list
        Input files.
    _decisions_model : set
        Decision models set, value : {"Dummy", "Logit", "CentricLogit"}.
    _graph_path : str
        Relative path to the graph file.
    """
    # Constructor

    def __init__(self, input_path="", graph_path=""):
        """
        Production's constructor.

        Parameters
        ----------
        input_path : str
            Relative path to a folder containing MnMS demand files.
        graph_path : str
            Relative path to the graph file.
        
        """

        # check
        if not input_path or input_path.strip == "" :
            logger.info("Null or invalid input path.")
            raise ValueError("Null or invalid input path.")
        if not graph_path or graph_path.strip == "" :
            logger.info("Null or invalid graph path.")
            raise ValueError("Null or invalid graph path.")

        # assignment
        self._input_path = input_path
        self._graph_path = graph_path

        # retrieves the inputs files in a list
        self._inputs = list(Path(self._input_path).glob("*.csv"))
        self._inputs.sort()

        if len(self._inputs) == 0 :
            logger.info("No demand files in the given directory.")
            raise ValueError("No demand files in the given directory.")

        # variables
        self._decision_models = {"Dummy", "Logit", "CentricLogit"}

        logger.info(f"DemandVariationSimulation object built with _input_path : {self._input_path}.")




    def change_input(self, input_path) : 
        """
        Changes the input.

        Parameters
        ----------
        input_path : str
            Relative path to a folder containing MnMS demand files.
        """

        # check
        if not input_path or input_path.strip == "" :
            logger.info("Null or invalid input path.")
            raise ValueError("Null or invalid input path.")

        self._input_path = input_path

        # retrieves the inputs files in a list
        self._inputs = list(Path(self._input_path).glob("*.csv"))
        self._inputs.sort()

        if len(self._inputs) == 0 :
            logger.info("No demand files in the given directory.")
            raise ValueError("No demand files in the given directory.")

        logger.info(f"Input changed with path {self._input_path}.")

    def change_graph(self, graph_path):
        """
        Changes the graph path.

        Parameters
        ----------
        graph_path : str
            Relative path to the graph file.
        """

        if not graph_path or graph_path.strip == "" :
            logger.info("Null or invalid graph path.")
            raise ValueError("Null or invalid graph path.")

        self._graph_path = graph_path

    def create_network_files(self, nx=14, ny=16, dist_connection=1000, directory=""):
        """
        Creates the od layer and the transit link files with nx*ny nodes and dist_connection then saved them at directory.
        For the od layer, the bbox method is used : it creates a bbox from the graph file and then adds a grid representing the od layer with 
        nx*ny nodes. 

        Parameters
        ----------
        nx : int
            Number of nodes per line.
        ny : int
            Number of nodes per column.
        dist_connection : int
            Used to connect each layer with the graph.
        directory : 
            Directory to save the files.
        """

        # check
        if not directory or directory.strip() == "" : 
            logger.info("Null or invalid directory.")
            raise ValueError("Null or invalid directory.")

        NX = nx
        NY = ny
        DIST_CONNECTION = dist_connection

        mmgraph = load_graph(self._graph_path)
        graph_name = Path(self._graph_path).stem

        odlayer = generate_bbox_origin_destination_layer(mmgraph.roads, NX, NY)
        mmgraph.add_origin_destination_layer(odlayer)
        mmgraph.connect_origindestination_layers(DIST_CONNECTION)
        
        save_odlayer(odlayer, directory + f"/{graph_name}_od_layer_{NX}_{NY}_{DIST_CONNECTION}.json")
        save_transit_link_odlayer(mmgraph, directory + f"/{graph_name}_transit_link_{NX}_{NY}_{DIST_CONNECTION}_grid.json")

        logger.info(f"Od layer and transit link files created and saved at {directory} with : nx = {nx}, ny : {ny}, dist_connection : {dist_connection}.")




    # Simulation

    def simulate(self, od_layer_path="", transit_link_path="", MFD_variables=[], decision_model="Dummy", output_directory=""):
        """
        Simulations for the given inputs with a given od layer, transit link and MFD variables, the outputs are saved at directory.

        Parameters
        ----------
        od_layer_path : str
            Relative path to an od layer file.
        transit_link_path : str
            Relative path to a transit link file.
        MFD_variables : list
            MFD variables for a single reservoir with only cars.
        output_directory : str
            Directory to save the simulation outputs.
        """

        # check
        if not od_layer_path or od_layer_path.strip() == "" : 
            logger.info("Null or invalid od layer path.")
            raise ValueError("Null or invalid od layer path.")
        if not transit_link_path or transit_link_path.strip() == "" : 
            logger.info("Null or invalid transit link path.")
            raise ValueError("Null or invalid transit link path.")
        if len(MFD_variables) != 2 : 
            logger.info("Invalid MFD variables.")
            raise ValueError("Invalid MFD variables.")
        if not decision_model in self._decision_models : 
            logger.info("Invalid decision model.")
            raise ValueError("Invalid decision model.")
        if not output_directory or output_directory.strip() == "" : 
            logger.info("Null or invalid output directory.")
            raise ValueError("Null or invalid output directory.")

        # outputs configuration
        
        # creates an outputs directory for each demand file
        for file in self._inputs:
            directory_name = file.name[:-4]
            path = Path(output_directory) / directory_name
            path.mkdir(parents=True, exist_ok=True)

        # retrieves the outputs directory in a list
        path = Path(output_directory)
        outputs =[p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
        outputs.sort()

        if len(outputs) == 0: raise ValueError("No outputs directory.")

        a = MFD_variables[0]
        b = MFD_variables[1]

        # MFD definition
        def calculate_MFD(acc):
            V = 0  # data from fit dsty
            N = acc["CAR"]
            
            #a = 2.8981940480441857
            #b = -0.00010553060526140915
            V = np.exp(a + b * N)
    
            V = max(V, 0.001)  # min speed to avoid gridlock
            return {"CAR": V}

        # simulation for each demand file
        for i in range(len(self._inputs)):
            
            logger.info(f"{i+1}e simulation with input {self._inputs[i]}.")

            # outputs configuration
            outdir = outputs[i]
            logger.info(f"outdir {outdir.name}")
            
            # network defintion
            mmgraph = load_graph(self._graph_path)
            odlayer = load_odlayer(od_layer_path)
            mmgraph.add_origin_destination_layer(odlayer)
            load_transit_links(mmgraph, transit_link_path)
 
            # layer
            personal_car = PersonalMobilityService("CAR")
            mmgraph.layers["CAR"].add_mobility_service(personal_car)
            personal_car.attach_vehicle_observer(CSVVehicleObserver(outdir / "veh.csv"))            

            # load demand
            demand_file=self._inputs[i]
            demand = CSVDemandManager(demand_file)
            
            demand.add_user_observer(CSVUserObserver(outdir / "user.csv"), user_ids="all")

            flow_motor = MFDFlowMotor(outfile=outdir / "flow.csv")

            transport = ["CAR"]
            
            flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], transport, calculate_MFD))

            # decision model
            if decision_model == "Dummy":
                travel_decision = DummyDecisionModel(mmgraph, outfile=outdir / "path.csv")
            if decision_model == "Logit" : 
                travel_decision = LogitDecisionModel(mmgraph, outfile=outdir / "path.csv")
            if decision_model == "CentricLogit" : 
                travel_decision = ModeCentricLogitDecisionModel(mmgraph, ["Car"], outfile=outdir / "path.csv")
            

            supervisor = Supervisor(graph=mmgraph,
                                    flow_motor=flow_motor,
                                    demand=demand,
                                    decision_model=travel_decision,
                                    outfile=outdir / "travel_time_link.csv")
            
            buffer = pd.read_csv(demand_file, sep=';')
            buffer["DEPARTURE"] = pd.to_datetime(buffer["DEPARTURE"], format="%H:%M:%S").dt.time
            start_time = buffer["DEPARTURE"].min().strftime("%H:%M:%S")
            end_time = buffer["DEPARTURE"].max().strftime("%H:%M:%S")

            supervisor.run(Time(start_time), Time(end_time), Dt(minutes=1), 10)

            logger.info(f"{i+1}e simulation with input {self._inputs[i]} done.")

        logger.info("Simulations done.")


    # getters

    def get_input_path(self):
        return self._input_path

    
    def get_graph_path(self):
        return self._graph_path

    
    def get_inputs(self):
        return self._inputs


    def get_decision_models(self):
        return self._decision_models