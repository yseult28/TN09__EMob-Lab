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
from mnms.travel_decision.logit import LogitDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_bbox_origin_destination_layer, generate_matching_origin_destination_layer
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

class Clustering2Simulation:

    # Constructor
    
    def __init__(self, input_path="", output_path="", graph_path="", odlayer_path="",transit_link_path="", MFD_variables=[]):
        """
        Production's constructor.
    
        Parameters
        ----------
        input_path : string
            Path to the variations' folder.
        output_path : string
            Path to save the outputs.
        graph_path : string
            Path to the multilayer graph file of the simulation.
        odlayer_path : string
            Path to the odlayer file of the simulation.
        transit_link_path : string
            Path to the transit link layer file of the simulation.
        MFD_variables : list
            List of parameters for the MFD.
            Their must be 5 float in the list.
        """

        # check
        if not input_path or input_path.strip() == "": 
            logger.error("Invalid or null input_path.")
            raise ValueError("Invalid or null input_path.")
        if not output_path or output_path.strip() == "": 
            logger.error("Invalid or null output_path.")
            raise ValueError("Invalid or null output_path.")
        if not graph_path or graph_path.strip() == "": 
            logger.error("Invalid or null graph_path.")
            raise ValueError("Invalid or null graph_path.")
        if not odlayer_path or odlayer_path.strip() == "": 
            logger.error("Invalid or null odlayer_path.")
            raise ValueError("Invalid or null odlayer_path.")
        if not transit_link_path or transit_link_path.strip() == "": 
            logger.error("Invalid or null transit_link_path.")
            raise ValueError("Invalid or null transit_link_path.")
        if len(MFD_variables) != 5 : 
            logger.error("Invalid or null MFD_variables.")
            raise ValueError("Invalid or null MFD_variables.")

        # assignment
        self._input_path = input_path # path to directory file containing the demand files
        self._output_path = output_path # path to directory file that will contain the outputs files
        os.makedirs(output_path, exist_ok=True)
        self._graph_path = graph_path # path to the graph of the network
        self._odlayer_path = odlayer_path # path to the odlayer of the network
        self._transit_link_path = transit_link_path # path to the transit link layer of the network

        self._a = MFD_variables[0]
        self._b = MFD_variables[1]
        self._v_bus = MFD_variables[2]
        self._v_tram = MFD_variables[3]
        self._v_metro = MFD_variables[4]

        # retrieves the inputs files in a list
        self._inputs = list(Path(self._input_path).glob("*.csv"))
        self._inputs.sort()

        # events variables
        self._events = {"PublicTransportStrike":["METRO","METRO_TRAM","METRO_TRAM_BUS","METRO_BUS","TRAM","BUS", "TRAM_BUS","None"]}
        self._events_parameters = ["METRO","METRO_TRAM","METRO_TRAM_BUS","METRO_BUS","TRAM","BUS", "TRAM_BUS","None"]
        self._active_events = []
        self._avalaible_PublicTransport_means = {"METRO", "BUS","TRAM"}

        
        if len(self._inputs) == 0: raise ValueError("No demand files.")

        # creates an outputs directory for each demand file
        for file in self._inputs:
            directory_name = file.stem
            path = Path(self._output_path) / directory_name
            path.mkdir(parents=True, exist_ok=True)

        # retrieves the outputs directory in a list
        path = Path(self._output_path)
        self._outputs=[p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
        self._outputs.sort()

        if len(self._outputs) == 0: raise ValueError("No outputs directory.")

        self._outputs_produced = False

        logger.info(f"Production initialized with input path {input_path}, output path {output_path}, graph path {graph_path}, odlayer path {odlayer_path} and transit link path {transit_link_path} and with MFD variables {MFD_variables}.")
        logger.info(f"Simulations on the following directory : {self._outputs}")



    # Configuration's methods
    
    def change_inputs(self, input_path="", output_path="", graph_path="", odlayer_path="",transit_link_path="", MFD_variables=[]):
        """
        Change the inputs of the simulation
    
        Parameters
        ----------
        input_path : string
            Path to the variations' folder.
        output_path : string
            Path to save the outputs.
        graph_path : string
            Path to the multilayer graph file of the simulation.
        odlayer_path : string
            Path to the odlayer file of the simulation.
        transit_link_path : string
            Path to the transit link layer file of the simulation.
        MFD_variables : list
            List of parameters for the MFD.
            Their must be 5 float in the list.
        """
        
        # check
        if not input_path or input_path.strip() == "": 
            logger.error("Invalid or null input_path.")
            raise ValueError("Invalid or null input_path.")
        if not output_path or output_path.strip() == "": 
            logger.error("Invalid or null output_path.")
            raise ValueError("Invalid or null output_path.")
        if not graph_path or graph_path.strip() == "": 
            logger.error("Invalid or null graph_path.")
            raise ValueError("Invalid or null graph_path.")
        if not odlayer_path or odlayer_path.strip() == "": 
            logger.error("Invalid or null odlayer_path.")
            raise ValueError("Invalid or null odlayer_path.")
        if not transit_link_path or transit_link_path.strip() == "": 
            logger.error("Invalid or null transit_link_path.")
            raise ValueError("Invalid or null transit_link_path.")
        if len(MFD_variables) != 5 : 
            logger.error("Invalid or null MFD_variables.")
            raise ValueError("Invalid or null MFD_variables.")

        # assignment
        self._input_path = input_path # path to directory file containing the demand files
        self._output_path = output_path # path to directory file that will contain the outputs files
        self._graph_path = graph_path # path to the graph of the network
        self._odlayer_path = odlayer_path # path to the odlayer of the network
        self._transit_link_path = transit_link_path # path to the transit link layer of the network

        self._a = MFD_variables[0]
        self._b = MFD_variables[1]
        self._v_bus = MFD_variables[2]
        self._v_tram = MFD_variables[3]
        self._v_metro = MFD_variables[4]

        # retrieves the inputs files in a list
        self._inputs = list(Path(self._input_path).glob("*.csv"))
        self._inputs.sort()

       
        
        if len(self._inputs) == 0: raise ValueError("No demand files.")

        # creates an outputs directory for each demand file
        for file in self._inputs:
            directory_name = file.stem
            path = Path(self._output_path) / directory_name
            path.mkdir(parents=True, exist_ok=True)

        # retrieves the outputs directory in a list
        path = Path(self._output_path)
        self._outputs=[p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
        self._outputs.sort()

        if len(self._outputs) == 0: raise ValueError("No outputs directory.")

        self._outputs_produced = False

        logger.info(f"inputs changed with input path {input_path}, output path {output_path}, graph path {graph_path}, odlayer path {odlayer_path} and transit link path {transit_link_path} and with MFD variables {MFD_variables}.")
        logger.info(f"Simulations on the following directory : {self._outputs}")


    def remove_PublicTransport(self):
        """
        Removes Public Transport for available means.

        No parameters
        """

        self._avalaible_PublicTransport_means = {}

        logger.info("Public Transport removed.")

    
    # Production's method
       

    
    def produce_outputs(self):
        """
        Launch simulations and saves the outputs.
    
        No parameters
        """
        
        # check
        if self._outputs_produced == True: 
            logger.error("outputs already produced, you must change the inputs.")
            raise ValueError("outputs already produced, you must change the inputs.")

        # MFD definition
        def calculate_V_MFD(acc):

            V = 0  
            N = acc["CAR"]

            #a = 18.323199544221808
            #b = 4813.38024078608
            #V = a * np.exp(-N / (2 * b))
            V = self._a * np.exp(-N / (2 * self._b))
        
            # V = 11.5*(1-N/60000)
            # V = max(V, 0.001)  # min speed to avoid gridlock
            # V_TRAM_BUS = 0.7 * V
            # V = 13.8
        
            #V_BUS = 20
            #V_TRAM = 40
            #V_METRO = 60

            V = self._a * np.exp(-N / (2 * self._b))
        
            V_BUS = self._v_bus
            V_TRAM = self._v_tram
            V_METRO = self._v_metro

            # return {"CAR": V, "METRO": 17, "BUS": V_TRAM_BUS, "TRAM": V_TRAM_BUS}

            return {"CAR": V, "BUS": V_BUS, "TRAM": V_TRAM, "METRO": V_METRO}

        # simulation for each variation
        for i in range(len(self._inputs)):
            
            logger.info(f"{i+1}e simulation with input {self._inputs[i]}.")

            # outputs configuration
            outdir = self._outputs[i]
            logger.info(f"outdir {outdir.stem}")
            #self.extract_event_parameters(str(outdir.stem))
            #self.apply_events()
        
            # network defintion
            mmgraph = load_graph(self._graph_path)
            odlayer = load_odlayer(self._odlayer_path)
            mmgraph.add_origin_destination_layer(odlayer)
            load_transit_links(mmgraph, self._transit_link_path)

            
            # layer
            personal_car = PersonalMobilityService("CAR")
            mmgraph.layers["CAR"].add_mobility_service(personal_car)
            personal_car.attach_vehicle_observer(CSVVehicleObserver(outdir / "veh.csv"))
    
            for mobility_service in self._avalaible_PublicTransport_means:
                service = PublicTransportMobilityService(mobility_service)
                mmgraph.layers[mobility_service+"Layer"].add_mobility_service(service)
                service.attach_vehicle_observer(CSVVehicleObserver(outdir / "veh.csv"))
                    
            avalaible_PublicTransport_means = list(self._avalaible_PublicTransport_means)
            if len(avalaible_PublicTransport_means) > 0 :
                avalaible_PublicTransport_means = [str(x) + "Layer" for x in avalaible_PublicTransport_means]
                mmgraph.connect_inter_layers(avalaible_PublicTransport_means, 100)

            
            
            
            demand_file=self._inputs[i]
            demand = CSVDemandManager(demand_file)
            
            demand.add_user_observer(CSVUserObserver(outdir / "user.csv"), user_ids="all")

            flow_motor = MFDFlowMotor(outfile=outdir / "flow.csv")
            flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["CAR", "BUS", "TRAM", "METRO"], calculate_V_MFD))

            travel_decision = LogitDecisionModel(mmgraph, outfile=outdir / "path.csv")

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

        self._outputs_produced = True
    
        logger.info("outputs produced.")

        


    # getters

    def get_input_path():
        # check
        if len(self._input_path) == 0: logger.info("_input_path is null.")
        return self._input_path

    
    def get_output_path():
        # check
        if len(self._output_path) == 0: logger.info("_output_path is null.")
        return self._output_path

    
    def get_inputs():
        # check
        if len(self._inputs) == 0: logger.info("_inputs is empty.")
        return self._inputs

    
    def get_outputs():
        # check
        if len(self._outputs) == 0: logger.info("_outputs is empty.")
        return self._outputs