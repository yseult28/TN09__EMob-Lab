
# dependencies

from pathlib import Path

import json

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from shapely.geometry import mapping
from shapely import LineString

import plotly.graph_objects as go

import json
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go

from mnms.mobility_service.on_demand import OnDemandDepotMobilityService
from mnms.io.graph import load_graph, load_odlayer, save_odlayer
from mnms.travel_decision.logit import LogitDecisionModel, ModeCentricLogitDecisionModel
from mnms.generation.layers import generate_bbox_origin_destination_layer, generate_matching_origin_destination_layer
from mnms.generation.layers import generate_bbox_origin_destination_layer
from mnms.io.graph import save_transit_link_odlayer, load_transit_links
from mnms.graph.layers import OriginDestinationLayer

from shapely.ops import unary_union
from shapely.geometry import Point
import os
from shapely.geometry import MultiPolygon, box
from shapely.geometry import shape


import logging



# log
logger = logging.getLogger(__name__)

class NetworkManager:
    """
    Creates useful dataframes to harness. Gives access to display methods.

    Attributes
    ----------
    _path : string
        Path to the graph file.
    _old_crs : string
        network's original projection system.
    _crs : string
        network's actual projection system .
    _network : dict
        dictionnary that holds the json file's content.
    _nodes : gpd.GeoDataFrames
        GeoDataframe with the following variables : NODE (string, node's name) , GEOMETRY (point, node's position).
    _sections : gpd.GeoDataFrames
        GeoDataframe with the following variables : SECTION (string, section's name) , LENGTH (float, section's length), #ZONE (string, section's reservoir), GEOMETRY (LineString, section's position).
    _links : pd.DataFrames
        Dataframe with the following variables : SECTION (strig, section's name) , NODES (string,the two nodes forming the #section, in the following format, "upstream downstream").
    """

   # Constructor
    
    def __init__(self,path:str="", crs:str=""):
        """
        NetworkManager's constructor.
    
        Parameters
        ----------
        path : string
            Path to network's file.
        crs : string
            Projection system used in the network's file.

        Returns
        -------
        None
        """
        
        # check 
        if not path or path.strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not crs or crs.strip() == "": 
            logger.error("Invalid or null projection system.")
            raise ValueError("Invalid or null projection system.")

        # assignment
        self._path = path # path to the json file 
        self._old_crs = crs # network's original projection system
        self._crs = crs # network's actual projection system 
        
        self._network = None # dictionnary that holds the json file's content 
        
        self._nodes = None #  GeoDataframe with the following variables : NODE (string, node's name) , GEOMETRY (point, node's position) 
        self._sections = None #  GeoDataframe with the following variables : SECTION (string, section's name) , LENGTH (float, section's length), #ZONE (string, section's reservoir), GEOMETRY (LineString, section's position) 
        self._links = [] #  Dataframe with the following variables : SECTION (strig, section's name) , NODES (string,the two nodes forming the #section, in the following format, "upstream downstream") 

        self.load_network_file()

        logger.info("NetworkManager initialized with path: %s and CRS: %s", path, crs)

        
    # configuration's methods 
    
    # sets up a new network by accessing a new file with its projection system
    def set_new_network(self, path, crs):
        """
        Sets up a new network by accessing a new file with its projection system.
    
        Parameters
        ----------
        path : string
            Path to network's file.
        crs : string
            Projection system used in the network's file.

        Returns
        -------
        None
        """
        
        # check 
        if not path or path.strip() == "": 
            logger.error("Invalid or null path.")
            raise ValueError("Invalid or null path.")
        if not crs or crs.strip() == "": 
            logger.error("Invalid or null projection system.")
            raise ValueError("Invalid or null projection system.")
            
        # assignment
        self._path = path # path to the json file 
        self._old_crs = crs # network's original projection system
        self._crs = crs # network's actual projection system 
        
        self._network = None # dictionnary that holds the json file's content 
        
        self._nodes = None #  GeoDataframe with the following variables : NODE (string, node's name) , GEOMETRY (point, node's position) 
        self._sections = None #  GeoDataframe with the following variables : SECTION (string, section's name) , LENGTH (float, section's length), #ZONE (string, section's reservoir), GEOMETRY (LineString, section's position) 
        self._links = [] #  Dataframe with the following variables : SECTION (strig, section's name) , NODES (string,the two nodes forming the #section, in the following format, "upstream downstream") 

        self.load_network_file()
        
        logger.info(f"path change to {self._path}.")
        logger.info(f"projection system change to {self._crs}.")
        logger.info("Call load_network_file() to load the new network.")

    

    def change_crs(self, crs):
        """
        Sets up a new projection system.
    
        Parameters
        ----------
        crs : string
            Projection system used in the network's file.

        Returns 
        -------
        None
        """
        # check 
        if not crs or crs.strip() == "": 
            logger.error("Invalid or null projection system.")
            raise ValueError("Invalid or null projection system.")
            
        self._crs = crs
        logger.info(f"projection system change to {self._crs}, call the create's method to get the new GeoDataFrames.")


    # resets the projection system 
    def reset_crs(self):
        """
        Resets the projection system to the original one.
    
        Parameters
        ----------
        crs : string
            Projection system used in the network's file.

        Returns
        -------
        None
        """
        
        self._crs = self._old_crs
        self._nodes = self._nodes.to_crs(self._crs)
        self._sections = self._sections.to_crs(self._crs)
        logger.info(f"projection system change to {self._crs}, GeoDataFrames updated.")

    
    def load_network_file(self):
        """
        Loads the network's file.
    
        Parameters
        ----------
        crs : string
            Projection system used in the network's file.

        Returns
        -------
        None
        """
        
        try:
            with open(self._path, 'r') as file:
                self._network = json.load(file)
                logger.info("Network file loaded successfully from %s", self._path)
        except Exception as e:
            logger.exception("Failed to load network file: %s", e)
            raise  

    
    # creation's methods

    def create_nodes(self):
        """
        Creates the nodes dataframe.
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if not self._network: 
            logger.error("_network is empty, call load_network_file().")
            raise ValueError("_network is empty, call load_network_file().")

        # car
        buffer = pd.DataFrame(self._network["ROADS"]["NODES"].values())
        buffer["position"] = buffer["position"].astype(str).str.extract(r'\[(.*?)\]')
        buffer[["x", "y"]] = buffer["position"].str.split(",", expand=True).astype(float)
        buffer["GEOMETRY"] = gpd.points_from_xy(buffer["x"], buffer["y"])
        buffer.drop(columns=["position", "x", "y"], inplace=True)
        buffer.rename(columns={"id":"NODE"},inplace=True)
        car_gdf = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=self._crs)
        
        # public transport
        buffer = pd.DataFrame(self._network["ROADS"]["STOPS"].values())
        if len(buffer) != 0 :
            buffer["NODE"] = buffer["section"] + '_' + buffer["id"]
            buffer["x"] = buffer["absolute_position"].apply(lambda pos: float(pos[0]))
            buffer["y"] = buffer["absolute_position"].apply(lambda pos: float(pos[1]))

            buffer["GEOMETRY"] = gpd.points_from_xy(buffer["x"], buffer["y"])
            buffer.drop(columns=["x", "y","section","id","relative_position","absolute_position"], inplace=True)
            publictransport_gdf = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=self._crs)
            #logger.info(publictransport_gdf.sample(3))

            self._nodes = pd.concat([car_gdf, publictransport_gdf], ignore_index=True)
        else : 
            self._nodes = car_gdf

        self._nodes = gpd.GeoDataFrame(  self._nodes, geometry="GEOMETRY", crs=self._crs)
 
        logger.info("_nodes created.")

    def create_sections(self):
        """
        Creates the sections dataframe.
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if not self._network: 
            logger.error("_network is empty, call load_network_file().")
            raise ValueError("_network is empty, call load_network_file().")

        # car
        geom_dict = self._nodes.set_index("NODE")["GEOMETRY"].to_dict()
        buffer = pd.DataFrame(self._network["ROADS"]["SECTIONS"].values())
        lines = []
        for i, row in buffer.iterrows():
            up_geom = geom_dict.get(row["upstream"])
            down_geom = geom_dict.get(row["downstream"])
            if up_geom is None or down_geom is None:
                continue
            lines.append({
                "SECTION": row["id"],
                "LENGTH": row["length"],
                "ZONE": row["zone"],
                "GEOMETRY": LineString([up_geom, down_geom])
            })
        self._sections = gpd.GeoDataFrame(lines, geometry="GEOMETRY", crs=self._crs)
        

        # public transport
        buffer = pd.DataFrame(self._network["ROADS"]["STOPS"].values())
        if len(buffer) != 0 :
            buffer["NODE"] = buffer["section"] + '_' + buffer["id"]
            nodes_dict = self._sections.set_index("SECTION")[["LENGTH","ZONE"]].to_dict()
            buffer["LENGTH"] = buffer.apply(lambda row : nodes_dict["LENGTH"].get(row["section"]) * row["relative_position"], axis=1)
            buffer["ZONE"] = buffer.apply(lambda row : nodes_dict["ZONE"].get(row["section"]), axis=1)
        
            pos_dict = self._nodes.set_index("NODE")["GEOMETRY"]
            buffer["GEOMETRY"] = buffer.apply(lambda row : pos_dict.get(row["NODE"]), axis=1)
        
            # extracts every layers except the car one
            publictransport_layers_buffer = pd.DataFrame(self._network["LAYERS"][1:])
            #logger.info(publictransport_layers_buffer["LINES"])

            sections = []
            nodes1 = []
            nodes2 = []

            #for layer_id, layer_value in publictransport_layers_buffer.items():
                #for line in layer_value["LINES"]:
            for publictransport_layer in publictransport_layers_buffer["LINES"]:
                for line in publictransport_layer:
                    #logger.info(line)
                    ID = line["ID"]
                    STOPS = line["STOPS"]

                    for i in range(len(STOPS)-1):
                        sections.append(f"{ID}_{STOPS[i]} {ID}_{STOPS[i+1]}")
                        nodes1.append(f"{ID}_{STOPS[i]}")
                        nodes2.append(f"{ID}_{STOPS[i+1]}")

            df = pd.DataFrame({
                "SECTION": sections,
                "NODES1": nodes1,
                "NODES2": nodes2
            })

            rel_dict = buffer.set_index("NODE")[["LENGTH","ZONE","GEOMETRY"]]
            #logger.info(rel_dict["LENGTH"])
            df["LENGTH"] = df.apply(lambda row : rel_dict["LENGTH"].get(row["NODES2"]) -  rel_dict["LENGTH"].get(row["NODES1"]), axis = 1)
            df["ZONE"] = df.apply(lambda row : rel_dict["ZONE"].get(row["NODES1"]), axis = 1)
            df["GEOMETRY"] = df.apply(lambda row: LineString([rel_dict["GEOMETRY"].get(row["NODES1"]), rel_dict["GEOMETRY"].get(row["NODES2"])]), axis=1)
            df.drop(columns=["NODES1","NODES2"], inplace=True)

            self._sections = pd.concat([self._sections, df], ignore_index=True)
        else : 
            self._sections = self._sections
        self._sections = gpd.GeoDataFrame(self._sections, geometry="GEOMETRY", crs=self._crs)
        
        logger.info("_sections created.")

    def create_links(self):
        """
        Creates the links dataframe.
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if not self._network: 
            logger.error("_network is empty, call load_network_file().")
            raise ValueError("_network is empty, call load_network_file().")

        # car
        buffer = pd.DataFrame(self._network["ROADS"]["SECTIONS"].values())
        buffer = buffer.rename(columns={"id":"SECTION"})
        #logger.info(buffer.sample(1))
        buffer["NODES"] = buffer.apply(lambda row : str(row["upstream"]) + ' ' + str(row["downstream"]),axis=1)
        buffer = buffer.drop(columns=["upstream","downstream"])
        self._links = buffer

        # public transport
        buffer = pd.DataFrame(self._network["ROADS"]["STOPS"].values())
        if len(buffer) != 0 : 
            # extracts every layers except the car one
            publictransport_layers_buffer = pd.DataFrame(self._network["LAYERS"][1:])

            sections = []
            nodes1 = []
            nodes2 = []

            for publictransport_layer in publictransport_layers_buffer["LINES"]:
                for line in publictransport_layer:
                    ID = line["ID"]
                    STOPS = line["STOPS"]

                    for i in range(len(STOPS)-1):
                        sections.append(f"{ID}_{STOPS[i]} {ID}_{STOPS[i+1]}")

            df = pd.DataFrame({
                "SECTION": sections,
                "NODES": sections
            })
            self._links = pd.concat([self._links, df], ignore_index=True)
        else : 
            self._links = self._links

        self._links.drop(columns=["length", "zone"], inplace=True)
        
        logger.info("_links created.")

    def create_network_dfs(self):
        """
        Creates the nodes, sections and links dataframe.
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        self.create_nodes()
        self.create_sections()
        self.create_links()

    
    def create_grid_layers(self, nx=14, ny=16, dist_connection=1000, directory=""):
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

        Returns
        -------
        None
        """

        # check
        if not directory or directory.strip() == "" : 
            logger.info("Null or invalid directory.")
            raise ValueError("Null or invalid directory.")

        NX = nx
        NY = ny
        DIST_CONNECTION = dist_connection

        mmgraph = load_graph(self._path)
        graph_name = Path(self._path).stem

        odlayer = generate_bbox_origin_destination_layer(mmgraph.roads, NX, NY)
        mmgraph.add_origin_destination_layer(odlayer)
        mmgraph.connect_origindestination_layers(DIST_CONNECTION)
        
        save_odlayer(odlayer, directory + f"/{graph_name}__GridOdLayer__{NX}_{NY}_{DIST_CONNECTION}.json")
        save_transit_link_odlayer(mmgraph, directory + f"/{graph_name}__GridTransitLink__{NX}_{NY}_{DIST_CONNECTION}.json")

        logger.info(f"Od layer and transit link files created and saved at {directory} with : nx = {nx}, ny : {ny}, dist_connection : {dist_connection}.")

    def create_layer_layer(self,dist_connection=0, directory=""):
        """
        Create an od layer and a transit layer file with the restricted graph at graph file in the demand area and the dist_connection and saved them at directory.
        All the file must have the same projection system.
    
        Parameters
        ----------
        dist_connection : int
            Used to connect each layer with the graph.
        directory : 
            Directory to save the files.

        Returns
        -------
        None
        """

        # check
        if not dist_connection or dist_connection == 0 : 
            logger.info("Invalid or null dist connection.")
            raise ValueError("Invalid or null dist connection.")
        if not directory or directory.strip()=="":
            logger.info("Invalid or null directory.")
            raise ValueError("Invalid or null directory.")  

        network = []
        with open(self._path, 'r') as graph_file:
                    network = json.load(graph_file)

        buffer = pd.DataFrame(network["ROADS"]["NODES"].values()).copy()
        buffer["position"] = buffer["position"].astype(str).str.extract(r'\[(.*?)\]')
        buffer[["x", "y"]] = buffer["position"].str.split(",", expand=True).astype(float)
        buffer["GEOMETRY"] = gpd.points_from_xy(buffer["x"], buffer["y"])
        #buffer.drop(columns=["position", "x", "y"], inplace=True)
        buffer.drop(columns=["position"], inplace=True)
        buffer.rename(columns={"id":"NODE"},inplace=True)
        inner_graph = gpd.GeoDataFrame(buffer, geometry="GEOMETRY", crs=self._crs)

        # Générer les dictionnaires
        origins = {f"ORIGIN_{i}": [row.GEOMETRY.x, row.GEOMETRY.y] for i, row in inner_graph.iterrows()}
        destinations = {f"DESTINATION_{i}": [row.GEOMETRY.x, row.GEOMETRY.y] for i, row in inner_graph.iterrows()}

        odlayer = OriginDestinationLayer()

        for i, row in inner_graph.iterrows():
            odlayer.create_origin_node(f"ORIGIN_{i}", [row.GEOMETRY.x, row.GEOMETRY.y])

        for i, row in inner_graph.iterrows():
            odlayer.create_destination_node(f"DESTINATION_{i}",[row.GEOMETRY.x, row.GEOMETRY.y])

    
        if "m" in network['LAYERS'][0]['MAP_ROADDB']['LINKS'].values() : print("test")

        mmgraph = load_graph(self._path)
        graph_name = Path(self._path).stem

        mmgraph.add_origin_destination_layer(odlayer)
        mmgraph.connect_origindestination_layers(dist_connection)
        
        save_odlayer(odlayer, directory + f"/{graph_name}__LayerOdLayer__{dist_connection}.json")
        save_transit_link_odlayer(mmgraph, directory + f"/{graph_name}__LayerTransitLink__{dist_connection}.json")

        logger.info(f"Od layer and transit link files created and saved at {directory} with : demand area : Restricted, graph : {self._path}, dist_connection : {dist_connection}.")



    def restrict_graph(self, demand_area_path: str, crs: str, output_path: str, distance=0, name="test_graph"):
        """
        Restrict a graph file to the area of the file saved at demand_area_path.

        Parameters
        ----------
        demand_area_path : str
            Relative path to the area file.
        crs : str
            Projection system as string.
        output_path : str
            Relative path to save the new graph.
        distance : int
            Distance buffer.
        name : str
            Name of the new graph file.
            
        Returns
        -------
        None
        """
        # --- 1. Read the demand area and project to target CRS ---
        demand_area = gpd.read_file(demand_area_path)
        global_demand_area = unary_union(demand_area.geometry)
   
        global_demand_area = shape(global_demand_area)
        global_demand_area = global_demand_area.buffer( distance )
        
        global_demand_area_gdf = gpd.GeoDataFrame(
            geometry=[global_demand_area],
            crs=demand_area.crs
        ).to_crs(crs)

        # Utility function to check if a point is inside the demand area
        def is_in_demand_area(coords):
            point = Point(coords)
            return global_demand_area_gdf.geometry.iloc[0].contains(point)

        # --- 2. Load the graph JSON ---
        with open(self._path, 'r', encoding='utf-8') as f:
            graph = json.load(f)

        # --- 3. Filter ROADS/NODES ---
        new_nodes = {nid: nd for nid, nd in graph['ROADS']['NODES'].items() if is_in_demand_area(nd['position'])}
        graph['ROADS']['NODES'] = new_nodes

        # --- 4. Filter SECTIONS ---
        new_sections = {sid: sd for sid, sd in graph['ROADS']['SECTIONS'].items()
                        if sd.get('upstream') in new_nodes and sd.get('downstream') in new_nodes}
        graph['ROADS']['SECTIONS'] = new_sections

        # --- 5. Filter ZONES/RES sections ---
        if 'RES' in graph['ROADS']['ZONES']:
            zone_res = graph['ROADS']['ZONES']['RES']
            zone_res['sections'] = [s for s in zone_res['sections'] if s in new_sections]
            minx, miny, maxx, maxy = global_demand_area_gdf.total_bounds

            zone_res["contour"] = [
            [minx, miny],  # bas-gauche
            [maxx, miny],  # bas-droit
            [maxx, maxy],  # haut-droit
            [minx, maxy],  # haut-gauche
        ]


        # --- 6. Filter LAYERS[0] ---
        layer0 = graph['LAYERS'][0]

        # NODES: keep only those whose ID is in new_nodes and reindex with consecutive integers
        kept_nodes = [nd for nd in layer0['NODES'] if nd['ID'] in new_nodes]
        new_layer_nodes = []
    #new_layer_nodes = {}
        for idx, nd in enumerate(kept_nodes):
            #new_layer_nodes[str(idx)] = nd  # copy value as-is
            new_layer_nodes.append(nd)
        layer0['NODES'] = new_layer_nodes

        # LINKS: keep only those whose ID is in new_sections and reindex with consecutive integers
        kept_links = [lk for lk in layer0['LINKS'] if lk['ID'] in new_sections]
        new_layer_links = []
        #new_layer_links = {}
        for idx, lk in enumerate(kept_links):
            #new_layer_links[str(idx)] = lk  # copy value as-is
            new_layer_links.append(lk)
            if lk["ID"] == "m" : print("test")
        layer0['LINKS'] = new_layer_links

    
    
        # MAP_ROADDB: rebuild NODES and LINKS with ID as both key and value
        map_roaddb = layer0.get('MAP_ROADDB', {})
        #map_roaddb['NODES'] = {nd['ID']: nd['ID'] for nd in new_layer_nodes.values()}
        #map_roaddb['LINKS'] = {lk['ID']: lk['ID'] for lk in new_layer_links.values()}
        map_roaddb['NODES'] = {nd['ID']: nd['ID'] for nd in new_layer_nodes}
        map_roaddb['LINKS'] = {lk['ID']: [lk['ID']] for lk in new_layer_links}
        layer0['MAP_ROADDB'] = map_roaddb
    
        if "m" in map_roaddb['LINKS'] : print("test")

        # Update LAYERS[0]
        graph['LAYERS'][0] = layer0

    
        #print(graph['LAYERS'][0]['MAP_ROADDB']['LINKS'])
        #print(graph)

        # --- 7. Save the restricted graph ---
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{name}_{distance}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=4, ensure_ascii=False)

        print(f"Restricted graph saved to: {output_path}")

    # display

    def plot_nodes_map(
        self,
        marker_size=5,
        marker_color="red",
        zoom=10,
        width=900,
        height=600
    ):
        """
        Display nodes stored in self._nodes on an interactive map.

        The function plots each node as a point using Plotly scattermap,
        with a street-style basemap. Node names are displayed and shown
        in the hover information.

        Parameters
        ----------
         marker_size : int, optional
            Size of the node markers. Default is 5.
        marker_color : string, optional
            Color of the node markers. Default is "red".
        zoom : int or float, optional
            Initial zoom level of the map. Default is 10.
        width : int, optional
            Width of the figure in pixels. Default is 900.
        height : int, optional
            Height of the figure in pixels. Default is 600.
        """

        # ------------------------------------------------------------------
        # Reproject GeoDataFrame to WGS84 (lat / lon)
        # ------------------------------------------------------------------
        gdf = self._nodes.to_crs("EPSG:4326")

        # ------------------------------------------------------------------
        # Extract longitude and latitude from point geometries
        # ------------------------------------------------------------------
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y

        # ------------------------------------------------------------------
        # Create the scattermap figure for nodes
        # ------------------------------------------------------------------
        fig = go.Figure(
            go.Scattermap(
                lat=gdf["lat"],
                lon=gdf["lon"],
                mode="markers",
                hovertext=gdf["NODE"],
                marker=dict(
                    size=marker_size,
                    color=marker_color
                ),
                hovertemplate="<b>Node :</b> %{hovertext}<extra></extra>"
            )
        )

        # ------------------------------------------------------------------
        # Configure map layout
        # ------------------------------------------------------------------
        fig.update_layout(
            width=width,
            height=height,
            map=dict(
                style="streets",
                center=dict(
                    lat=gdf["lat"].mean(),
                    lon=gdf["lon"].mean()
                ),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # ------------------------------------------------------------------
        # Display the figure
        # ------------------------------------------------------------------
        fig.show()

    def plot_sections_map(
        self,
        line_width=3,
        zoom=10,
        width=900,
        height=600
    ):
        """
        Display sections stored in self._sections on an interactive map.

        The function plots each section as a line using Plotly scattermap,
        with a street-style basemap. Section names and lengths are shown
        in the hover information.

        Parameters
        ----------
        line_width : int, optional
            Width of the section lines. Default is 3.
        zoom : int or float, optional
            Initial zoom level of the map. Default is 10.
        width : int, optional
            Width of the figure in pixels. Default is 900.
        height : int, optional
            Height of the figure in pixels. Default is 600.
        """

        # ------------------------------------------------------------------
        # Reproject GeoDataFrame to WGS84 (lat / lon)
        # ------------------------------------------------------------------
        gdf = self._sections.to_crs("EPSG:4326")

        # ------------------------------------------------------------------
        # Initialize the Plotly figure
        # ------------------------------------------------------------------
        fig = go.Figure()

        # ------------------------------------------------------------------
        # Add each section as a line on the map
        # ------------------------------------------------------------------
        for _, row in gdf.iterrows():
            geom = row[gdf.geometry.name]
            lons, lats = geom.xy

            fig.add_trace(
                go.Scattermap(
                    lon=list(lons),
                    lat=list(lats),
                    mode="lines",
                    line=dict(width=line_width),
                    name=row["SECTION"],
                    text=[row["SECTION"]] * len(lons),
                    hovertemplate=(
                        "<b>Section :</b> %{text}<br>"
                        "<b>Length :</b> " + str(row["LENGTH"]) +
                        "<extra></extra>"
                    )
                )
            )

        # ------------------------------------------------------------------
        # Configure map layout
        # ------------------------------------------------------------------
        fig.update_layout(
            width=width,
            height=height,
            map=dict(
                style="streets",
                center=dict(
                    lat=gdf.geometry.centroid.y.mean(),
                    lon=gdf.geometry.centroid.x.mean()
                ),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # ------------------------------------------------------------------
        # Display the figure
        # ------------------------------------------------------------------
        fig.show()

    def plot_origins_json(self,
        json_file,
        crs="EPSG:2154",  # CRS du fichier JSON
        marker_size=5,
        marker_color="blue",
        zoom=10,
        width=900,
        height=600
    ):
        """
        Display origins from a JSON file on an interactive map.

        Parameters
        ----------
        json_file : str
            Path to the JSON file.
        crs : str, optional
            CRS of the coordinates in the JSON file. Default is "EPSG:2154".
        marker_size : int, optional
            Size of the markers. Default is 5.
        marker_color : str, optional
            Color of the markers. Default is "blue".
        zoom : int or float, optional
            Initial zoom level of the map. Default is 10.
        width : int, optional
            Width of the figure in pixels. Default is 900.
        height : int, optional
            Height of the figure in pixels. Default is 600.
        """

        # ------------------------------------------------------------------
        # Load JSON
        # ------------------------------------------------------------------
        with open(json_file, "r") as f:
            data = json.load(f)

        origins = []
        for key, coords in data["ORIGINS"].items():
            x, y = coords[0], coords[1]
            origins.append(Point(x, y))

        # ------------------------------------------------------------------
        # Create GeoDataFrame
        # ------------------------------------------------------------------
        gdf = gpd.GeoDataFrame(geometry=origins, crs=crs)
    
        # ------------------------------------------------------------------
        # Reproject to WGS84 (EPSG:4326) for Plotly
        # ------------------------------------------------------------------
        gdf = gdf.to_crs("EPSG:4326")
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y

        # ------------------------------------------------------------------
        # Plot with Plotly
        # ------------------------------------------------------------------
        fig = go.Figure(
            go.Scattermap(
                lat=gdf["lat"],
                lon=gdf["lon"],
                mode="markers",
                marker=dict(size=marker_size, color=marker_color),
                hovertext=[f"Origin {i+1}" for i in range(len(gdf))],
                hovertemplate="<b>%{hovertext}</b><extra></extra>"
            )
        )

        fig.update_layout(
            width=width,
            height=height,
            map=dict(
                style="streets",
                center=dict(
                    lat=gdf["lat"].mean(),
                    lon=gdf["lon"].mean()
                ),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        fig.show()

    def plot_destinations_json(self,
        json_file,
        crs="EPSG:2154",
        marker_size=5,
        marker_color="red",
        zoom=10,
        width=900,
        height=600
    ):
        """
        Display destinations from a JSON file on an interactive map.

        Parameters
        ----------
        json_file : str
            Path to the JSON file.
        crs : str, optional
            CRS of the coordinates in the JSON file. Default is "EPSG:2154".
        marker_size : int, optional
            Size of the markers. Default is 5.
        marker_color : str, optional
            Color of the markers. Default is "red".
        zoom : int or float, optional
            Initial zoom level of the map. Default is 10.
        width : int, optional
            Width of the figure in pixels. Default is 900.
        height : int, optional
            Height of the figure in pixels. Default is 600.
        """

        # ------------------------------------------------------------------
        # Load JSON
        # ------------------------------------------------------------------
        with open(json_file, "r") as f:
            data = json.load(f)

        destinations = []
        for key, coords in data["DESTINATIONS"].items():
            x, y = coords[0], coords[1]
            destinations.append(Point(x, y))

        # ------------------------------------------------------------------
        # Create GeoDataFrame
        # ------------------------------------------------------------------
        gdf = gpd.GeoDataFrame(geometry=destinations, crs=crs)

        # ------------------------------------------------------------------
        # Reproject to WGS84 (EPSG:4326) for Plotly
        # ------------------------------------------------------------------
        gdf = gdf.to_crs("EPSG:4326")
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y

        # ------------------------------------------------------------------
        # Plot with Plotly
        # ------------------------------------------------------------------
        fig = go.Figure(
            go.Scattermap(
                lat=gdf["lat"],
                lon=gdf["lon"],
                mode="markers",
                marker=dict(size=marker_size, color=marker_color),
                hovertext=[f"Destination {i+1}" for i in range(len(gdf))],
                hovertemplate="<b>%{hovertext}</b><extra></extra>"
            )
        )

        fig.update_layout(
            width=width,
            height=height,
            map=dict(
                style="streets",
                center=dict(
                    lat=gdf["lat"].mean(),
                    lon=gdf["lon"].mean()
                ),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        fig.show()

    
    # getters 

    def get_path(self):
        return self._path

    def get_crs(self):
        return self._crs

    def get_nodes(self):
        if self._nodes.empty : logger.info("nodes is empty.")
        return self._nodes

    def get_sections(self):
        if self._sections.empty : logger.info("sections is empty.")
        return self._sections

    def get_links(self):
        if self._sections.empty : logger.info("links is empty.")
        return self._links








        
        
