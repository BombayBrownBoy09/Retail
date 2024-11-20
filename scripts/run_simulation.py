import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.retail_model import initialize_registry
import yaml
import numpy as np
import argparse
from tqdm import trange
from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_config, read_from_file, grid_network

import torch
import networkx as nx
import osmnx as ox

def map_network(params=None):
    """
    Generates a graph and its adjacency matrix based on the given parameters.

    Parameters:
    - params (dict): Dictionary with optional keys:
        - "coordinates" (tuple): Latitude and longitude of the central point.
        - "distance" (int): Radius in meters to generate the graph.

    Returns:
    - graph (networkx.Graph): The generated graph.
    - adjacency_matrix (torch.Tensor): The adjacency matrix of the graph as a PyTorch tensor.
    """
    # Default parameters
    default_params = {
        "coordinates": (40.78264403323726, -73.96559413265355),  # Central Park
        "distance": 550,  # Default radius in meters
    }

    # Use default parameters if not provided
    if params is None:
        params = default_params
    else:
        params = {**default_params, **params}  # Merge defaults with provided params

    coordinates = params["coordinates"]
    distance = params["distance"]

    # Generate the graph using osmnx
    graph = ox.graph_from_point(
        coordinates, dist=distance, simplify=True, network_type="walk"
    )

    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(graph).todense()

    return graph, torch.tensor(adjacency_matrix)


def load_config(config_path):
    """
    Load configuration from config.yaml and format it for the runner.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        # Format substeps as required by Runner
        substeps = {}
        for substep_key, substep_dict in config["substeps"].items():
            step_type = substep_dict['type']
            step_name = substep_dict['name']
            substeps[substep_key] = {  # Keep the dictionary format with original keys
                "name": step_name,
                "active_agents": ["consumer"],  # Default agent type
                "type": step_type
            }
        config["substeps"] = substeps

        # Ensure simulation metadata is properly formatted
        if "simulation_metadata" not in config:
            config["simulation_metadata"] = {}
        config["simulation_metadata"]["num_substeps_per_step"] = len(substeps)
        if "num_steps_per_episode" not in config["simulation_metadata"]:
            config["simulation_metadata"]["num_steps_per_episode"] = config["simulation"]["steps"]
        if "num_episodes" not in config["simulation_metadata"]:
            config["simulation_metadata"]["num_episodes"] = 1
        # Initialize agents state
        agents = []
        for i in range(config["consumers"]["count"]):
            agent = {
                "id": i,
                "budget": np.random.uniform(
                    config["consumers"]["budget_range"]["min"],
                    config["consumers"]["budget_range"]["max"]
                ),
                "price_sensitivity": np.random.uniform(
                    config["consumers"]["price_sensitivity_range"]["min"],
                    config["consumers"]["price_sensitivity_range"]["max"]
                ),
                "purchase_frequency": np.random.randint(
                    config["consumers"]["purchase_frequency_range"]["min"],
                    config["consumers"]["purchase_frequency_range"]["max"]
                )
            }
            agents.append(agent)
        # Create initial state structure
        initial_state = {
            "agents": {
                "name": "agents",
                "value": agents
            },
            "environment": {
                "properties": {
                "restock_threshold": {
                    "name": "restock_threshold",
                    "value": config["state"]["environment"]["restock_threshold"]
                },
                    "restock_quantity": {
                        "name": "restock_quantity",
                        "value": config["state"]["environment"]["restock_quantity"]
                    },
                    "products": {
                        "name": "products",
                        "value": config["state"]["environment"]["products"]
                    }
                }
            }
        }
        # Add promotions to state
        for promo in config["promotions"]:
            # Access the "value" key to retrieve the list of products
            for product in config["state"]["environment"]["products"]["value"]:
                if product["id"] == promo["id"]:
                    product["promotion"] = 1 - promo["discount"]


        # Add initial state to config
        config["initial_state"] = initial_state
        return config


def parse_args():
    parser = argparse.ArgumentParser(description='Run retail simulation')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='/Users/bhargav/random_projects/Retail/config.yaml')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    # Load and format configuration
    config_data = load_config(args.config)
    # Initialize registry
    registry = Registry()
    registry.register(read_from_file, "read_from_file", "initialization")
    registry.register(grid_network, "grid", key="network")
    registry.register(map_network, "map", key="network")
    # Initialize the runner
    runner = Runner(config=config_data, registry=registry)
    # Initialize the simulation
    runner.init()
    # Run the simulation
    runner.step()

    print("Simulation completed successfully!")