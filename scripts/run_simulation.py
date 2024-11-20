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

def load_config(config_path):
    """
    Load configuration from config.yaml and format it for the runner.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        # Format substeps as required by Runner
        substeps = []
        for substep_dict in config["substeps"]:
            substep = {}
            step_type = substep_dict['type']
            step_name = substep_dict['name']
            substep = {
                step_name: {
                    "active_agents": ["consumer"],  # Default agent type
                    "type": step_type
                }
            }
            substeps.append(substep)
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
                        "value": config["environment"]["restock_threshold"]
                    },
                    "restock_quantity": {
                        "name": "restock_quantity",
                        "value": config["environment"]["restock_quantity"]
                    },
                    "products": {
                        "name": "products",
                        "value": config["environment"]["products"]
                    }
                }
            }
        }
        # Add promotions to state
        for promo in config["promotions"]:
            for product in initial_state["environment"]["properties"]["products"]["value"]:
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
    # registry.register(map_network, "map", key="network")
    # Initialize the runner
    runner = Runner(config=config_data, registry=registry)
    # Initialize the simulation
    runner.init()
    # Run the simulation
    runner.step()

    print("Simulation completed successfully!")