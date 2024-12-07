import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.retail_model import initialize_registry
from models.map import map_network
import yaml
import numpy as np
import torch
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
        
        # Validate environment properties
        for prop_key, prop_value in config["state"]["environment"].items():
            if "shape" not in prop_value:
                if isinstance(prop_value["value"], list):
                    prop_value["shape"] = [len(prop_value["value"])]
                else:
                    prop_value["shape"] = [1]
            
            # Ensure dtype is specified
            if "dtype" not in prop_value:
                if isinstance(prop_value["value"], (int, np.integer)):
                    prop_value["dtype"] = "int"
                elif isinstance(prop_value["value"], (float, np.floating)):
                    prop_value["dtype"] = "float"
                elif isinstance(prop_value["value"], str):
                    prop_value["dtype"] = "str"
                else:
                    prop_value["dtype"] = "float"
            
            # Ensure learnable is specified
            if "learnable" not in prop_value:
                prop_value["learnable"] = False
                
        # Format substeps
        formatted_substeps = {}
        for substep_key, substep_dict in config["substeps"].items():
            formatted_substeps[substep_key] = {
                "name": substep_dict["name"],
                "description": substep_dict.get("description", ""),
                "active_agents": substep_dict.get("active_agents", []),
                "observation": substep_dict.get("observation", {}),
                "policy": substep_dict.get("policy", {}),
                "transition": substep_dict.get("transition", {})
            }
        config["substeps"] = formatted_substeps

        # Format simulation metadata
        if "simulation_metadata" not in config:
            config["simulation_metadata"] = {}
        config["simulation_metadata"].update({
            "num_substeps_per_step": len(formatted_substeps),
            "num_steps_per_episode": config["simulation_metadata"].get("num_steps_per_episode", 50),
            "num_episodes": config["simulation_metadata"].get("num_episodes", 1)
        })

        # Format agents section
        for agent_type, agent_config in config["state"]["agents"].items():
            for prop_name, prop_config in agent_config["properties"].items():
                if "shape" not in prop_config:
                    if isinstance(prop_config.get("value", []), list):
                        prop_config["shape"] = [len(prop_config["value"])]
                    else:
                        prop_config["shape"] = [agent_config["number"], 1]
                if "dtype" not in prop_config:
                    prop_config["dtype"] = "float"
                if "learnable" not in prop_config:
                    prop_config["learnable"] = False

        return config

def parse_args():
    parser = argparse.ArgumentParser(description='Run retail simulation')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='Path to configuration file')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load and format configuration
    config_data = load_config(args.config)
    
    print(config_data)
    # Initialize registry
    registry = Registry()
    registry.register(read_from_file, "read_from_file", "initialization")
    registry.register(grid_network, "grid", key="network")
    registry.register(map_network, "map", key="network")

    # Initialize the runner
    runner = Runner(config=config_data, registry=registry)
    
    try:
        # Initialize the simulation
        runner.init()
        
        # Run the simulation
        runner.step()
        
        print("Simulation completed successfully!")
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise