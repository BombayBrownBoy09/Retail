import argparse
from tqdm import trange
from agent_torch.core import Runner
from agent_torch.core.helpers import read_config
from models.retail_model import initialize_registry

print(":: execution started")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to YAML config file", required=True)
config_file = parser.parse_args().config

# Load configuration
config = read_config(config_file)

# Dynamic conversion of environment properties to ensure compatibility
environment = config["environment"]
for key, value in environment.items():
    if isinstance(value, list):  # Convert lists to dictionaries with name, value, learnable
        config["environment"][key] = {
            "name": key,
            "value": value,
            "learnable": False,
        }
    elif not isinstance(value, dict):  # Wrap scalar values in a dictionary
        config["environment"][key] = {
            "name": key,
            "value": value,
            "learnable": False,
        }

# Optional: Debugging output to verify conversion
print(":: Converted environment configuration:")
for key, value in config["environment"].items():
    print(f"{key}: {value}")

# Access simulation metadata
simulation_metadata = config["simulation_metadata"]
num_episodes = simulation_metadata["num_episodes"]
num_steps_per_episode = simulation_metadata["num_steps_per_episode"]

# Initialize registry and substeps
registry, substeps = initialize_registry()

# Optional: Debugging output
print(":: Substeps returned by initialize_registry:")
print(substeps)

# Initialize the runner
runner = Runner(config, registry)
runner.init()

print(":: preparing simulation...")

# Run the simulation
for episode in range(num_episodes):
    runner.reset()
    print(f":: starting episode {episode + 1}")

    for step in trange(num_steps_per_episode, desc=f":: running simulation {episode + 1}"):
        runner.step(1)  # Advance the simulation by 1 step

print(":: execution completed")

