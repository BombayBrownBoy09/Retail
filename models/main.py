import argparse
from tqdm import trange

# Agent Torch imports
from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_config, read_from_file

# Import your substeps so that their @Registry.register_substep decorators execute
# (assuming they're all in a 'substeps' folder or package).
# If your substeps folder is named differently, adjust accordingly.
import restock, purchase, deliver  

# If you have any extra helpers, import them as well:
# from helpers import *

print(":: execution started")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="/Users/bhargav/random_projects/Retail/config.yaml")
args = parser.parse_args()

# Load the config from YAML
config_file = args.config
config = read_config(config_file)

# Extract simulation metadata
metadata = config.get("simulation_metadata", {})
num_episodes = metadata.get("num_episodes", 1)
num_steps_per_episode = metadata.get("num_steps_per_episode", 10)
visualize = metadata.get("visualize", False)

# Create the Registry instance
registry = Registry()

# Example of manually registering any initialization functions or generic functions
registry.register(read_from_file, "read_from_file", "initialization")

# Create the Runner using your config and registry
runner = Runner(config, registry)
runner.init()

print(":: preparing simulation...")

# Outer loop for episodes
for episode in trange(num_episodes, desc=":: running episode", ncols=108):
    # Reset the runner for each new episode
    runner.reset()

    # Inner loop for steps in each episode
    for step in trange(
        num_steps_per_episode,
        desc=":: executing substeps",
        leave=False,
        ncols=108,
        ascii=True,
    ):
        # Execute one step, which in turn executes the substeps:
        runner.step(1)

print(":: execution completed")
