from models.retail_model import ConsumerAgent, RetailEnvironment, RetailSimulation
from agent_torch import Runner
import yaml
import numpy as np

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def initialize_simulation(config):
    # Initialize environment
    products = {
        1: {"id": 1, "name": "Product A", "price": 20, "stock": 100},
        2: {"id": 2, "name": "Product B", "price": 15, "stock": 150},
        3: {"id": 3, "name": "Product C", "price": 10, "stock": 200},
    }
    environment = RetailEnvironment(
        products=products,
        restock_threshold=config['environment']['restock_threshold'],
        restock_quantity=config['environment']['restock_quantity'],
    )

    # Initialize agents
    agents = [
        ConsumerAgent(
            budget=np.random.uniform(
                config['consumers']['budget_range']['min'],
                config['consumers']['budget_range']['max']
            ),
            price_sensitivity=np.random.uniform(
                config['consumers']['price_sensitivity_range']['min'],
                config['consumers']['price_sensitivity_range']['max']
            )
        )
        for _ in range(config['consumers']['count'])
    ]

    return RetailSimulation(agents, environment)

if __name__ == "__main__":
    config = load_config()
    simulation = initialize_simulation(config)
    runner = Runner(simulation, config['simulation']['steps'])
    runner.run()
