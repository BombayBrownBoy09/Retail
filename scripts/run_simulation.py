from models.retail_model import initialize_registry
from agent_torch import Runner
import yaml


def load_config():
    """
    Load configuration from config.yaml.
    """
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def initialize_simulation(config):
    """
    Initialize the simulation state with consumers and environment.
    """
    # Define products
    products = [
        {"id": 1, "name": "Product A", "price": 20, "stock": 100},
        {"id": 2, "name": "Product B", "price": 15, "stock": 150},
        {"id": 3, "name": "Product C", "price": 10, "stock": 200},
    ]

    # Define agents
    agents = [
        {
            "id": i,
            "budget": np.random.uniform(
                config['consumers']['budget_range']['min'],
                config['consumers']['budget_range']['max']
            ),
            "price_sensitivity": np.random.uniform(
                config['consumers']['price_sensitivity_range']['min'],
                config['consumers']['price_sensitivity_range']['max']
            ),
            "purchase_frequency": np.random.randint(
                config['consumers']['purchase_frequency_range']['min'],
                config['consumers']['purchase_frequency_range']['max']
            )
        }
        for i in range(config['consumers']['count'])
    ]

    # Define environment
    environment = {
        "products": products,
        "restock_threshold": config['environment']['restock_threshold'],
        "restock_quantity": config['environment']['restock_quantity'],
    }

    # Add promotions to products
    for promo in config['promotions']:
        for product in products:
            if product['id'] == promo['id']:
                product['promotion'] = 1 - promo['discount']

    # Return the initial state
    return {"agents": agents, "environment": environment}


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Initialize the registry and state
    registry = initialize_registry()
    state = initialize_simulation(config)

    # Initialize and run the simulation
    runner = Runner(state=state, registry=registry, steps=config['simulation']['steps'])
    runner.run()
