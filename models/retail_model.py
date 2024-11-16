from agent_torch import Agent, Environment, Simulation
import numpy as np

class ConsumerAgent(Agent):
    def __init__(self, budget, price_sensitivity):
        super().__init__()
        self.budget = budget
        self.price_sensitivity = price_sensitivity

    def observe(self, env_state):
        # Observe product prices, promotions, and stock levels
        return env_state['products']

    def act(self, observations):
        # Decide purchases based on utility functions
        purchases = []
        for product in observations:
            adjusted_price = product['price'] * product.get('promotion', 1)
            if adjusted_price <= self.budget:
                purchases.append(product)
                self.budget -= adjusted_price
        return purchases


class RetailEnvironment(Environment):
    def __init__(self, products, restock_threshold, restock_quantity):
        super().__init__()
        self.products = products
        self.restock_threshold = restock_threshold
        self.restock_quantity = restock_quantity

    def update(self, actions):
        # Update stock levels based on purchases
        for action in actions:
            for product in action:
                self.products[product['id']]['stock'] -= 1
                if self.products[product['id']]['stock'] < self.restock_threshold:
                    self.products[product['id']]['stock'] += self.restock_quantity

    def state(self):
        return {'products': self.products}

class RetailSimulation(Simulation):
    def __init__(self, agents, environment):
        super().__init__(agents, environment)

    def substep_purchase(self):
        # Each agent decides their purchases
        for agent in self.agents:
            observations = agent.observe(self.environment.state())
            actions = agent.act(observations)
            self.environment.update(actions)

    def substep_restock(self):
        # Restock the environment if needed
        self.environment.update([])  # No actions needed for restocking
