from agent_torch.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch import Registry
import numpy as np


class Purchase(SubstepAction):
    """
    Substep for consumers making purchases based on environment observations.
    """
    def execute(self, state):
        agents = state['agents']
        products = state['environment']['products']
        purchases = []

        for agent in agents:
            chosen_products = []
            for product in products:
                promotion = product.get('promotion', 1)
                adjusted_price = product['price'] * promotion
                if adjusted_price <= agent['budget']:
                    chosen_products.append(product)
                    agent['budget'] -= adjusted_price
            purchases.append(chosen_products)
        state['actions'] = purchases
        return state


class Deliver(SubstepTransition):
    """
    Substep for updating the environment after purchases.
    """
    def execute(self, state):
        products = state['environment']['products']
        actions = state.get('actions', [])

        for purchase in actions:
            for product in purchase:
                product_id = product['id']
                for prod in products:
                    if prod['id'] == product_id:
                        prod['stock'] -= 1
        return state


class Restock(SubstepObservation):
    """
    Substep for restocking products below the threshold.
    """
    def execute(self, state):
        products = state['environment']['products']
        restock_threshold = state['environment']['restock_threshold']
        restock_quantity = state['environment']['restock_quantity']

        for product in products:
            if product['stock'] < restock_threshold:
                product['stock'] += restock_quantity
        return state


def initialize_registry():
    """
    Initializes and registers the substeps for the simulation.
    """
    registry = Registry()
    registry.register_substep(Purchase())
    registry.register_substep(Deliver())
    registry.register_substep(Restock())
    return registry
