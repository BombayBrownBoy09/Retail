from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.core.registry import Registry


class Purchase(SubstepAction):
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or ["agents", "environment"]
        output_variables = output_variables or ["actions"]
        arguments = arguments or {"learnable": {}, "fixed": {}}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
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
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or ["environment", "actions"]
        output_variables = output_variables or ["environment"]
        arguments = arguments or {"learnable": {}, "fixed": {}}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
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
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or ["environment"]
        output_variables = output_variables or ["environment"]
        arguments = arguments or {"learnable": {}, "fixed": {}}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
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
    registry.register_substep(
        Purchase(config={"simulation_metadata": {"calibration": False}},
                 input_variables=["agents", "environment"],
                 output_variables=["actions"]),
        "purchase",  # Key as a positional argument
    )
    registry.register_substep(
        Deliver(config={"simulation_metadata": {"calibration": False}},
                input_variables=["environment", "actions"],
                output_variables=["environment"]),
        "deliver",  # Key as a positional argument
    )
    registry.register_substep(
        Restock(config={"simulation_metadata": {"calibration": False}},
                input_variables=["environment"],
                output_variables=["environment"]),
        "restock",  # Key as a positional argument
    )
    return registry




