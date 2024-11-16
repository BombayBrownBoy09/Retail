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
                promotion = product.get('promotion', 1.0)  # Default to no promotion
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
                        print(f"Delivering product {product_id}: stock before={prod['stock']}")
                        prod['stock'] -= 1
                        print(f"Delivering product {product_id}: stock after={prod['stock']}")
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

        for i, product in enumerate(products):
            print(f"[Restock] Before: Product {product['id']} - Stock: {product['stock']}")
            if product['stock'] < restock_threshold:
                state['environment']['products'][i]['stock'] += restock_quantity
                print(f"[Restock] After: Product {product['id']} - Restocked to: {state['environment']['products'][i]['stock']}")

        # Explicitly update the environment to ensure persistence
        state['environment']['products'] = products
        print(f"[Restock] Final State: {state['environment']['products']}")
        return state






def initialize_registry():
    """
    Initializes and registers the substeps for the simulation.
    """
    registry = Registry()

    purchase = Purchase()
    deliver = Deliver()
    restock = Restock()

    # Register metadata instead of objects
    registry.register("Purchase Substep", "purchase", "policy")
    registry.register("Deliver Substep", "deliver", "transition")
    registry.register("Restock Substep", "restock", "observation")

    print("Registry contents after initialization:")
    print(registry.view())  # Prints JSON-serializable output

    # Return both the registry and the actual substep objects for test execution
    return registry, {"purchase": purchase, "deliver": deliver, "restock": restock}
