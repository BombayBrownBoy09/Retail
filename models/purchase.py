import torch
import torch.nn as nn

from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.core.registry import Registry

@Registry.register_substep("Purchase", "observation")
class GetEnvironmentInfo(SubstepObservation):
    """
    Reads environment/product_stocks and environment/product_prices,
    and makes them available to consumers in substep/consumer_observation.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "environment/product_stocks",
            "environment/product_prices",
        ]
        output_variables = output_variables or ["substep/consumer_observation"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        product_stocks = state["environment"]["product_stocks"]  # e.g. [10.0, 5.0, 50.0]
        product_prices = state["environment"]["product_prices"]  # e.g. [20.0, 15.0, 10.0]

        # Convert to torch Tensors for potential manipulations
        product_stocks_t = torch.tensor(product_stocks, dtype=torch.float32)
        product_prices_t = torch.tensor(product_prices, dtype=torch.float32)

        # Example small operation: clamp negative stocks to 0 (just in case)
        product_stocks_t = torch.clamp(product_stocks_t, min=0)

        consumer_observation = {
            "stocks": product_stocks_t.tolist(),
            "prices": product_prices_t.tolist()
        }

        state["substep"]["consumer_observation"] = consumer_observation
        print(f"[Purchase/Observation] consumer_observation={consumer_observation}")
        return state


@Registry.register_substep("Purchase", "policy")
class Purchase(SubstepAction):
    """
    Each consumer decides which product(s) to purchase based on their budget
    and price sensitivity.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "environment",              # or specifically environment/product_stocks, etc.
            "substep/consumer_observation"
        ]
        output_variables = output_variables or ["substep/purchase_actions"]
        arguments = arguments or {
            "shape": [50, 1]  # shape for actions, as per your YAML
        }
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        consumer_observation = state["substep"]["consumer_observation"]
        product_stocks = consumer_observation["stocks"]  # list of floats
        product_prices = consumer_observation["prices"]  # list of floats

        # Convert to torch Tensors for some operations
        product_stocks_t = torch.tensor(product_stocks, dtype=torch.float32)
        product_prices_t = torch.tensor(product_prices, dtype=torch.float32)

        # Access consumer data
        # Each consumer is in state["agents"]["bap"], which is typically a list of dicts
        consumers = state["agents"]["bap"]

        purchase_actions = []

        for consumer in consumers:
            budget_t = torch.tensor(consumer["budget"], dtype=torch.float32)  # shape=[1]
            sensitivity = consumer.get("price_sensitivity", [1.0])
            sensitivity_t = torch.tensor(sensitivity, dtype=torch.float32)  # shape=[1]

            chosen_product_idx = None
            # Simple logic: pick first affordable product
            for idx in range(len(product_prices_t)):
                # Adjusted price
                adjusted_price_t = product_prices_t[idx] * sensitivity_t[0]
                # Check if there's stock and enough budget
                if product_stocks_t[idx] > 0 and budget_t[0] >= adjusted_price_t:
                    chosen_product_idx = idx
                    break

            if chosen_product_idx is not None:
                # Deduct from budget
                budget_t[0] -= product_prices_t[chosen_product_idx]
                # Update the consumer's budget in the state
                consumer["budget"][0] = float(budget_t[0].item())

                purchase_actions.append({
                    "consumer_id": consumer["id"][0],
                    "product_index": chosen_product_idx,
                    "price_paid": float(product_prices_t[chosen_product_idx].item())
                })
            else:
                purchase_actions.append({
                    "consumer_id": consumer["id"][0],
                    "product_index": None,
                    "price_paid": 0.0
                })

        state["substep"]["purchase_actions"] = purchase_actions
        print(f"[Purchase/Policy] purchase_actions={purchase_actions}")
        return state


@Registry.register_substep("Purchase", "transition")
class FinalizePurchases(SubstepTransition):
    """
    Optionally finalize or transform the purchase actions, storing them as final_purchase_actions.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or ["substep/purchase_actions"]
        output_variables = output_variables or ["substep/final_purchase_actions"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        purchase_actions = state["substep"]["purchase_actions"]

        # If needed, we can run some Torch operation here 
        # (though typically you'd just pass them through).
        # For example, let's just create a dummy tensor.
        dummy_t = torch.tensor([len(purchase_actions)], dtype=torch.float32)
        print(f"[Purchase/Transition] Number of purchase actions = {dummy_t.item()}")

        final_purchase_actions = purchase_actions
        state["substep"]["final_purchase_actions"] = final_purchase_actions

        print(f"[Purchase/Transition] final_purchase_actions={final_purchase_actions}")
        return state

