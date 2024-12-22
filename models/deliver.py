import torch
import torch.nn as nn

from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.core.registry import Registry

@Registry.register_substep("Deliver", "observation")
class GatherDeliveryInfo(SubstepObservation):
    """
    Reads environment + substep/final_purchase_actions and
    stores them in substep/delivery_info.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "environment",
            "substep/final_purchase_actions"
        ]
        output_variables = output_variables or ["substep/delivery_info"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        environment = state["environment"]
        final_purchases = state["substep"].get("final_purchase_actions", [])

        # Convert final_purchases to a torch Tensor for demonstration
        # We'll keep it simple, just get the length:
        final_purchases_t = torch.tensor([len(final_purchases)], dtype=torch.float32)
        print(f"[Deliver/Observation] final_purchase_actions count = {final_purchases_t.item()}")

        delivery_info = {
            "final_purchases": final_purchases,
        }

        state["substep"]["delivery_info"] = delivery_info
        print(f"[Deliver/Observation] delivery_info={delivery_info}")
        return state


@Registry.register_substep("Deliver", "policy")
class DecideDeliveryPlan(SubstepAction):
    """
    Takes delivery_info, decides how to deliver each product,
    outputs substep/delivery_plan.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or ["substep/delivery_info"]
        output_variables = output_variables or ["substep/delivery_plan"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        delivery_info = state["substep"]["delivery_info"]
        final_purchases = delivery_info.get("final_purchases", [])

        # Example: build a delivery_plan as is
        delivery_plan = []
        for purchase_action in final_purchases:
            # purchase_action might look like:
            #   {"consumer_id": X, "product_index": Y, "price_paid": Z}
            if purchase_action["product_index"] is not None:
                delivery_plan.append({
                    "consumer_id": purchase_action["consumer_id"],
                    "product_index": purchase_action["product_index"],
                    "deliver_now": True
                })
            else:
                # consumer didn't buy anything
                delivery_plan.append({
                    "consumer_id": purchase_action["consumer_id"],
                    "product_index": None,
                    "deliver_now": False
                })

        state["substep"]["delivery_plan"] = delivery_plan
        print(f"[Deliver/Policy] delivery_plan={delivery_plan}")
        return state


@Registry.register_substep("Deliver", "transition")
class DeliverProducts(SubstepTransition):
    """
    Updates the environment (e.g. product stocks) based on the delivery plan.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "environment",
            "substep/delivery_plan"
        ]
        output_variables = output_variables or ["environment"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        environment = state["environment"]
        delivery_plan = state["substep"]["delivery_plan"]

        product_stocks_list = environment["product_stocks"]  # e.g. [10., 5., 50.]
        product_ids = environment["product_ids"]             # e.g. [1,    2,   3 ]

        # Convert to Torch
        product_stocks_t = torch.tensor(product_stocks_list, dtype=torch.float32)

        for delivery in delivery_plan:
            p_idx = delivery.get("product_index")
            deliver_now = delivery.get("deliver_now", False)
            if deliver_now and p_idx is not None and 0 <= p_idx < len(product_stocks_t):
                # Decrement the stock by 1
                print(f"[Deliver/Transition] Delivering product_index {p_idx} stock before={product_stocks_t[p_idx].item()}")
                product_stocks_t[p_idx] = product_stocks_t[p_idx] - 1.0
                print(f"[Deliver/Transition] Delivering product_index {p_idx} stock after={product_stocks_t[p_idx].item()}")

        # Convert back to list
        environment["product_stocks"] = product_stocks_t.tolist()
        state["environment"] = environment

        print(f"[Deliver/Transition] Updated environment.product_stocks={environment['product_stocks']}")
        return state
