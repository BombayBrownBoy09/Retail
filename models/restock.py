import torch
import torch.nn as nn

from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)

@Registry.register_substep("Restock", "observation")
class CheckStockLevels(SubstepObservation):
    """
    Reads environment/product_stocks and environment/product_ids,
    and determines which products are below a given threshold.
    Outputs substep/stock_below_threshold (a dictionary).
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "environment/product_stocks", 
            "environment/product_ids"
        ]
        output_variables = output_variables or ["substep/stock_below_threshold"]
        arguments = arguments or {"threshold": 10}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        # Convert product_stocks to a Torch tensor
        product_stocks_list = state["environment"]["product_stocks"]  # e.g. [10.0, 5.0, 50.0]
        product_ids = state["environment"]["product_ids"]             # e.g. [1, 2, 3]

        product_stocks = torch.tensor(product_stocks_list, dtype=torch.float32)
        threshold = float(self.arguments.get("threshold", 10))

        # We'll compare stocks to threshold using Torch
        below_threshold_mask = product_stocks < threshold
        # (below_threshold_mask is a boolean tensor)

        stock_below_threshold = {}
        for idx, is_below in enumerate(below_threshold_mask):
            if is_below.item():
                pid = int(product_ids[idx])
                deficit = threshold - product_stocks[idx].item()
                stock_below_threshold[pid] = deficit

        state["substep"]["stock_below_threshold"] = stock_below_threshold
        print(f"[Restock/Observation] stock_below_threshold={stock_below_threshold}")
        return state


@Registry.register_substep("Restock", "policy")
class DetermineRestockQuantity(SubstepAction):
    """
    Reads substep/stock_below_threshold and environment/restock_quantity,
    and decides how many units to restock for each product.
    Outputs substep/restock_quantity (a dictionary).
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "substep/stock_below_threshold", 
            "environment/restock_quantity"
        ]
        output_variables = output_variables or ["substep/restock_quantity"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        stock_below_threshold = state["substep"]["stock_below_threshold"]
        # restock_quantity in environment is the max number to restock, e.g. 100
        restock_capacity = state["environment"]["restock_quantity"]
        # Convert to a tensor if needed
        restock_capacity_t = torch.tensor(restock_capacity, dtype=torch.float32)

        restock_quantity_dict = {}
        for product_id, deficit in stock_below_threshold.items():
            # You could, for instance, clamp the deficit to not exceed restock_capacity:
            needed_t = torch.tensor(deficit, dtype=torch.float32)
            restock_amount = torch.min(needed_t, restock_capacity_t)
            restock_quantity_dict[product_id] = float(restock_amount.item())

        state["substep"]["restock_quantity"] = restock_quantity_dict
        print(f"[Restock/Policy] restock_quantity={restock_quantity_dict}")
        return state


@Registry.register_substep("Restock", "transition")
class UpdateStockLevels(SubstepTransition):
    """
    Reads substep/restock_quantity and environment/product_stocks,
    updates environment/product_stocks in-place.
    """
    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None):
        config = config or {"simulation_metadata": {"calibration": False}}
        input_variables = input_variables or [
            "substep/restock_quantity", 
            "environment/product_stocks"
        ]
        output_variables = output_variables or ["environment/product_stocks"]
        arguments = arguments or {}
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        restock_quantity_dict = state["substep"]["restock_quantity"]
        product_stocks_list = state["environment"]["product_stocks"]
        product_ids = state["environment"]["product_ids"]

        # Convert to Torch for the update
        product_stocks = torch.tensor(product_stocks_list, dtype=torch.float32)

        # Update each stock by the restock amount
        for idx, pid in enumerate(product_ids):
            pid = int(pid)
            if pid in restock_quantity_dict:
                product_stocks[idx] += restock_quantity_dict[pid]

        # Store back as a Python list (or keep as a tensor if your system allows)
        state["environment"]["product_stocks"] = product_stocks.tolist()
        print(f"[Restock/Transition] Updated product_stocks={state['environment']['product_stocks']}")
        return state
