import sys
import os
import numpy as np
from models.retail_model import initialize_registry, Purchase, Deliver, Restock


def test_purchase_substep():
    """
    Test the Purchase substep to ensure agents correctly make purchases.
    """
    state = {
        "agents": [
            {"id": 1, "budget": 100, "price_sensitivity": 1.0},
            {"id": 2, "budget": 50, "price_sensitivity": 1.2},
        ],
        "environment": {
            "products": [
                {"id": 1, "name": "Product A", "price": 20, "stock": 10, "promotion": 0.8},
                {"id": 2, "name": "Product B", "price": 50, "stock": 5},
            ]
        }
    }

    purchase = Purchase()
    updated_state = purchase.execute(state)

    # Assert at least one purchase is made
    assert "actions" in updated_state
    assert len(updated_state["actions"]) > 0

    # Assert budgets are updated correctly
    for agent in updated_state["agents"]:
        assert agent["budget"] <= 100


def test_deliver_substep():
    """
    Test the Deliver substep to ensure stock levels are updated correctly.
    """
    state = {
        "agents": [],
        "environment": {
            "products": [
                {"id": 1, "name": "Product A", "price": 20, "stock": 10},
                {"id": 2, "name": "Product B", "price": 50, "stock": 5},
            ]
        },
        "actions": [[{"id": 1, "name": "Product A", "price": 20}]]
    }

    deliver = Deliver()
    updated_state = deliver.execute(state)

    # Assert stock levels are updated
    assert updated_state["environment"]["products"][0]["stock"] == 9  # Stock reduced by 1
    assert updated_state["environment"]["products"][1]["stock"] == 5  # No change


def test_restock_substep():
    """
    Test the Restock substep to ensure products are restocked correctly.
    """
    state = {
        "agents": [],
        "environment": {
            "products": [
                {"id": 1, "name": "Product A", "price": 20, "stock": 1},
                {"id": 2, "name": "Product B", "price": 50, "stock": 10},
            ],
            "restock_threshold": 5,
            "restock_quantity": 10
        }
    }

    restock = Restock()
    updated_state = restock.execute(state)

    # Assert products below the threshold are restocked
    assert updated_state["environment"]["products"][0]["stock"] == 11  # Restocked
    assert updated_state["environment"]["products"][1]["stock"] == 10  # No change


def test_registry_execution():
    """
    Test the full registry to ensure substeps execute in the correct order.
    """
    state = {
        "agents": [
            {"id": 1, "budget": 100, "price_sensitivity": 1.0},
        ],
        "environment": {
            "products": [
                {"id": 1, "name": "Product A", "price": 20, "stock": 10, "promotion": 0.8},
            ],
            "restock_threshold": 5,
            "restock_quantity": 10
        }
    }

    registry = initialize_registry()
    for substep in registry.get_substeps():
        state = substep.execute(state)

    # Assert the final state reflects all substeps
    assert state["environment"]["products"][0]["stock"] >= 10  # Restocked after purchase
    assert state["agents"][0]["budget"] < 100  # Budget reduced
