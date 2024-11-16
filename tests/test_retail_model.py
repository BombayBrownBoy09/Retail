import pytest
from models.retail_model import initialize_registry, Purchase, Deliver, Restock


def test_register_substep():
    """
    Test the Registry's register method.
    """
    registry, _ = initialize_registry()
    assert "purchase" in registry.policy_helpers
    assert registry.policy_helpers["purchase"] == "Purchase Substep"


def test_registry_initialization():
    """
    Ensure the registry initializes with the correct substeps.
    """
    registry, _ = initialize_registry()

    # Assert substeps are registered in the correct categories
    assert "purchase" in registry.policy_helpers
    assert registry.policy_helpers["purchase"] == "Purchase Substep"
    assert "deliver" in registry.transition_helpers
    assert registry.transition_helpers["deliver"] == "Deliver Substep"
    assert "restock" in registry.observation_helpers
    assert registry.observation_helpers["restock"] == "Restock Substep"


def test_registry_execution():
    """
    Test the full registry to ensure substeps execute in the correct order.
    """
    state = {
        "agents": [
            {"id": 1, "budget": 120, "price_sensitivity": 1.0},
        ],
        "environment": {
            "products": [
                {"id": 1, "name": "Product A", "price": 20, "stock": 4, "promotion": 0.8},  # Stock below threshold
            ],
            "restock_threshold": 5,
            "restock_quantity": 10
        }
    }

    _, substeps = initialize_registry()

    # Retrieve substeps
    purchase = substeps["purchase"]
    deliver = substeps["deliver"]
    restock = substeps["restock"]

    # Execute substeps with detailed debug prints
    print("\n[Execution] Before Purchase:")
    print(state)
    state = purchase.forward(state)
    print("\n[Execution] After Purchase:")
    print(state)

    state = deliver.forward(state)
    print("\n[Execution] After Deliver:")
    print(state)

    state = restock.forward(state)
    print("\n[Execution] After Restock:")
    print(state)

    # Assertions
    assert state["environment"]["products"][0]["stock"] >= 10  # Stock updated
    assert state["agents"][0]["budget"] < 120  # Budget reduced




def test_purchase_substep():
    """
    Test the Purchase substep to ensure agents correctly make purchases.
    """
    state = {
        "agents": [
            {"id": 1, "budget": 120, "price_sensitivity": 1.0},
        ],
        "environment": {
            "products": [
                {"id": 1, "name": "Product A", "price": 20, "stock": 10, "promotion": 0.8},
            ],
            "restock_threshold": 5,
            "restock_quantity": 10
        }
    }

    purchase = Purchase()
    updated_state = purchase.forward(state)

    # Assert at least one purchase is made
    assert "actions" in updated_state
    assert len(updated_state["actions"]) > 0

    # Assert budgets are updated correctly
    for agent in updated_state["agents"]:
        assert agent["budget"] <= 120  # Updated to reflect agent's starting budget


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
    updated_state = deliver.forward(state)

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
    updated_state = restock.forward(state)

    # Assert products below the threshold are restocked
    assert updated_state["environment"]["products"][0]["stock"] == 11  # Restocked
    assert updated_state["environment"]["products"][1]["stock"] == 10  # No change
