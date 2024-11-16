
# Retail Agent-Based Model

This project implements an agent-based model (ABM) for retail simulations using the AgentTorch framework. The model simulates consumer purchasing behavior and retailer inventory management, incorporating promotions and restocking logic.

## Features

- **Agent Behavior**: Consumers make purchase decisions based on budget, price sensitivity, and promotional offers.
- **Environment Dynamics**: Tracks product stock levels, applies promotions, and manages restocking.
- **Simulation Steps**:
  - **Purchase**: Consumers decide on purchases.
  - **Deliver**: Updates stock levels after purchases.
  - **Restock**: Products are restocked if stock levels fall below a threshold.
- **Configuration**: Fully customizable via a `config.yaml` file.

## Directory Structure

- `data/`: Raw and processed data.
- `models/`: Model definition, including substeps and simulation logic.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `scripts/`: Scripts to run simulations.
- `tests/`: Unit tests for validating the model.
- `requirements.txt`: Python dependencies.
- `setup.py`: Package setup configuration.

## Setup Instructions

1. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate 
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The model parameters are specified in a `config.yaml` file, which includes:

- **Simulation steps**: Number of steps to simulate.
- **Environment settings**:
  - Restock threshold: Minimum stock level before restocking.
  - Restock quantity: Number of units to reorder.
- **Promotions**:
  - Product IDs, discount rates, and durations.
- **Consumers**:
  - Count, budget range, price sensitivity range, and purchase frequency range.

Example `config.yaml`:
```yaml
simulation:
  steps: 10

environment:
  restock_threshold: 20
  restock_quantity: 100

promotions:
  - id: 1
    discount: 0.2
    duration: 7
  - id: 2
    discount: 0.1
    duration: 14

consumers:
  count: 50
  budget_range:
    min: 50
    max: 200
  price_sensitivity_range:
    min: 0.5
    max: 1.5
  purchase_frequency_range:
    min: 2
    max: 5
```

## Usage

Run simulations using:
   ```bash
   python scripts/run_simulation.py
   ```

The simulation outputs step-by-step logs of purchases, stock levels, and applied promotions.

## Testing

Run unit tests to validate model functionality:
   ```bash
   pytest tests/
   ```

## Requirements

This project requires the following Python libraries:
- `agent-torch`
- `numpy`
- `pyyaml`
- `pytest`

Install them using:
   ```bash
   pip install -r requirements.txt
   ```

## Contributions

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
