
## Retail Promotion and Consumer Behavior Simulation

## Overview
Simulates consumer behavior, pricing, and inventory management to optimize retail strategies. Built using AgentTorch, this model provides insights into consumer-product interactions for enhancing sales and stock efficiency.

## Intended Use
- **Users**: Retail analysts, strategists.
- **Applications**: Analyze promotions, pricing strategies, and inventory management.

## Inputs
- **Agent State**: Budget, preferences, price sensitivity.
- **Environment State**: Product prices, stock levels, promotions.

## Model Architecture
An agent-based model where consumers interact with a dynamic retail environment. Core components include:
- **Agents**: Represent consumers.
- **Environment**: Simulates retail space.
- **Policy**: Guides decisions based on utility functions.
- **Transition**: Updates state post-purchase.
- **Aggregator**: Collects data for analysis.

## Substeps
1. **Purchase**: Agents gather information on prices, promotions, and stock, then use utility functions to assess and make purchase decisions.
2. **Deliver**: Stock levels drop based on purchase decisions
3. **Restock**: store environment updates stock, triggers reorders as necessary

## Model Parameters

### Consumer Agents
- **Initial Budget**: $50 - $200 (randomized).
- **Price Sensitivity**: Learnable.
- **Purchase Frequency**: Learnable.

### Environment
- **Stock Levels**: Initial stock per product (customizable).
- **Restock Threshold**: Customizable.
- **Promotion Duration**: Learnable.

### Promotions
- **Discount Rate**: Learnable.
- **Type**: Configurable (e.g., BOGO, % off).
- **Consumer Reach**: Learnable.

### Stock Management
- **Initial Inventory**: Customizable.
- **Reorder Quantity**: Configurable.
- **Lead Time**: Learnable.


## Outputs
- **Sales Data**: Products sold, volume, velocity.
- **Stock Levels**: Trends over time.
- **Metrics**: Consumer behavior and promotion impact.

## Performance
- **Metrics**: Sales volume, stock depletion, promotion effectiveness.
- **Evaluation**: Benchmarked with test scenarios and historical data.

## Limitations
- Assumes static preferences and simplified behavior.
- Reordering rules may not capture real-world complexities.

## Ethical Considerations
- Ensure unbiased modeling and transparency of assumptions.
