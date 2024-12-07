import torch
import networkx as nx
import osmnx as ox

def map_network(params=None):
    """
    Generates a graph and its adjacency matrix based on the given parameters.

    Parameters:
    - params (dict): Dictionary with optional keys:
        - "coordinates" (tuple): Latitude and longitude of the central point.
        - "distance" (int): Radius in meters to generate the graph.

    Returns:
    - graph (networkx.Graph): The generated graph.
    - adjacency_matrix (torch.Tensor): The adjacency matrix of the graph as a PyTorch tensor.
    """
    # Default parameters
    default_params = {
        "coordinates": (40.78264403323726, -73.96559413265355),  # Central Park
        "distance": 550,  # Default radius in meters
    }

    # Use default parameters if not provided
    if params is None:
        params = default_params
    else:
        params = {**default_params, **params}  # Merge defaults with provided params

    coordinates = params["coordinates"]
    distance = params["distance"]

    # Generate the graph using osmnx
    graph = ox.graph_from_point(
        coordinates, dist=distance, simplify=True, network_type="walk"
    )

    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(graph).todense()

    return graph, torch.tensor(adjacency_matrix)