import numpy as np
from copy import deepcopy


def generate_cities(qty: int, max_offset:int = 100):
    """Generates a set of interconnected cities (graph nodes) connected with roads (edges)
    of length from range [1,100]

    Args:
        qty (int): Number of cities (nodes) in the graph
    """
    cities = []
    for _ in range(qty):
        while True:
            coordinates = np.random.randint(low=-max_offset, high=max_offset+1, size=3).tolist()
            coordinates[2] = 0
            if coordinates not in cities:
                cities.append(coordinates)
                break
    for coordinates in cities:
        coordinates[2] = np.random.randint(low=0, high=51)
    return cities


def get_2d_distance(city1:list, city2:list):
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** (1/2)


def get_symmetric_graph(cities: list) -> np.ndarray:
    """Accepts a list of cities and creates a graph in a
    numpy array.

    Args:
        cities (list): List of cities, each item is another city
        with 3 variables: x_coord, y_coord, z_coord (height level)

    Returns:
        np.ndarray: graph of connections (costs of traveling from
        one city to another)
    """
    dimmension = len(cities)
    graph = np.zeros((dimmension, dimmension))

    for row_id in range(dimmension):
        for col_id in range(row_id+1, dimmension):
            graph[row_id][col_id] = get_2d_distance(cities[row_id], cities[col_id])
            graph[col_id][row_id] = graph[row_id][col_id]
    return graph


def get_asymmetric_graph(graph_2d:np.ndarray, cities:list) -> np.ndarray:
    """Takes a graph, coppies it, basing on information from cities list
    assigns weights to all connections. The connections's costs may be
    multiplied by 0.9 or 1.1 depending on whether the city is placed
    lower or higher.

    Args:
        graph (np.ndarray): template graph
        cities (list): list of cities

    Returns:
        np.ndarray: new instance with changes applied
    """
    dimmension = len(cities)
    graph_3d = deepcopy(graph_2d)
    for row_id in range(dimmension):
        for col_id in range(row_id+1, dimmension):
            if cities[row_id][2] > cities[col_id][2]:
                multiplier = 0.9
            elif cities[row_id][2] < cities[col_id][2]:
                multiplier = 1.1
            else:
                multiplier = 1
            graph_3d[row_id][col_id] *= multiplier
            graph_3d[col_id][row_id] *= (2 - multiplier)
    return graph_3d


def remove_connections(graph: np.ndarray, connections_percentage: float = 20) -> np.ndarray:
    """Accepts a graph and returns a copy of it with removed percentage of connections
    between nodes. Makes sure the graph has no dead end cities.

    Args:
        graph (np.ndarray): graph to copy
        connections_percentage (float, optional): percentage of connections
        to remove. Defaults to 10.

    Returns:
        np.ndarray: new instance of a graph, with removed connections
    """
    if connections_percentage > 1:
        connections_percentage /= 100
    remove_qty = int((graph.size - graph.shape[0]) * connections_percentage)

    all_cities_reachable = False
    while not all_cities_reachable:  # safety loop to make sure the graph has no dead end cities
        all_cities_reachable = True
        result_graph:np.ndarray = deepcopy(graph)
        for _ in range(remove_qty):
            while True:
                coords = np.random.random_integers(0, result_graph.shape[0]-1, 2)
                if coords[0] != coords[1] and result_graph[coords[0]][coords[1]] != 0:
                    result_graph[coords[0]][coords[1]] = result_graph[coords[1]][coords[0]] = 0
                    break
        for row_id in range(dim := result_graph.shape[0]):
            available_paths = 0
            for col_id in range(dim):
                if result_graph[row_id][col_id] != 0:
                    available_paths += 1
            if available_paths < 2:
                all_cities_reachable = False
                break
    return result_graph


def get_child_paths(state: list, space: np.array) -> list:
    """_summary_

    Args:
        state (list): list containing current path (node list) as the first element, and its cost as the second
        space (np.array): array of costs of all interconnected nodes

    Returns:
        list: _description_
    """
    children = []
    possible_steps = [_ for _ in range(space.shape[0]) if _ not in state[0]]
    if not possible_steps:
        if len(state[0]) == space.shape[0]:
            state[1] += space[state[0][-1]][state[0][0]]
            state[0].append(state[0][0])
            return [state]
        return None
    for x in possible_steps:
        child_state = deepcopy(state)
        child_state[0].append(x)
        child_state[1] += space[state[0][-1]][x]
        children.append(child_state)
    return children


if __name__ == '__main__':
    np.random.seed(6)
    cities = generate_cities(5)
    graph_symmetric = get_symmetric_graph(cities)
    graph_asymmetric = get_asymmetric_graph(graph_symmetric, cities)
    print(cities)
    print(graph_symmetric)
    print(graph_asymmetric)