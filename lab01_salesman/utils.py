import numpy as np
from random import choice
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
            coordinates = np.random.randint(low=-max_offset, high=max_offset+1, size=3)
            coordinates[2] = 0
            if coordinates not in cities:
                cities.append(coordinates)
                break
    for coordinates in cities:
        coordinates[2] = np.random.randint(low=-50, high=51)
    return cities


def get_2d_distance(city1:list, city2:list):
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** (1/2)


def get_symmetric_graph(cities: list):
    dimmension = len(cities)
    graph = np.zeros((dimmension, dimmension))

    for row_id in range(dimmension):
        for col_id in range(row_id+1, dimmension):
            graph[row_id][col_id] = get_2d_distance(cities[row_id], cities[col_id])
            graph[col_id][row_id] = graph[row_id][col_id]
    return graph


def get_asymmetric_graph(graph_2d:np.ndarray, cities:list):
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


def make_graph_asymmetric(graph: np.ndarray):
    size = graph.shape[0]
    for row in range(size):
        for col in range(row, size):
            weight = choice([.9, 1, 1.1])
            graph[row][col] *= weight
            graph[col][row] *= (2 - weight)
    return graph


def remove_connections(graph: np.ndarray, connections_percentage: float = 10):
    if connections_percentage > 1:
        connections_percentage /= 100
    remove_qty = graph.size * connections_percentage // 1
    for _ in range(remove_qty):
        while True:
            coords = np.random.random_integers(0, graph.shape[0], 2)
            if coords[0] != coords[1] and graph[coords[0]][coords[1]] != 0:
                graph[coords[0]][coords[1]] = graph[coords[1]][coords[0]] = 0
                break


if __name__ == '__main__':
    np.random.seed(6)
    cities = generate_cities(5)
    graph_symmetric = get_symmetric_graph(cities)
    graph_asymmetric = get_asymmetric_graph(graph_symmetric, cities)
    print(cities)
    print(graph_symmetric)
    print(graph_asymmetric)