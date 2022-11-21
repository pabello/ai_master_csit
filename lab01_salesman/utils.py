import numpy as np
from random import choice


def generate_graph_2d(qty: int):
    """Generates a set of interconnected cities (graph nodes) connected with roads (edges)
    of length from range [1,100]

    Args:
        qty (int): Number of cities (nodes) in the graph
    """
    graph = np.zeros((qty, qty))
    for row in range(qty):
        for col in range(row+1, qty):
            distance = np.random.randint(100) + 1
            graph[row, col] = distance
            graph[col, row] = distance
    return graph


def make_graph_asymmetric(graph: np.ndarray):
    size = graph.shape[0]
    for row in range(size):
        for col in range(row, size):
            weight = choice([.9, 1, 1.1])
            graph[row][col] *= weight
            graph[col][row] *= (2 - weight)
    return graph


if __name__ == '__main__':
    np.random.seed(6)
    graph = generate_graph_2d(5)
    print(graph)
    print(make_graph_asymmetric(graph))