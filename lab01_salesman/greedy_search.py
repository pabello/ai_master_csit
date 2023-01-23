import numpy as np
import utils
from time import time


def search_with_nn(space: np.array):
    solutions = []
    
    for starting_point in range(len(space)):
        current_path = [starting_point]
        current_cost = 0.0
        unsolvable = False
        
        for _ in range(len(space) - 1):
            possible_steps = [(node, space[current_path[-1]][node]) 
                            for node in range(len(space))
                            if node not in current_path
                               and space[current_path[-1]][node] != 0]
            try:
                next_step = sorted(possible_steps, key=lambda x: x[1])[0]
            except:
                unsolvable = True
                break
            current_path.append(next_step[0])
            current_cost += next_step[1]

        if unsolvable:
            continue
        current_cost += space[current_path[-1]][starting_point]
        current_path.append(starting_point)
        
        solutions.append((current_path, current_cost))
    
    return solutions


def greedy_search(space: np.array):
    s_time = time()
    solutions = search_with_nn(space)
    e_time = time()
    solutions_sorted = sorted(solutions, key=lambda x: x[1])
    try:
        best_cost = solutions_sorted[0][1]
    except:
        best_cost = "Unsolvable"
    elapsed_time = e_time - s_time
    return best_cost, elapsed_time


if __name__ == "__main__":
    cities = utils.generate_cities(5)
    graph = utils.get_symmetric_graph(cities)
    solutions = search_with_nn(graph)
    for solution in solutions:
        print(solution)


