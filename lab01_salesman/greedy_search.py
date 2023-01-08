import numpy as np
import utils


def search_with_nn(space: np.array):
    solutions = []
    
    for starting_point in range(len(space)):
        current_path = [starting_point]
        current_cost = 0.0
        
        for _ in range(len(space) - 1):
            possible_steps = [(node, space[current_path[-1]][node]) for node
                            in range(len(space)) if node not in current_path
                            and space[current_path[-1]][node] != 0]
            
            next_step = sorted(possible_steps, key=lambda x: x[1])[0]
            current_path.append(next_step[0])
            current_cost += next_step[1]
            
        solutions.append((current_path, current_cost))
    
    return solutions


if __name__ == "__main__":
    cities = utils.generate_cities(5)
    graph = utils.get_symmetric_graph(cities)
    solutions = search_with_nn(graph)
    for solution in solutions:
        print(solution)


