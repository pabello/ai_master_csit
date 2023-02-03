import utils
import numpy as np
from time import time
from enum import Enum
from copy import deepcopy

from random import seed


class Heuristic(Enum):
    def n_average_heuristic(state:list, total_cities_number:int):
        cost_so_far = state[1]  # this already includes the cost for a step that is being processed
        steps_done = len(state[0][1:])
        
        steps_left = total_cities_number - steps_done  # this includes the way back to the first city
        average_cost = cost_so_far / steps_done
        return steps_left * average_cost

    def n_min_cost_heuristic(state:list, total_cities_number:int, global_min:float):
        min_cost = min(step[1] for step in state[0][1:])  # we need to skip the first city as its cost is 0
        steps_done = len(state[0][1:])
        
        steps_left = total_cities_number - steps_done
        # return steps_left * min_cost
        return steps_left * global_min


def get_child_states(state: list, space: np.array, heuristic:Heuristic) -> list:
    children = []
    current_node = state[0][-1][0]
    visited_nodes = [node[0] for node in state[0]]
    possible_steps = [node for node in range(space.shape[0]) if node not in visited_nodes and space[current_node][node] != 0]
    
    for x in possible_steps:
        step_cost = space[visited_nodes[-1]][x]
        child_state = deepcopy(state)
        child_state[0].append( (x, step_cost) )
        child_state[1] += step_cost
        child_state[2] = child_state[1] + heuristic(child_state, space.shape[0])
        children.append(child_state)
    return children


def graph_search_heurictic(space:np.ndarray, heuristic:Heuristic):
    """                       0               1                 2
    state_definition: [[cities_visited], cost_so_far, total_cost_estimation]
    """
    return_count = 0
    best_solution = None
    starting_states = [[[(x, 0)], 0, 0] for x in range(space.shape[0])]
    for starting_state in starting_states:
        state_queue = get_child_states(starting_state, space, heuristic)
        state_queue.sort(key=lambda state:state[2])
        
        while len(state_queue):
            best_state = state_queue.pop(0)
            if len(best_state[0]) > space.shape[0]:
                if best_state[2] <= best_state[1]:
                    print("estimation was better")
                if not best_solution or best_state[1] < best_solution[1]:
                    best_solution = deepcopy(best_state)
                break
            elif len(best_state[0]) == space.shape[0]:
                first_node = best_state[0][0][0]
                last_node = best_state[0][-1][0]
                way_back_cost = space[last_node][first_node]
                if way_back_cost != 0:
                    best_state[0].append( (first_node, way_back_cost) )
                    best_state[1] += way_back_cost
                    best_state[2] = best_state[1]
                    state_queue.append(best_state)
            else:
                child_states = get_child_states(best_state, space, heuristic)
                state_queue += child_states
                return_count += 1
            state_queue.sort(key=lambda state:state[2])
    print(return_count)
    return best_solution[1] if isinstance(best_solution, list) else "Unsolvable"
    # return best_solution[1], return_count, [city[0] for city in best_solution[0]] if isinstance(best_solution, list) else "Unsolvable"
    
def heuristic_search(space:np.ndarray):
    s_time = time()
    avg_best_cost = graph_search_heurictic(space, Heuristic.n_average_heuristic)
    e_time = time()
    n_avg_time = e_time - s_time
    
    s_time = time()
    min_best_cost = graph_search_heurictic(space, Heuristic.n_min_cost_heuristic)
    e_time = time()
    n_min_time = e_time - s_time

    return {"avg":(avg_best_cost, n_avg_time), "min":(min_best_cost, n_min_time)}


if __name__ == "__main__":
    np.random.seed(6)
    cities = utils.generate_cities(9)
    cities = [[89, -87, 18], [-26, -49, 25], [60, 1, 30], [-79, 8, 21], [-28, 69, 37], [58, -39, 7], [12, -1, 18], [-74, -43, 15], [19, -9, 7]]
    graph = utils.get_symmetric_graph(cities)
    utils.remove_connections(graph)
    graph = utils.get_asymmetric_graph(graph, cities)
    graph = utils.remove_connections(graph)

    start_time = time()
    cost = graph_search_heurictic(graph, Heuristic.n_average_heuristic)
    # cost, returns, path = graph_search_heurictic(graph, Heuristic.n_average_heuristic)
    end_time = time()
    print(f"Cost: {cost} | solved in {(end_time - start_time):0.6f} seconds")
    # print(f"Cost: {cost} | {returns:4d} steps | solved in {(end_time - start_time):0.6f} seconds")
    # print(path)
    print()
    start_time = time()
    cost = graph_search_heurictic(graph, Heuristic.n_min_cost_heuristic)
    # cost, returns, path = graph_search_heurictic(graph, Heuristic.n_min_cost_heuristic)
    end_time = time()
    print(f"Cost: {cost} | solved in {(end_time - start_time):0.6f} seconds")
    # print(f"Cost: {cost} | {returns:4d} steps | solved in {(end_time - start_time):0.6f} seconds")
    # print(path)