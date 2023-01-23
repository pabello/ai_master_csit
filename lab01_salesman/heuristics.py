import utils
import numpy as np
from time import time
from enum import Enum
from copy import deepcopy


class Heuristic(Enum):
    def n_average_heuristic(state:list, total_cities_number:int):
        cost_so_far = state[1]  # this already includes the cost for a step that is being processed
        steps_done = len(state[0][1:])
        
        steps_left = total_cities_number - steps_done
        average_cost = cost_so_far / steps_done
        return steps_left * average_cost

    def n_min_cost_heuristic(state:list, total_cities_number:int):
        min_cost = min(step[1] for step in state[0][1:])  # we need to skip the first city as its cost is 0
        steps_done = len(state[0][1:])
        
        steps_left = total_cities_number - steps_done
        return steps_left * min_cost


def get_child_states(state: list, space: np.array, heuristic:Heuristic) -> list:
    children = []
    current_node = state[0][-1][0]
    visited_nodes = [node[0] for node in state[0]]
    possible_steps = [node for node in range(space.shape[0]) if node not in visited_nodes and space[current_node][node] != 0]
    
    if not possible_steps:
        if len(visited_nodes) == space.shape[0]:
            way_back_cost = space[visited_nodes[-1]][visited_nodes[0]]
            state[0].append( (visited_nodes[0], way_back_cost) )
            state[1] += way_back_cost
            state[2] = state[1]
            return [state]
        return None
    
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
        unsolvable = False
        
        while not state_queue[0][1] == state_queue[0][2]:  # checking if solution was reached for this starting point
            try:
                state_queue += get_child_states(state_queue.pop(0), space, heuristic)
            except:
                unsolvable = True
                break
            state_queue.sort(key=lambda state:state[2])
            return_count += 1
            
        if unsolvable:
            continue
        if not best_solution or state_queue[0][1] < best_solution[1]:
            best_solution = state_queue[0]
    return best_solution[1]
    
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
    graph = utils.get_symmetric_graph(cities)

    start_time = time()
    graph_search_heurictic(graph, Heuristic.n_average_heuristic)
    end_time = time()
    print(f"{(end_time - start_time):0.6f} seconds")
    print()
    start_time = time()
    graph_search_heurictic(graph, Heuristic.n_min_cost_heuristic)
    end_time = time()
    print(f"{(end_time - start_time):0.6f} seconds")