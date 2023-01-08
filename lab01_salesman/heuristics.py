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
    visited_nodes = [node[0] for node in state[0]]
    possible_steps = [_ for _ in range(space.shape[0]) if _ not in visited_nodes]
    
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
    """
    state_definition = [[cities_visited], cost_so_far, total_cost_estimation]
    """
    return_count = 0
    best_solution = None
    starting_states = [[[(x, 0)], 0, 0] for x in range(space.shape[0])]
    for starting_state in starting_states:
        state_queue = get_child_states(starting_state, space, heuristic)
        state_queue.sort(key=lambda state:state[2])
        
        while not state_queue[0][1] == state_queue[0][2]:  # checking if solution was reached for this starting point
            state_queue += get_child_states(state_queue.pop(0), space, heuristic)
            state_queue.sort(key=lambda state:state[2])
            return_count += 1
            
        if not best_solution or state_queue[0][1] < best_solution[1]:
            best_solution = state_queue[0]
    print(best_solution)
    print(f"needed to do {return_count} returns")
    return best_solution
    
"""_summary_

Args:
    state (list): list containing current path (node list) as the first element, its cost as the second,
        and approximation of total cost as the third element
    space (np.array): array of costs of all interconnected nodes
    heuristic (function): cost approximation function

Returns:
    list: _description_
"""

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