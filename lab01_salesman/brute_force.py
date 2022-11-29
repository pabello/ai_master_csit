import utils
import numpy as np
from copy import deepcopy
from time import time
from enum import Enum

np.random.seed(6)


class OrderType(Enum):
    BFS = 1
    DFS = 2


def count_calculations(dimmension: int):
    sum = 0
    factorial = 1
    for number in range(dimmension, 0, -1):
        factorial *= number
        sum += factorial
    return sum


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


def find_solutions(space: np.array, order: OrderType):
    if order == OrderType.BFS:
        take_index = 0
    if order == OrderType.DFS:
        take_index = -1
    
    solution_space = [[[x], 0] for x in range(space.shape[0])]
    solutions = []
    
    total_calculations = count_calculations(space.shape[0])
    print(f'A total of {total_calculations} calculations will be made.')
    percent = total_calculations // 100
    bar_resolution = 2
    calculations_count = 0
    
    while solution_space:
        calculations_count += 1
        child_paths = get_child_paths(solution_space.pop(take_index), space)
        if child_paths:
            if len(child_paths) == 1 and len(child_paths[0][0]) > space.shape[0]:
                solutions.append(child_paths[0])
            else:
                solution_space += child_paths
        pc_complete = calculations_count // percent
        hashes = '#' * (pc_complete // bar_resolution)
        dashes = '-' * ((100 // bar_resolution) - (pc_complete // bar_resolution))
        
        print(f'[{hashes}{dashes}] {pc_complete}%', end='\r')
    print()
    return solutions


def get_best_solutions_list(solutions: list):
    sorted_solutions = sorted(solutions, key=lambda x: x[1])
    lowest_cost = sorted_solutions[0][1]
    best_solutions = []
    for solution in sorted_solutions:
        if solution[1] == lowest_cost:
            best_solutions.append(solution)
        else:
            break
    return best_solutions


if __name__ == "__main__":
    dimmension = 9
    max_distance = 100
    
    # try:
    #     dimmension = int(input("How many cities are there?\n"))
    # except:
    #     print('This is not a number... Bye.')
    #     exit()
    print(f'Searching for a solution in {dimmension}-cities graph.')
    start_time = time()
    cities = utils.generate_cities(dimmension, max_distance)
    space = utils.get_symmetric_graph(cities)
    print(space)
    # exit()
    # space = utils.make_graph_asymmetric(space)
    time_generated = time()
    
    solutions = find_solutions(space, OrderType.DFS)
    time_solved = time()
    
    best_solutions = get_best_solutions_list(solutions)
    time_find_best = time()
    
    for solution in best_solutions:
        print(solution)
    print(f'{len(solutions)} possible paths in total')
    print(f'{len(best_solutions)} of them give equal, best result\r\n')
    print(f'Generated the space in {time_generated - start_time:.8f} seconds')
    print(f'Found the solutions in {time_solved - time_generated:.8f} seconds')
    print(f'Chose the best ones in {time_find_best - time_solved:.8f} seconds')
    
    # TODO: get rid of repeated solutions in case of symmetric
    
    # print(space)
    # lvl_1 = get_child_paths([[0], 0], space)
    # print()
    # for path in lvl_1:
    #     get_child_paths(path, space)
    #     print()
    
    exit()
    #####=====----- TEST CASES -----=====#####
    dimmension = 9
    cities = utils.generate_cities(dimmension)
    
    # Generate a symmetric graph of the cities
    space = utils.get_symmetric_graph(cities)
    
    # Symmetric, fully connected, brute force bfs, x cities
    start_time = time()
    solutions = find_solutions(space, OrderType.BFS)
    finish_time = time()
    with open('results.txt', 'a') as file:
        file.write(f'{finish_time - start_time:.4f} seconds - Symmetric, fully connected, brute force bfs, {dimmension} cities.\r\n')

    # Symmetric, fully connected, brute force bfs, x cities
    start_time = time()
    solutions = find_solutions(space, OrderType.DFS)
    finish_time = time()
    with open('results.txt', 'a') as file:
        file.write(f'{finish_time - start_time:.4f} seconds - Symmetric, fully connected, brute force dfs, {dimmension} cities.\r\n')
    
    # Generate an asymmetric graph basing on the symetric one and the cities
    space = utils.get_asymmetric_graph(space)
    
    # Asymmetric, fully connected, brute force bfs, x cities
    start_time = time()
    solutions = find_solutions(space, OrderType.BFS)
    finish_time = time()
    with open('results.txt', 'a') as file:
        file.write(f'{finish_time - start_time:.4f} seconds - Asymmetric, fully connected, brute force bfs, {dimmension} cities.\r\n')

    # Asymmetric, fully connected, brute force bfs, x cities
    start_time = time()
    solutions = find_solutions(space, OrderType.DFS)
    finish_time = time()
    with open('results.txt', 'a') as file:
        file.write(f'{finish_time - start_time:.4f} seconds - Asymmetric, fully connected, brute force dfs, {dimmension} cities.\r\n')