import utils
import numpy as np
from time import time
from enum import Enum

from concurrent.futures import ThreadPoolExecutor


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


def find_solutions(space: np.array, order: OrderType):
    if order == OrderType.BFS:
        take_index = 0
    if order == OrderType.DFS:
        take_index = -1
    
    solution_space = [[[x], 0] for x in range(space.shape[0])]
    solutions = []
    
    total_calculations = count_calculations(space.shape[0])
    print(f'A total of {total_calculations} calculations will be made.')
    percent = total_calculations / 100
    bar_resolution = 2
    calculations_count = 0
    
    previous_percentage = 0
    while solution_space:
        calculations_count += 1
        child_paths = utils.get_child_paths(solution_space.pop(take_index), space)
        if child_paths:
            if len(child_paths) == 1 and len(child_paths[0][0]) > space.shape[0]:
                solutions.append(child_paths[0])
            else:
                solution_space += child_paths
        pc_complete = int(calculations_count // percent)
        hashes = '#' * (pc_complete // bar_resolution)
        dashes = '-' * ((100 // bar_resolution) - (pc_complete // bar_resolution))
        
        if pc_complete != previous_percentage:
            previous_percentage = pc_complete
            print(f'[{hashes}{dashes}] {pc_complete}%', end='\r')
    print()
    return solutions


def quiet_find_solutions(space: np.array, order: OrderType):
    if order == OrderType.BFS:
        take_index = 0
    if order == OrderType.DFS:
        take_index = -1
    
    solution_space = [[[x], 0] for x in range(space.shape[0])]
    solutions = []
    
    while solution_space:
        current_state = solution_space.pop(take_index)
        # print(current_state)
        child_paths = utils.get_child_paths(current_state, space)
        if child_paths:  # unvisited cities
            # if len(child_paths) == 1 and len(child_paths[0][0]) > space.shape[0]:
            #     solutions.append(child_paths[0])
            # else:
            solution_space += child_paths
        else:  # no available cities (possibly all visited already)
            # print(current_state)
            if len(current_state[0]) == space.shape[0]:
                back_cost = space[current_state[0][-1]][current_state[0][0]]
                if back_cost > 0:
                    current_state[1] += back_cost
                    current_state[0].append(current_state[0][0])
                    # print("-----------")
                    # print(current_state)
                    # print("-----------")
                    # exit()
                    solutions.append(current_state)
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


def usecase_description(dimmension:int, fully_connected:bool, symmetric:bool, order_type:OrderType):
    message = ""
    message += f"{dimmension:2d} {order_type.name} "
    if fully_connected:
        message += "FC "
    else:
        message += "NFC "
    if symmetric:
        message += "SYMMETRIC"
    else:
        message += "ASYMMETRIC"
    return message


def run_benchmark(solution_space:np.ndarray, order_type:OrderType, fully_connected:bool, symmetric:bool):
    start_time = time()
    solutions = find_solutions(solution_space, order_type)
    finish_time = time()

    message = usecase_description(solution_space.shape[0], fully_connected, symmetric, order_type)
    message += f" {finish_time - start_time:.4f} seconds."
    print(message)

    with open('results.txt', 'a') as file:
        file.write(message + "\r\n")


def get_results_full_range(min_cities:int=5, max_cities:int = 9):
    nfc_percent = 20
    for i in range(min_cities, max_cities+1):
        cities = utils.generate_cities(i)
        fc_sym_space = utils.get_symmetric_graph(cities)
        fc_asym_space = utils.get_asymmetric_graph(fc_sym_space, cities)
        nfc_sym_space = utils.remove_connections(fc_sym_space, nfc_percent)
        nfc_asym_space = utils.remove_connections(fc_asym_space, nfc_percent)
        run_benchmark(fc_sym_space, OrderType.DFS, True, True)
        run_benchmark(fc_asym_space, OrderType.DFS, True, False)
        run_benchmark(nfc_sym_space, OrderType.DFS, False, True)
        run_benchmark(nfc_asym_space, OrderType.DFS, False, False)
        run_benchmark(fc_sym_space, OrderType.BFS, True, True)
        run_benchmark(fc_asym_space, OrderType.BFS, True, False)
        run_benchmark(nfc_sym_space, OrderType.BFS, False, True)
        run_benchmark(nfc_asym_space, OrderType.BFS, False, False)


def get_results_full_range_async(min_cities:int=5, max_cities:int = 9):
    nfc_percent = 20
    pool = ThreadPoolExecutor(6)
    for i in range(min_cities, max_cities+1):
        cities = utils.generate_cities(i)
        fc_sym_space = utils.get_symmetric_graph(cities)
        fc_asym_space = utils.get_asymmetric_graph(fc_sym_space, cities)
        nfc_sym_space = utils.remove_connections(fc_sym_space, nfc_percent)
        nfc_asym_space = utils.remove_connections(fc_asym_space, nfc_percent)
        
        pool.submit(run_benchmark, fc_sym_space, OrderType.DFS, True, True)
        pool.submit(run_benchmark, fc_asym_space, OrderType.DFS, True, False)
        pool.submit(run_benchmark, nfc_sym_space, OrderType.DFS, False, True)
        pool.submit(run_benchmark, nfc_asym_space, OrderType.DFS, False, False)
        pool.submit(run_benchmark, fc_sym_space, OrderType.BFS, True, True)
        pool.submit(run_benchmark, fc_asym_space, OrderType.BFS, True, False)
        pool.submit(run_benchmark, nfc_sym_space, OrderType.BFS, False, True)
        pool.submit(run_benchmark, nfc_asym_space, OrderType.BFS, False, False)
    pool.shutdown()


def brute_force(space:np.ndarray):
    s_time = time()
    solutions = quiet_find_solutions(space, OrderType.DFS)
    e_time = time()
    dfs_best_solution = sorted(solutions, key=lambda x: x[1])[0]
    dfs_best_cost = dfs_best_solution[1]
    dfs_time = e_time - s_time

    s_time = time()
    solutions = quiet_find_solutions(space, OrderType.BFS)
    e_time = time()
    bfs_best_solution = sorted(solutions, key=lambda x: x[1])[0]
    bfs_best_cost = bfs_best_solution[1]
    bfs_time = e_time - s_time
    
    # dcost = [space[dfs_best_solution[0][i-1]][dfs_best_solution[0][i]] for i in range(1, len(dfs_best_solution[0]))]
    # bcost = [space[bfs_best_solution[0][i-1]][bfs_best_solution[0][i]] for i in range(1, len(bfs_best_solution[0]))]
    
    # print(f"{dfs_best_solution} | real_cost: {dcost}")
    # print(f"{bfs_best_solution} | real_cost: {bcost}")
    # print()

    return {"dfs":(dfs_best_cost, dfs_time), "bfs":(bfs_best_cost, bfs_time)}


if __name__ == "__main__":
    np.random.seed(6)
    dimmension = 9
    max_distance = 100
    get_results_full_range(min_cities=11, max_cities=11)