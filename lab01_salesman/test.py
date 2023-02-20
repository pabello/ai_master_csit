from time import time


record = [[(1, 0),
          (3, 29.274562336608895),
          (2, 53.600373133029585),
          (6, 68.26419266350405),
          (10, 31.064449134018133),
          (0, 44.40720662234904),
          (7, 19.6468827043885),
          (9, 81.41252974819048),
          (5, 109.11003620199197),
          (4, 42.0),
          (8, 75.8023746329889),
          (1, 67.05221845696084)], 621.6348256340304, 621.6348256340304]

def n_average_heuristic(state:list, total_cities_number:int):
    cost_so_far = state[1]  # this already includes the cost for a step that is being processed
    steps_done = len(state[0][1:])
    
    steps_left = total_cities_number - steps_done
    average_cost = cost_so_far / steps_done
    # print(f"average cost: {average_cost}")
    return steps_left * average_cost

def n_min_cost_heuristic(state:list, total_cities_number:int):
    min_cost = min(step[1] for step in state[0][1:])  # we need to skip the first city as its cost is 0
    steps_done = len(state[0][1:])
    
    steps_left = total_cities_number - steps_done
    # print(f"min cost: {min_cost}")
    return steps_left * min_cost

dim = 11

start_time = time()
for _ in range(10000000):
    # n_average_heuristic(record, dim)
    # n_min_cost_heuristic(record, dim)
    _ = 4352.3455234635 / 15
    # _ = [x[0] for x in record[0][1:]]
end_time = time()

print(f"{(end_time - start_time):0.6f} seconds")

print(_)