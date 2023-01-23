import pandas as pd
import utils

from copy import deepcopy
from random import choice, choices, seed


# def get_first_generation_task_order(job_matrix:pd.DataFrame):
#     incomplete_jobs = {job:[] for job in job_matrix.columns}
#     complete_jobs = {}
    
#     tasks_total = len(job_matrix) * len(job_matrix.columns)
#     for i in range(tasks_total):
#         job = choice(list(incomplete_jobs.keys()))
#         incomplete_jobs[job].append(i)
#         if len(incomplete_jobs[job]) == len(job_matrix):
#             complete_jobs[job] = incomplete_jobs.pop(job)
#     output = pd.DataFrame(complete_jobs)
#     print(output)
#     return output


def get_first_generation_task_order(job_matrix:pd.DataFrame) -> list:
    available_tasks = {job:job_matrix[job].tolist() for job in job_matrix}
    task_order = []
    
    tasks_total = len(job_matrix) * len(job_matrix.columns)
    for _ in range(tasks_total):
        job = choice(list(available_tasks.keys()))
        task = available_tasks[job].pop(0)
        task_order.append( (job, task) )
        if not len(available_tasks[job]):
            del(available_tasks[job])
    return task_order


def fitness(task_order:pd.DataFrame, df:pd.DataFrame):
    resources_count = max(job[1][i][0] for job in df.items() for i in range(len(job[1])))
    resource_timers = {name+1:0 for name in range(resources_count)}
    job_timers = {job:[0, []] for job in df.columns}
    
    for i, task in enumerate(task_order):
        job = task[0]
        resource = task[1][0]
        time = task[1][1]
        
        job_timers[job][0] = resource_timers[resource] = max(job_timers[job][0], resource_timers[resource])
        
        resource_timers[resource] += time
        job_timers[job][0] += time
        job_timers[job][1].append((i+1, task[1]))
    
    return max([job[0] for job in job_timers.values()]), job_timers


def task_order_dict_to_list(order_dict:pd.DataFrame) -> list:
    tasks_per_job = len(order_dict[list(order_dict.keys())[0]][1])
    jobs_count = len(order_dict.keys())
    tasks_total =  jobs_count * tasks_per_job
    task_order = [0 for _ in range(tasks_total)]
    
    # all_tasks = [order_dict[1][job][1][item] for job in order_dict[1].keys() for item in range(11)]
    # sorted_tasks = sorted(all_tasks, key=lambda x: x[0])
    
    idis = []

    for job in order_dict.keys():
        for task in order_dict[job][1]:
            idis.append(task[0]-1)
            task_order[task[0]-1] = (job, task[1])
            
    idtable = [0 for _ in range(550)]
    for id in idis:
        idtable[id] += 1
    idtable = [i for i,x in enumerate(idtable) if x > 1]
    print()
    print(idtable)
    print()
    
    return task_order


   # Problemem jest to, że taski są zamieniane na podstawie słownika, ale nie mamy pewności, że task o id, które dodajemy też będzie zamieniony i przez to są podwójne IDIKI
def crossover(parent_1:pd.DataFrame, parent_2:pd.DataFrame):  ### TODO: To nie może być po kolei, musi być losowo, bo przy tych zamych rodzicach wyjdą zawsze takie same dzieci
    child_1_timers = deepcopy(parent_1[1])
    child_2_timers = deepcopy(parent_2[1])
    
    tasks_in_total = len(child_1_timers)
    
    p1_changes = 0
    p2_changes = 0
    
    for key in child_1_timers:
        tasks_in_job = len(child_1_timers[key][1])
        for i in range(tasks_in_job):
            temp = child_1_timers[key][1][i]
            low1 = child_1_timers[key][1][i-1][0]
            new1 = child_2_timers[key][1][i][0]
            high1 = child_1_timers[key][1][i+1][0]
            low2 = child_2_timers[key][1][i-1][0]
            new2 = child_1_timers[key][1][i][0]
            high2 = child_2_timers[key][1][i+1][0]
            # Child 1
            if (i == 0 or child_1_timers[key][1][i-1][0] < child_2_timers[key][1][i][0]) and \
                (i == tasks_in_job-1 or child_1_timers[key][1][i+1][0] > child_2_timers[key][1][i][0]):
                    child_1_timers[key][1][i] = child_2_timers[key][1][i]
                    p1_changes += 1
            # Child 2
            if (i == 0 or child_2_timers[key][1][i-1][0] < child_1_timers[key][1][i][0]) and \
                (i == tasks_in_job-1 or child_2_timers[key][1][i+1][0] > child_1_timers[key][1][i][0]):
                    child_2_timers[key][1][i] = temp
                    p2_changes += 1
    
    print(child_1_timers)
    child_1 = task_order_dict_to_list(child_1_timers)
    child_2 = task_order_dict_to_list(child_2_timers)
    # print(child_1)
    # print()
    # print(child_2)
    print(f"{p1_changes} vs {p2_changes} changes")
    return child_1, child_2


if __name__ == "__main__":
    seed(6)
    
    population = 10
    epochs = 1
    selection_pc = 10
    selection_units = max(round(population * (selection_pc / 100)), 2)  # Only percentage of population is saved to next generation, rounded, at least 2
    print(selection_units)
    
    df = utils.load_data()
    
    # Initial population
    time_results = []
    for _ in range(population):
        task_order = get_first_generation_task_order(df)
        result = fitness(task_order, df)
        time_results.append(result)
        
        # order_from_result = task_order_dict_to_list(result)
        # print(task_order == order_from_result)
    
    # Evolution
    print([result[0] for result in sorted(time_results, key=lambda x: x[0])[:10]])

    best_performers = sorted(time_results, key=lambda x: x[0])[:selection_units]
    # Populate new generation
    time_results = [] + best_performers
    for _ in range((population - selection_units + 1) // 2):
        parent_1, parent_2 = choices(best_performers, k=2)
        child_1, child_2 = crossover(parent_1, parent_2)
        
        print()
        # print(parent_1)
        # print()
        # print(parent_2)
        # print()
        # print(child_1)
        # print()
        # print(child_2)
        print()
        
        time_results.append(fitness(child_1, df))
        time_results.append(fitness(child_2, df))
    
    print([result[0] for result in sorted(time_results, key=lambda x: x[0])[:10]])
    # print(parent_1['job_1'][1])
    
