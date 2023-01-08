import pandas as pd
import utils

from random import choice, choices, seed


# def get_task_order(job_matrix:pd.DataFrame):
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


def get_task_order(job_matrix:pd.DataFrame) -> list:
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
        job_timers[job][1].append(i+1)
    
    return max([job[0] for job in job_timers.values()]), job_timers


def task_order_dict_to_list(order_dict:pd.DataFrame) -> list:
    tasks_per_job = len(order_dict[list(order_dict.keys())[0]][1])
    jobs_count = len(order_dict)
    tasks_total =  jobs_count * tasks_per_job
    task_order = [0 for _ in range(tasks_total)]

    for job in order_dict.keys():
        for task in order_dict[job][1]:
            # task_order[]
            pass


def crossover(task_order_1:pd.DataFrame, task_order_2:pd.DataFrame):
    pass


if __name__ == "__main__":
    seed(6)
    
    population = 100
    epochs = 1
    selection_pc = 10
    
    df = utils.load_data()
    
    time_results = []
    for _ in range(population):
        task_order = get_task_order(df)
        result = fitness(task_order, df)
        time_results.append(result)
    
    print([result[0] for result in sorted(time_results, key=lambda x: x[0])[:10]])
    
    best_performers = [result[1] for result in sorted(time_results, key=lambda x: x[0])[:10]]
    parent_1, parent_2 = choices(best_performers, k=2)
    # child = crossover(parent_1, parent_2)
    print(parent_1['job_1'][1])
    
