import pandas as pd
import numpy as np
import utils
import gc
import os, psutil

from math import ceil
from copy import deepcopy, copy
from random import choice, choices, seed, sample


EPOCHS = 100
MIN_SURVIVORS = 3
POPULATION_COUNT = 20
SELECTION_RATIO = 0.1
MUTATION_PROBABILITY = 0.01


def generate_initial_order(dimmensions:tuple[int, int]):
    """Generates a list of jobs to complete their tasks in order.

    Args:
        dimmensions (tuple[int, int]): tuple of task matrix dimmensions (tasks per job, jobs)
    """
    task_list = [job for job in range(dimmensions[1]) for _ in range(dimmensions[0])]
    np.random.shuffle(task_list)
    return task_list


def fitness(jobs_order:list[int], dataset:pd.DataFrame):
    resources_count = max(job[1][i][0] for job in dataset.items() for i in range(len(job[1])))
    resource_timers = {name+1:0 for name in range(resources_count)}
    job_timers = {job:0 for job in dataset.columns}
    job_task_count = [0 for _ in range(dataset.shape[1])]
    
    for job in jobs_order:
        task_number = job_task_count[job]
        task = dataset.iloc[task_number, job]
        
        job_timers[job] = resource_timers[task[0]] = max(job_timers[job], resource_timers[task[0]]) + task[1]
        job_task_count[job] += 1
    
    return max(job_timers.values())


# def get_crossover_cycle(parent_1:list[int], parent_2:list[int], starting_index:int):
    


def crossover(parent_1_source:list[int], parent_2_source:list[int]) -> tuple[tuple[int], tuple[int]]:
    task_count = len(parent_1_source)
    
    parent_1 = parent_1_source[:]
    parent_2 = parent_2_source[:]
    child_1 = [None for _ in range(task_count)]
    child_2 = [None for _ in range(task_count)]
    
    # cycle crossover
    cycle_count = 1
    cycle_indexes = []
    index = np.random.randint(task_count)
    while True:
        cycle_indexes.append(index)
        
        child_1[index] = parent_1[index]
        child_2[index] = parent_2[index]
        
        index = parent_1.index(child_2[index])
        if index in cycle_indexes:
            if len(cycle_indexes) >= 0.3 * task_count:
                break
            else:
                # print(f"{len(cycle_indexes)} indexes done | taking cycle no. {cycle_count}")
                cycle_count += 1
                index = np.random.choice([id for id in range(task_count) if id not in cycle_indexes])
                for id in cycle_indexes:
                    parent_1[id] = parent_2[id] = None
    
    tasks_per_job = task_count / (max(parent_1_source) + 1)
    jobs_count_1 = {job:child_1.count(job) for job in range(50)}
    jobs_count_2 = {job:child_2.count(job) for job in range(50)}
    for index in [id for id in range(task_count) if id not in cycle_indexes]:
        if jobs_count_1[parent_2[index]] < tasks_per_job:
            child_1[index] = parent_2[index]
            jobs_count_1[parent_2[index]] += 1
        if jobs_count_2[parent_1[index]] < tasks_per_job:
            child_2[index] = parent_1[index]
            jobs_count_2[parent_1[index]] += 1
            
    # print(jobs_count_1)
    # print(jobs_count_2)
    
    child_1_empties = [index for index in range(len(child_1)) if child_1[index] == None]
    child_2_empties = [index for index in range(len(child_2)) if child_2[index] == None]
    child_1_missing = []
    child_2_missing = []
    for job in [item for item in jobs_count_1.items() if item[1]<11]:
        child_1_missing += [job[0]] * (11-job[1])
    for job in [item for item in jobs_count_2.items() if item[1]<11]:
        child_2_missing += [job[0]] * (11-job[1])
    np.random.shuffle(child_1_missing)
    np.random.shuffle(child_2_missing)
    for index, value in zip(child_1_empties, child_1_missing):
        child_1[index] = value
    for index, value in zip(child_2_empties, child_2_missing):
        child_2[index] = value
    
    return child_1, child_2


def mutate(specimen:list[int]):
    for _ in range(int(len(specimen) * MUTATION_PROBABILITY)):
        index_1, index_2 = np.random.randint(low=0, high=len(specimen), size=2)
        specimen[index_1], specimen[index_2] = specimen[index_2], specimen[index_1]
    return specimen


if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    # np.random.seed(6)
    dataset = utils.load_data()
    
    selection_count = max(MIN_SURVIVORS, ceil(POPULATION_COUNT * SELECTION_RATIO))
    
    children = [generate_initial_order(dataset.shape) for _ in range(POPULATION_COUNT)]
    for _ in range(EPOCHS):
        results = [(specimen, fitness(specimen, dataset)) for specimen in children]
        np.random.shuffle(results)
        
        group_size = ceil(POPULATION_COUNT / selection_count)
        # selected = sorted([
        #                     sorted(results[group_size*group:group_size*group+group_size], key=lambda x:x[1])[0]
        #                     for group in range(selection_count)], key=lambda x:x[1])
        selected = sorted(results, key=lambda x:x[1])[:selection_count]

        print([solution[1] for solution in selected])
        
        children = [copy(specimen[0]) for specimen in selected]
        children += [mutate(specimen[0]) for specimen in selected]
        crossover_count = (POPULATION_COUNT - len(children) + 1) // 2
        for _ in range(crossover_count):
            parent_1, parent_2 = sample(selected, k=2)
            children += [mutate(child) for child in crossover(parent_1[0], parent_2[0])]
        
        jobs_counts = [{job:genome.count(job) for job in range(50)} for genome in children]

        check = [item.items() for item in jobs_counts if item[1] != 11]
        if len(check):
            # print(check)
            raise(Exception())
        # gc.collect()
        # print(f"Memory usage: {process.memory_info().rss / (1024 ** 2):0.2f} MB")
