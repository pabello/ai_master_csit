import pandas as pd
from os.path import abspath


def load_data():
    path = abspath(".")
    input_data = pd.read_excel(f"{path}/GA_task.xlsx", skiprows=[0])
    cols = [input_data[x] for x in input_data]
    tasks = {}
    for i in range(0, len(cols), 2):
        resource = cols[i]
        time = cols[i+1]
        
        task = tuple((resource[j], time[j]) for j in range(len(resource)))
        tasks[f"job_{i//2 + i%2 + 1}"] = task

    df = pd.DataFrame(tasks)
    return df