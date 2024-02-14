import json
import random
from utils.base import list_subdirectories

def split_data():

    data_dir = "./data/list_functions"

    tasks = list_subdirectories(data_dir)

    for task in tasks:
        
        if task == "results":
            continue
        
        task_dir = f"{data_dir}/{task}"
        task_datafile = f"{task_dir}/task.json"
        
        task_data = json.load(open(task_datafile, 'r'))
        
        data = task_data["examples"]
        random.shuffle(data)
        
        train_data = [data[:5], data[5:10], data[10:15]]
        test_data = data[15:]
        
        json.dump(train_data, open(f"{task_dir}/train.json", "w"))
        json.dump(test_data, open(f"{task_dir}/test.json", "w"))