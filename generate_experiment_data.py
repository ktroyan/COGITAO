from arcworld.general_utils import generate_key
import os
import numpy as np
from tqdm import tqdm
import copy
import json
import pandas
import time
from utils.db_utils import access_db, close_db, store_task_in_db, hash_task
from generator import generator
from experiment_configs.c0 import compositionality_configs as c0_configs
from experiment_configs.compositionality import compositionality_configs 
from experiment_configs.generalization import generalization_configs
from experiment_configs.sample_efficiency import sample_efficiency_configs
from experiment_configs.compositionality_gridsize import compositionality_gridsize_config
from experiment_configs.c4 import c4_configs

def adapt_task_format(task, task_key):
    task_dict = {}
    task_dict["input"] = np.int_(task["pairs"][-1]["input"]).tolist()
    task_dict["output"] = np.int_(task["pairs"][-1]["output"]).tolist()
    
    if len(task["pairs"]) > 1:
        task_dict["demo_input"] = np.int_(task["pairs"][0]["input"]).tolist()
        task_dict["demo_output"] = np.int_(task["pairs"][0]["output"]).tolist()
    
    task_dict["transformation_suite"] = task["transformation_suite"]
    task_dict["task_key"] = task_key
    return task_dict

def handle_paths(config): 
    path = config["saving_path"].split('/') # define the saving path
    folder_path = os.path.normpath('/'.join(path[:-1]))
    db_name = folder_path.strip('/').replace("/", "_") # Same database for train, val and test to avoid duplicates
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.normpath(config["saving_path"])
    return db_name, folder_path, file_path

def generate_equal_balance_from_transforms(config, n_tasks_to_generate):
    
    db_name, folder_path, file_path = handle_paths(config)
    cursor, conn = access_db(db_name, folder_path) 
    
    list_of_transforms = config["allowed_combinations"]
    n_per_transform = int(n_tasks_to_generate / len(list_of_transforms))
    editable_config = copy.deepcopy(config)
    
    task_list = []

    while len(task_list) < n_tasks_to_generate:
        for transform in list_of_transforms:
            count_per_transform = 0
            editable_config["allowed_combinations"] = [transform]
            gen = generator(editable_config)
            
            while count_per_transform < n_per_transform and len(task_list) < n_tasks_to_generate:
                try: 
                    task = gen.generate_single_task()
                    task_key = generate_key()
                    ready_to_export_task = adapt_task_format(task, task_key)
                    task_hash = hash_task(ready_to_export_task["input"], 
                                          ready_to_export_task["transformation_suite"])
                    if store_task_in_db(cursor, conn, task_key, task_hash, \
                                        str(ready_to_export_task['transformation_suite']), 
                                        debug=False): # If the task is not a duplicate
                        task_list.append(ready_to_export_task)
                        count_per_transform += 1
                        print(f"Generated {len(task_list)} / {n_tasks_to_generate} tasks", end='\r')
                except Exception as e:
                    print(e)
                    continue
    close_db(conn)
    # Save the tasks in a json file (not using the function)
    with open(f"{file_path}", "w") as f:
        json.dump(task_list, f)


if __name__ == "__main__":
    
    
    configs_to_loop = [compositionality_configs]
    # configs_to_loop = [compositionality_gridsize_config]

    n_train = 1000000
    n_test = 1000
    n_val = 1000
    
    # df will be used to store how long each config took
    time_dict = {}

    for study in configs_to_loop:
        for config in tqdm(study):
            start_time = time.time()
            print("working on config with path:",config["saving_path"])
            if "experiment_1" in config["saving_path"] and "exp_setting_1" in config["saving_path"]:
                if "train" in config["saving_path"]:
                    generate_equal_balance_from_transforms(config, n_train)
                    
                    # Add train_val split
                    train_val_config = copy.deepcopy(config)
                    train_val_config["saving_path"] = train_val_config["saving_path"].replace("train", "val")
                    generate_equal_balance_from_transforms(train_val_config, n_val)
                    
                    # # Add test split (in distribution)
                    train_test_config = copy.deepcopy(config)
                    train_test_config["saving_path"] = train_test_config["saving_path"].replace("train", "test")
                    generate_equal_balance_from_transforms(train_test_config, n_test)

                elif "test" in config["saving_path"]:
                    
                    # Val OOD split
                    val_ood_config = copy.deepcopy(config)
                    val_ood_config["saving_path"] = val_ood_config["saving_path"].replace("test", "val_ood")
                    generate_equal_balance_from_transforms(val_ood_config, n_val)
                    
                    # Test OOD split
                    test_ood_config = copy.deepcopy(config)
                    test_ood_config["saving_path"] = test_ood_config["saving_path"].replace("test", "test_ood")
                    generate_equal_balance_from_transforms(test_ood_config, n_test)
                    
                else:
                    print(f"Saving path {config['saving_path']} not recognized.")
            
            end = time.time()
            print(f"Time taken: {end - start_time} seconds")
            time_dict[config["saving_path"]] = end - start_time

    # convert the dict to a dataframe and save as csv
    df = pandas.DataFrame(list(time_dict.items()), columns=['config_path', 'time_taken_seconds'])
    df.to_csv("generation_times.csv", index=False)

    ## For sample efficiency - comment out the above and uncomment the below

    # n_train = [10, 20, 30, 40]
    # n_test = 10
    # n_val = 10

    # print("wtf is going on")
    # print(sample_efficiency_configs)

    # for config in tqdm(sample_efficiency_configs):
    #     print(config)
    #     for exp_setting in range(1, len(n_train) + 1):
    #         updated_config = copy.deepcopy(config)
    #         updated_config["saving_path"] = updated_config["saving_path"].replace("exp_setting_1", f"exp_setting_{exp_setting}")

    #         train_config = copy.deepcopy(updated_config)
    #         train_config["saving_path"] = train_config["saving_path"].replace("train", "train")
            
    #         val_config = copy.deepcopy(updated_config)
    #         val_config["saving_path"] = val_config["saving_path"].replace("train", "val")

    #         test_config = copy.deepcopy(updated_config)
    #         test_config["saving_path"] = test_config["saving_path"].replace("train", "test")

    #         generate_equal_balance_from_transforms(train_config, n_train[exp_setting - 1])
    #         generate_equal_balance_from_transforms(val_config, n_val)
    #         generate_equal_balance_from_transforms(test_config, n_test)
            


