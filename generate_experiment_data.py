from arcworld.general_utils import generate_key
import os
import numpy as np
from tqdm import tqdm
import copy
import json
import pandas
import time

from arcworld.utils.db_utils import access_db, close_db, store_task_in_db, hash_task
from arcworld.generator import Generator
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
            gen = Generator(editable_config)
            
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


def generate_paired_splits(id_config, ood_config, n_tasks_to_generate, id_file_path, ood_file_path):
    """Generate paired ID/OOD splits sharing the same input grids.

    For each sample, the same input grid is used with an ID transformation and
    an OOD transformation. Both must succeed for the pair to be kept, ensuring
    strict 1-to-1 alignment between the two output files.

    Balance is maintained across transformation combos on both sides.
    """
    db_name, folder_path, _ = handle_paths(id_config)

    cursor, conn = access_db(db_name, folder_path)

    id_combos = id_config["allowed_combinations"]
    ood_combos = ood_config["allowed_combinations"]
    n_per_id = n_tasks_to_generate // len(id_combos)
    id_remainder = n_tasks_to_generate % len(id_combos)
    n_per_ood = n_tasks_to_generate // len(ood_combos)
    ood_remainder = n_tasks_to_generate % len(ood_combos)

    # Per-combo targets (first `remainder` combos get +1)
    id_combo_targets = {i: n_per_id + (1 if i < id_remainder else 0) for i in range(len(id_combos))}
    ood_combo_targets = {i: n_per_ood + (1 if i < ood_remainder else 0) for i in range(len(ood_combos))}

    # Use id_config for grid generation (grid size, shapes, etc.)
    grid_gen_config = copy.deepcopy(id_config)
    grid_gen_config["allowed_combinations"] = id_combos
    grid_gen = Generator(grid_gen_config)

    id_task_list = []
    ood_task_list = []

    # Track per-combo counts for balance
    id_combo_counts = {i: 0 for i in range(len(id_combos))}
    ood_combo_counts = {i: 0 for i in range(len(ood_combos))}

    max_failures = 100
    consecutive_failures = 0

    while len(id_task_list) < n_tasks_to_generate:
        # Pick the ID combo furthest below its target
        id_combo_idx = max(id_combo_targets, key=lambda i: id_combo_targets[i] - id_combo_counts[i])
        if id_combo_counts[id_combo_idx] >= id_combo_targets[id_combo_idx]:
            break  # All ID combos are full
        id_combo = id_combos[id_combo_idx]

        # Pick the OOD combo furthest below its target
        ood_combo_idx = max(ood_combo_targets, key=lambda i: ood_combo_targets[i] - ood_combo_counts[i])
        if ood_combo_counts[ood_combo_idx] >= ood_combo_targets[ood_combo_idx]:
            break  # All OOD combos are full
        ood_combo = ood_combos[ood_combo_idx]

        # Generate a grid
        grid_result = grid_gen.generate_grid()
        if grid_result is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"Too many consecutive grid generation failures ({max_failures}). Stopping.")
                break
            continue

        input_grid, positioned_shapes = grid_result

        # Try ID transform on this grid
        id_task = grid_gen.generate_task_from_grid(input_grid, positioned_shapes, id_combo)
        if id_task is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"Too many consecutive failures ({max_failures}). Stopping.")
                break
            continue

        # Try OOD transform on the same grid
        ood_task = grid_gen.generate_task_from_grid(input_grid, positioned_shapes, ood_combo)
        if ood_task is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"Too many consecutive failures ({max_failures}). Stopping.")
                break
            continue

        consecutive_failures = 0

        # Dedup both tasks
        id_task_key = generate_key()
        id_ready = adapt_task_format(id_task, id_task_key)
        id_hash = hash_task(id_ready["input"], id_ready["transformation_suite"])

        ood_task_key = generate_key()
        ood_ready = adapt_task_format(ood_task, ood_task_key)
        ood_hash = hash_task(ood_ready["input"], ood_ready["transformation_suite"])

        # Both must be unique to keep the pair
        id_ok = store_task_in_db(cursor, conn, id_task_key, id_hash,
                                  str(id_ready['transformation_suite']), debug=False)
        if not id_ok:
            continue

        ood_ok = store_task_in_db(cursor, conn, ood_task_key, ood_hash,
                                   str(ood_ready['transformation_suite']), debug=False)
        if not ood_ok:
            continue

        id_task_list.append(id_ready)
        ood_task_list.append(ood_ready)
        id_combo_counts[id_combo_idx] += 1
        ood_combo_counts[ood_combo_idx] += 1
        print(f"Generated {len(id_task_list)} / {n_tasks_to_generate} paired tasks", end='\r')

    close_db(conn)

    with open(f"{id_file_path}", "w") as f:
        json.dump(id_task_list, f)
    with open(f"{ood_file_path}", "w") as f:
        json.dump(ood_task_list, f)


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
            


