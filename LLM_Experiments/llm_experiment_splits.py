from datasets import load_dataset
import pandas as pd
import random
import numpy as np


base_path = "../cogita-hf-new/COGITAO"

def open_parquet(file_path):
    return pd.read_parquet(file_path).to_dict(orient="records")

def print_summary(experiment_llm_data):
    """
    Reads and prints the transformation suites across all experiments and samples.
    Mimics the structure of the user's example but supports nested samples.
    """
    transformation_summary = {}

    # Loop over experiments
    for exp_num, samples in experiment_llm_data.items():
        transformation_summary[exp_num] = {"in_context": {}, "test": {}}

        # Loop over each sample
        for sample_idx, sample in samples.items():

            # ---- IN-CONTEXT ----
            for item in sample.get("in_context", []):
                ts_tuple = tuple(item["transformation_suite"])
                transformation_summary[exp_num]["in_context"][ts_tuple] = (
                    transformation_summary[exp_num]["in_context"].get(ts_tuple, 0) + 1
                )

            # ---- TEST ----
            for item in sample.get("test", []):
                ts_tuple = tuple(item["transformation_suite"])
                transformation_summary[exp_num]["test"][ts_tuple] = (
                    transformation_summary[exp_num]["test"].get(ts_tuple, 0) + 1
                )

    # -------- PRINT SUMMARY --------
    print("Transformation Summary:")
    for exp_num in transformation_summary.keys():
        print(f"\nExperiment {exp_num}:")
        
        print(" In-Context Examples:")
        if transformation_summary[exp_num]["in_context"]:
            for ts, count in transformation_summary[exp_num]["in_context"].items():
                print(f"  Transformation Suite {ts}: {count} samples")
        else:
            print("  None")

        print(" Test Examples:")
        if transformation_summary[exp_num]["test"]:
            for ts, count in transformation_summary[exp_num]["test"].items():
                print(f"  Transformation Suite {ts}: {count} samples")
        else:
            print("  None")

        print()

def print_transformation_suites_per_sample(experiment_llm_data):
    """
    Prints all transformation suites per experiment and per sample.
    Includes task_key for each in-context and test example.
    """

    print("Transformation Suites (Per Sample):")

    for exp_num, samples in experiment_llm_data.items():
        print(f"\n==============================")
        print(f"      Experiment {exp_num}")
        print(f"==============================")

        for sample_idx, sample in samples.items():
            print(f"\n  Sample {sample_idx}:")

            # ---- In-context suites ----
            print("   In-Context:")
            if len(sample.get("in_context", [])) == 0:
                print("     None")
            else:
                for i, item in enumerate(sample["in_context"]):
                    ts = item["transformation_suite"]
                    task_key = item.get("task_key", None)
                    print(f"     [{i}] task_key={task_key} | transformation_suite={ts}")

            # ---- Test suites ----
            print("   Test:")
            if len(sample.get("test", [])) == 0:
                print("     None")
            else:
                for i, item in enumerate(sample["test"]):
                    ts = item["transformation_suite"]
                    task_key = item.get("task_key", None)
                    print(f"     [{i}] task_key={task_key} | transformation_suite={ts}")

        print()  # spacing between experiments

### HELPER FUNCTIONS TO HAVE THE TASK CODE INSTEAD OF THE RAW TRANSFORMATION NAMES ### 

def get_all_transformations_from_experiment(file):
    "gets all single transformations from a given experiment file"
    dataset = pd.read_parquet(file).to_dict(orient="records")
    transformations = set()
    for data_point in dataset:
        if len(data_point['transformation_suite']) > 1:
            for transform in data_point['transformation_suite']:
                transformations.add(transform)
        else:
            transformations.add(data_point['transformation_suite'][0])
    return list(transformations)

def create_transformation_mapping(transformations):
    transformation_mapping = {}
    for idx, transform in enumerate(transformations):
        transformation_mapping[transform] = f"t{idx + 1}"
    return transformation_mapping


def map_transformation_suite_to_task_codes(transformation_suite, transformation_mapping):
    task_codes = []
    for transform in transformation_suite:
        if transform in transformation_mapping:
            task_codes.append(transformation_mapping[transform])
        else:
            print("Unknown transformation:", transform)
            task_codes.append("unknown")
    return task_codes





def generate_ID1(
        n_samples = 10, 
        n_context = 5, 
        exps_to_try = [1,2,3,4,5], 
        seed = 42, 
        print_level = "simple"):
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        file = f"{base_path}/CompGen/exp_setting_1/experiment_{exp_num}/test.parquet"
        dataset = open_parquet(file)
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(file))

        random.shuffle(dataset)
        counter = 0 

        for sample_idx in range(n_samples):

            experiment_llm_data[exp_num][sample_idx] = {"in_context": [], "test": [], 
                                                        "transformation_suite": [], "task_key": []}

            sample_finished = False

            transformation_list = []
            unique_transformations = []
            double_transformations = []

            while sample_finished == False:
                
                input = [arr.tolist() for arr in dataset[counter]["input"]]
                output = [arr.tolist() for arr in dataset[counter]["output"]]
                trans_i = dataset[counter]["transformation_suite"].tolist()
                
                if len(experiment_llm_data[exp_num][sample_idx]["in_context"]) < n_context:

                    ok = False 

                    if len(trans_i) == 2 and len(double_transformations) < 2 and trans_i not in double_transformations:
                        double_transformations.append(trans_i)
                        ok = True
                    elif len(trans_i) == 1 and len(unique_transformations) < 3 and trans_i not in unique_transformations:
                        unique_transformations.append(trans_i)
                        ok = True
                    else:
                        ok = False
                        
                    if ok:
                        experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                            "input": input,
                            "output": output,
                            "transformation_suite": trans_i,
                            "task_key": dataset[counter]["task_key"],
                            "coded_transformation_suite": map_transformation_suite_to_task_codes(trans_i, 
                                                                                                transformation_mapping)
                        })
                        transformation_list.append(trans_i)

                elif len(experiment_llm_data[exp_num][sample_idx]["in_context"]) == n_context and \
                    trans_i in transformation_list and \
                        len(trans_i) == 2: # Transformation seen in in-context and of size 2
                    experiment_llm_data[exp_num][sample_idx]["test"].append({
                        "input": input,
                        "output": output,
                        "transformation_suite": trans_i,
                        "task_key": dataset[counter]["task_key"],
                        "coded_transformation_suite": map_transformation_suite_to_task_codes(trans_i, 
                                                                                             transformation_mapping)
                    })
                    sample_finished = True
                
                counter += 1
                if counter >= len(dataset):
                    counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition

    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)
    return experiment_llm_data




def generate_ID2(
        n_samples=10,
        n_context=5,
        exps_to_try=[1,2,3,4,5],
        seed=42,
        print_level="summary"
    ):
    
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        dataset = open_parquet(f"{base_path}/CompGen/exp_setting_2/experiment_{exp_num}/test.parquet")
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(f"{base_path}/CompGen/exp_setting_2/experiment_{exp_num}/test.parquet"))
        random.shuffle(dataset)
        counter = 0 

        for sample_idx in range(n_samples):

            experiment_llm_data[exp_num][sample_idx] = {
                "in_context": [],
                "test": [],
                "transformation_suite": [],
                "coded_transformation_suite": [],
                "task_key": []
            }

            seen_transformations = []

            # -------------------------
            # 1. Pick in-context examples
            # -------------------------
            for i in range(n_context):

                input_i  = [arr.tolist() for arr in dataset[counter]["input"]]
                output_i = [arr.tolist() for arr in dataset[counter]["output"]]
                trans_i  = dataset[counter]["transformation_suite"].tolist()
                coded_trans_i = map_transformation_suite_to_task_codes(trans_i, transformation_mapping)
                task_i   = dataset[counter]["task_key"]

                seen_transformations.append(tuple(trans_i))  # store for lookup

                experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                    "input": input_i,
                    "output": output_i,
                    "transformation_suite": trans_i,
                    "coded_transformation_suite": coded_trans_i,
                    "task_key": task_i
                })

                counter += 1

            # -----------------------------------------
            # 2. Find a test sample with seen transform
            # -----------------------------------------
            found = False
            while counter < len(dataset) and not found:

                input_t  = [arr.tolist() for arr in dataset[counter]["input"]]
                output_t = [arr.tolist() for arr in dataset[counter]["output"]]
                trans_t  = dataset[counter]["transformation_suite"].tolist()
                coded_trans_t = map_transformation_suite_to_task_codes(trans_t, transformation_mapping)
                task_t   = dataset[counter]["task_key"]

                if tuple(trans_t) in seen_transformations:
                    # valid in-distribution test example
                    experiment_llm_data[exp_num][sample_idx]["test"].append({
                        "input": input_t,
                        "output": output_t,
                        "transformation_suite": trans_t,
                        "coded_transformation_suite": coded_trans_t,
                        "task_key": task_t
                    })
                    found = True

                counter += 1

            if not found:
                raise RuntimeError(
                    f"No in-distribution test sample found for experiment {exp_num}, set {sample_idx}."
                )

    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)

    return experiment_llm_data 

def generate_OOD1(n_samples=10,
        n_context=5, # This parameter is currently unused but kept for context
        exps_to_try=[1,2,3,4,5],
        seed=42,
        print_level="summary"
    ):
    
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        in_context_dataset = open_parquet(f"{base_path}/CompGen/exp_setting_1/experiment_{exp_num}/test.parquet")
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(f"{base_path}/CompGen/exp_setting_1/experiment_{exp_num}/test.parquet"))
        ood_dataset = open_parquet(f"{base_path}/CompGen/exp_setting_1/experiment_{exp_num}/test_ood.parquet")    
    
        random.shuffle(in_context_dataset)
        random.shuffle(ood_dataset)

        counter = 0 
        # Add a check for max data size to prevent IndexErrors
        max_in_context_len = len(in_context_dataset) 

        for sample_idx in range(n_samples):
            # Check if there is enough data remaining for a full loop
            if counter >= max_in_context_len:
                 # You might want to break or log an error if you run out of data or restart
                print(f"Warning: Ran out of in-context data for experiment {exp_num}. Resetting counter")
                counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition
                #  break 

            experiment_llm_data[exp_num][sample_idx] = {
                "in_context": [],
                "test": [],
                "transformation_suite": [],
                "coded_transformation_suite": [],
                "task_key": []
            }

            # -------------------------
            # 1. Pick in-context examples
            # -------------------------

            unique_transformations = []
            double_transformations = []
            
            # Change loop condition to check the total count OR if we run out of data
            while len(experiment_llm_data[exp_num][sample_idx]["in_context"]) < n_context:

                if counter >= max_in_context_len:
                    print(f"Warning: Ran out of in-context data for experiment {exp_num} while collecting in-context samples.")
                    counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition
                
                input_i  = [arr.tolist() for arr in in_context_dataset[counter]["input"]]
                output_i = [arr.tolist() for arr in in_context_dataset[counter]["output"]]
                trans_i  = in_context_dataset[counter]["transformation_suite"].tolist()
                coded_trans_i = map_transformation_suite_to_task_codes(trans_i, transformation_mapping)
                task_i   = in_context_dataset[counter]["task_key"]
                
                # Flag to check if the sample was accepted
                sample_accepted = False 

                # Prioritize collecting double transformations if we still need them
                if len(trans_i) == 2 and len(double_transformations) < 2 and trans_i not in double_transformations:
                    double_transformations.append(trans_i)
                    sample_accepted = True

                # Then, check for unique transformations if we still need them
                elif len(trans_i) == 1 and len(unique_transformations) < 3 and trans_i not in unique_transformations:
                    unique_transformations.append(trans_i)                    
                    sample_accepted = True
                
                # If the sample was accepted, add it to the list
                if sample_accepted:
                    experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                        "input": input_i,
                        "output": output_i,
                        "transformation_suite": trans_i,
                        "coded_transformation_suite": coded_trans_i,
                        "task_key": task_i
                    })

                # CRITICAL: Always increment the counter to move to the next data point.
                counter += 1

            # -----------------------------------------
            # 2. Test Sample
            # -----------------------------------------
            
            # Ensure we don't exceed the bounds of ood_dataset
            
            random_idx = random.randint(0, len(ood_dataset) - 1)

            input_t  = [arr.tolist() for arr in ood_dataset[random_idx]["input"]]
            output_t = [arr.tolist() for arr in ood_dataset[random_idx]["output"]]
            trans_t  = ood_dataset[random_idx]["transformation_suite"].tolist()
            coded_trans_t = map_transformation_suite_to_task_codes(trans_t, transformation_mapping)
            task_t   = ood_dataset[random_idx]["task_key"]

            experiment_llm_data[exp_num][sample_idx]["test"].append({
                "input": input_t,
                "output": output_t,
                "transformation_suite": trans_t,
                "coded_transformation_suite": coded_trans_t,
                "task_key": task_t
            })

    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)

    return experiment_llm_data

def generate_OOD2(n_samples=10,
        n_context=5, # This parameter is currently unused but kept for context
        exps_to_try=[1,2,3,4,5],
        seed=42,
        print_level="summary"
    ):
    
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        in_context_dataset = open_parquet(f"{base_path}/CompGen/exp_setting_2/experiment_{exp_num}/test.parquet")
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(f"{base_path}/CompGen/exp_setting_2/experiment_{exp_num}/test.parquet"))
        ood_dataset = open_parquet(f"{base_path}/CompGen/exp_setting_2/experiment_{exp_num}/test_ood.parquet")    
    
        random.shuffle(in_context_dataset)
        random.shuffle(ood_dataset)

        counter = 0 
        # Add a check for max data size to prevent IndexErrors
        max_in_context_len = len(in_context_dataset) 

        for sample_idx in range(n_samples):
            # Check if there is enough data remaining for a full loop
            if counter >= max_in_context_len:
                 # You might want to break or log an error if you run out of data
                 print(f"Warning: Ran out of in-context data for experiment {exp_num}. Resetting counter")
                 counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition

            experiment_llm_data[exp_num][sample_idx] = {
                "in_context": [],
                "test": [],
                "transformation_suite": [],
                "coded_transformation_suite": [],
                "task_key": []
            }

            # -------------------------
            # 1. Pick in-context examples
            # -------------------------

            transformations = []
            
            # Change loop condition to check the total count OR if we run out of data
            while len(experiment_llm_data[exp_num][sample_idx]["in_context"]) < n_context:

                if counter >= max_in_context_len:
                    print(f"Warning: Ran out of in-context data for experiment {exp_num} while collecting in-context samples.")
                    counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition
                
                input_i  = [arr.tolist() for arr in in_context_dataset[counter]["input"]]
                output_i = [arr.tolist() for arr in in_context_dataset[counter]["output"]]
                trans_i  = in_context_dataset[counter]["transformation_suite"].tolist()
                coded_trans_i = map_transformation_suite_to_task_codes(trans_i, transformation_mapping)
                task_i   = in_context_dataset[counter]["task_key"]

                # Prioritize collecting double transformations if we still need them
                if trans_i not in transformations:
                    transformations.append(trans_i)

                    experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                        "input": input_i,
                        "output": output_i,
                        "transformation_suite": trans_i,
                        "coded_transformation_suite": coded_trans_i,
                        "task_key": task_i
                    })

                # CRITICAL: Always increment the counter to move to the next data point.
                counter += 1

            # -----------------------------------------
            # 2. Test Sample
            # -----------------------------------------
            
            # Ensure we don't exceed the bounds of ood_dataset
            
            random_idx = random.randint(0, len(ood_dataset) - 1)

            input_t  = [arr.tolist() for arr in ood_dataset[random_idx]["input"]]
            output_t = [arr.tolist() for arr in ood_dataset[random_idx]["output"]]
            trans_t  = ood_dataset[random_idx]["transformation_suite"].tolist()
            coded_trans_t = map_transformation_suite_to_task_codes(trans_t, transformation_mapping)
            task_t   = ood_dataset[random_idx]["task_key"]

            experiment_llm_data[exp_num][sample_idx]["test"].append({
                "input": input_t,
                "output": output_t,
                "transformation_suite": trans_t,
                "coded_transformation_suite": coded_trans_t,
                "task_key": task_t
            })

    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)

    return experiment_llm_data

def generate_OOD3(n_samples=10,
        n_context=5, 
        exps_to_try=[1,2,3,4,5],
        seed=42,
        print_level="summary"
    ):
    
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        in_context_dataset = open_parquet(f"{base_path}/CompGen/exp_setting_3/experiment_{exp_num}/test.parquet")
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(f"{base_path}/CompGen/exp_setting_3/experiment_{exp_num}/test.parquet"))
        ood_dataset = open_parquet(f"{base_path}/CompGen/exp_setting_3/experiment_{exp_num}/test_ood.parquet")    
    
        random.shuffle(in_context_dataset)
        random.shuffle(ood_dataset)

        counter = 0 
        # Add a check for max data size to prevent IndexErrors
        max_in_context_len = len(in_context_dataset) 

        for sample_idx in range(n_samples):
            # Check if there is enough data remaining for a full loop
            if counter >= max_in_context_len:
                 # You might want to break or log an error if you run out of data
                 print(f"Warning: Ran out of in-context data for experiment {exp_num}.")
                 counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition 

            experiment_llm_data[exp_num][sample_idx] = {
                "in_context": [],
                "test": [],
                "transformation_suite": [],
                "coded_transformation_suite": [],
                "task_key": []
            }

            # -------------------------
            # 1. Pick in-context examples
            # -------------------------

            unique_transformations = []
            double_transformations = []
            
            # Change loop condition to check the total count OR if we run out of data
            while len(experiment_llm_data[exp_num][sample_idx]["in_context"]) < n_context:

                if counter >= max_in_context_len:
                    print(f"Warning: Ran out of in-context data for experiment {exp_num} while collecting in-context samples.")
                    counter = np.random.randint(1, 10) # Change starting point of the counter to avoid repetition
                
                input_i  = [arr.tolist() for arr in in_context_dataset[counter]["input"]]
                output_i = [arr.tolist() for arr in in_context_dataset[counter]["output"]]
                trans_i  = in_context_dataset[counter]["transformation_suite"].tolist()
                coded_trans_i = map_transformation_suite_to_task_codes(trans_i, transformation_mapping)
                task_i   = in_context_dataset[counter]["task_key"]
                
                # Flag to check if the sample was accepted
                sample_accepted = False 

                # Prioritize collecting double transformations if we still need them
                if len(trans_i) == 2 and len(double_transformations) < 2 and trans_i not in double_transformations:
                    double_transformations.append(trans_i)
                    sample_accepted = True

                # Then, check for unique transformations if we still need them
                elif len(trans_i) == 1 and len(unique_transformations) < 3 and trans_i not in unique_transformations:
                    unique_transformations.append(trans_i)                    
                    sample_accepted = True
                
                # If the sample was accepted, add it to the list
                if sample_accepted:
                    experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                        "input": input_i,
                        "output": output_i,
                        "transformation_suite": trans_i,
                        "coded_transformation_suite": coded_trans_i,
                        "task_key": task_i
                    })

                # CRITICAL: Always increment the counter to move to the next data point.
                counter += 1

            # -----------------------------------------
            # 2. Test Sample
            # -----------------------------------------
            
            # Ensure we don't exceed the bounds of ood_dataset
            
            random_idx = random.randint(0, len(ood_dataset) - 1)

            input_t  = [arr.tolist() for arr in ood_dataset[random_idx]["input"]]
            output_t = [arr.tolist() for arr in ood_dataset[random_idx]["output"]]
            trans_t  = ood_dataset[random_idx]["transformation_suite"].tolist()
            coded_trans_t = map_transformation_suite_to_task_codes(trans_t, transformation_mapping)
            task_t   = ood_dataset[random_idx]["task_key"]

            experiment_llm_data[exp_num][sample_idx]["test"].append({
                "input": input_t,
                "output": output_t,
                "transformation_suite": trans_t,
                "coded_transformation_suite": coded_trans_t,
                "task_key": task_t
            })

    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)

    return experiment_llm_data

def generate_envGen_objects(n_samples=10,
        n_context=5, 
        exps_to_try=[1,2,3,4,5],
        seed=42,
        print_level="summary"
    ):
    
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        in_context_dataset = open_parquet(f"{base_path}/EnvGen/exp_setting_1/experiment_{exp_num}/test.parquet")
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(f"{base_path}/EnvGen/exp_setting_1/experiment_{exp_num}/test.parquet"))
        ood_dataset = open_parquet(f"{base_path}/EnvGen/exp_setting_1/experiment_{exp_num}/test_ood.parquet")    
    
        random.shuffle(in_context_dataset)
        random.shuffle(ood_dataset)

        counter = 0 
        # Add a check for max data size to prevent IndexErrors
        max_in_context_len = len(in_context_dataset) 

        for sample_idx in range(n_samples):
            # Check if there is enough data remaining for a full loop
            if counter >= max_in_context_len:
                 # You might want to break or log an error if you run out of data
                 print(f"Warning: Ran out of in-context data for experiment {exp_num}.")
                 break 

            experiment_llm_data[exp_num][sample_idx] = {
                "in_context": [],
                "test": [],
                "transformation_suite": [],
                "coded_transformation_suite": [],
                "task_key": []
            }

            for i in range(n_context):
                input_i  = [arr.tolist() for arr in in_context_dataset[counter]["input"]]
                output_i = [arr.tolist() for arr in in_context_dataset[counter]["output"]]
                task_i   = in_context_dataset[counter]["task_key"]
                trans_i = in_context_dataset[counter]["transformation_suite"].tolist()
                coded_trans_i = map_transformation_suite_to_task_codes(trans_i, transformation_mapping)

                experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                        "input": input_i,
                        "output": output_i,
                        "transformation_suite": trans_i,
                        "coded_transformation_suite": coded_trans_i,
                        "task_key": task_i
                    })
                counter += 1

            # -----------------------------------------
            # 2. Test Sample
            # -----------------------------------------

            random_idx = random.randint(0, len(ood_dataset) - 1)

            input_t  = [arr.tolist() for arr in ood_dataset[random_idx]["input"]]
            output_t = [arr.tolist() for arr in ood_dataset[random_idx]["output"]]
            trans_t  = ood_dataset[random_idx]["transformation_suite"].tolist()
            coded_trans_t = map_transformation_suite_to_task_codes(trans_t, transformation_mapping)
            task_t   = ood_dataset[random_idx]["task_key"]

            experiment_llm_data[exp_num][sample_idx]["test"].append({
                "input": input_t,
                "output": output_t,
                "transformation_suite": trans_t,
                "coded_transformation_suite": coded_trans_t,
                "task_key": task_t
            })
            
    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)
    return experiment_llm_data

def generate_envGen_size(n_samples=10,
        n_context=5, # This parameter is currently unused but kept for context
        exps_to_try=[1,2,3,4,5],
        seed=42,
        print_level="summary"
    ):
    
    random.seed(seed)
    experiment_llm_data = {}

    for exp_num in exps_to_try:
        experiment_llm_data[exp_num] = {}

        in_context_dataset = open_parquet(f"{base_path}/EnvGen/exp_setting_2/experiment_{exp_num}/test.parquet")
        transformation_mapping = create_transformation_mapping(get_all_transformations_from_experiment(f"{base_path}/EnvGen/exp_setting_2/experiment_{exp_num}/test.parquet"))
        ood_dataset = open_parquet(f"{base_path}/EnvGen/exp_setting_2/experiment_{exp_num}/test_ood.parquet")    
    
        random.shuffle(in_context_dataset)
        random.shuffle(ood_dataset)

        counter = 0 
        # Add a check for max data size to prevent IndexErrors
        max_in_context_len = len(in_context_dataset) 

        for sample_idx in range(n_samples):
            # Check if there is enough data remaining for a full loop
            if counter >= max_in_context_len:
                 # You might want to break or log an error if you run out of data
                 print(f"Warning: Ran out of in-context data for experiment {exp_num}.")
                 break 

            experiment_llm_data[exp_num][sample_idx] = {
                "in_context": [],
                "test": [],
                "transformation_suite": [],
                "coded_transformation_suite": [],
                "task_key": []
            }

            for i in range(n_context):
                input_i  = [arr.tolist() for arr in in_context_dataset[counter]["input"]]
                output_i = [arr.tolist() for arr in in_context_dataset[counter]["output"]]
                task_i   = in_context_dataset[counter]["task_key"]
                trans_i = in_context_dataset[counter]["transformation_suite"].tolist()
                coded_trans_i = map_transformation_suite_to_task_codes(trans_i, transformation_mapping)

                experiment_llm_data[exp_num][sample_idx]["in_context"].append({
                        "input": input_i,
                        "output": output_i,
                        "transformation_suite": trans_i,
                        "coded_transformation_suite": coded_trans_i,
                        "task_key": task_i
                    })
                counter += 1

            # -----------------------------------------
            # 2. Test Sample
            # -----------------------------------------

            random_idx = random.randint(0, len(ood_dataset) - 1)

            input_t  = [arr.tolist() for arr in ood_dataset[random_idx]["input"]]
            output_t = [arr.tolist() for arr in ood_dataset[random_idx]["output"]]
            trans_t  = ood_dataset[random_idx]["transformation_suite"].tolist()
            coded_trans_t = map_transformation_suite_to_task_codes(trans_t, transformation_mapping)
            task_t   = ood_dataset[random_idx]["task_key"]

            experiment_llm_data[exp_num][sample_idx]["test"].append({
                "input": input_t,
                "output": output_t,
                "transformation_suite": trans_t,
                "coded_transformation_suite": coded_trans_t,
                "task_key": task_t
            })
            
    if print_level == "detailed":
        print_transformation_suites_per_sample(experiment_llm_data)
    elif print_level == "summary":
        print_summary(experiment_llm_data)
    return experiment_llm_data




