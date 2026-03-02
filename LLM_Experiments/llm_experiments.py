import time
import pandas as pd
import numpy as np
import os
from llm_experiments_utils import make_prompts, parse_llm_response, check_response
from llm_experiments_utils import call_llm_api
from llm_experiments_utils import BASE_PROMPT_IDCOMP, BASE_PROMPT_OODCOMP, BASE_PROMPT_GRIDGEN, BASE_PROMPT_OBJGEN
from llm_experiment_splits import generate_ID1, generate_ID2
from llm_experiment_splits import generate_OOD1, generate_OOD2, generate_OOD3
from llm_experiment_splits import generate_envGen_size, generate_envGen_objects
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Ensure the results directory exists
os.makedirs("llm_results", exist_ok=True)

# Lock for thread-safe CSV writing
csv_lock = threading.Lock()

def process_and_save(prompt_id, prompt_data, model, output_filename, file_exists_flag):
    """Process a single prompt AND save the result immediately."""
    start = time.time()
    try:
        llm_answer = call_llm_api(prompt_data["prompt"], model=model)
    except Exception as e:
        print(f"API Error for {prompt_id}: {e}")
        llm_answer = f"ERROR: {e}"
    end = time.time()
    
    gt = prompt_data["ground_truth"]
    test_input = prompt_data["test_input"]
    is_correct, acc, object_acc = check_response(ground_truth=gt, llm_answer=llm_answer)

    time_taken = np.round(end - start, 2)
    printed_acc = np.round(object_acc, 3)
    print(f"Completed {prompt_id}: {time_taken}s, Object Accuracy: {printed_acc}")
    
    result = {
        "prompt_id": prompt_id,
        "is_correct": is_correct,
        "accuracy": acc,
        "object_accuracy": object_acc,
        "time_taken_seconds": time_taken,
        "llm_original_answer": llm_answer,
        "ground_truth": gt[0],
        "test_input": test_input,
        "model_name": model
    }
    
    # Save immediately within the worker thread
    with csv_lock:
        temp_df = pd.DataFrame([result])
        write_header = not file_exists_flag[0]
        temp_df.to_csv(output_filename, mode='a', index=False, header=write_header)
        file_exists_flag[0] = True
        print(f"  -> Saved {prompt_id} to CSV")
    
    return result

if __name__ == "__main__":
    in_distribution_1 = generate_ID1(print_level="none", n_samples=10, n_context=5, seed=42)
    # in_distribution_2 = generate_ID2(print_level="none", n_samples=10, n_context=5, seed=42)
    # out_of_distribution_1 = generate_OOD1(print_level="none", n_samples=10, n_context=5, seed=42)
    # out_of_distribution_2 = generate_OOD2(print_level="none", n_samples=10, n_context=5, seed=42)
    # out_of_distribution_3 = generate_OOD3(print_level="none", n_samples=10, n_context=5, seed=42)
    # envgen_size = generate_envGen_size(print_level="none", n_samples=10, n_context=5, seed=42)
    # envgen_objects = generate_envGen_objects(print_level="none", n_samples=10, n_context=5, seed=42)

    experiments_to_run = {
        "ID1": in_distribution_1,
        # "ID2": in_distribution_2,
        # "OOD1": out_of_distribution_1,
        # "OOD2": out_of_distribution_2,
        # "OOD3": out_of_distribution_3,
        # "envGen_size": envgen_size,
        # "envGen_objects": envgen_objects,
    }
    
    base_prompts = {
        "ID1": BASE_PROMPT_IDCOMP,
        "ID2": BASE_PROMPT_IDCOMP,
        "OOD1": BASE_PROMPT_OODCOMP,
        "OOD2": BASE_PROMPT_OODCOMP,
        "OOD3": BASE_PROMPT_OODCOMP,
        "envGen_size": BASE_PROMPT_GRIDGEN,
        "envGen_objects": BASE_PROMPT_OBJGEN,
    }

    models = [
        "tngtech/deepseek-r1t2-chimera:free",
        "google/gemini-3-pro-preview",
        # "anthropic/claude-sonnet-4.5",
        "openai/gpt-4.1-mini",
        "x-ai/grok-code-fast-1",
        "openai/o3"
    ]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Running experiments for model: {model}")
        print(f"{'='*60}")
        
        model_name = model.replace("/", "_").replace(":", "_")
        output_filename = f"llm_results_ID1_non_buggy/{model_name}_llm_experiment_results.csv"
        
        file_exists = os.path.exists(output_filename)
        file_exists_flag = [file_exists]
        
        # Collect all prompts first
        all_prompts = []
        for exp_id in list(experiments_to_run.keys()):
            prompt_dict = make_prompts(experiments_to_run[exp_id], 
                                       base_prompt=base_prompts[exp_id], 
                                       name=exp_id, 
                                       task_embedding="original")
            for prompt_id in prompt_dict:
                all_prompts.append((prompt_id, prompt_dict[prompt_id]))
        
        print(f"Total prompts: {len(all_prompts)}")
        
        MAX_WORKERS = 30
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            for i, (prompt_id, prompt_data) in enumerate(all_prompts):
                if i > 0:
                    time.sleep(2)
                
                future = executor.submit(
                    process_and_save,  # Now saves inside the function
                    prompt_id, 
                    prompt_data, 
                    model, 
                    output_filename,
                    file_exists_flag  # Pass the flag to the worker
                )
                futures[future] = prompt_id
                print(f"Submitted {prompt_id} ({i+1}/{len(all_prompts)})")
            
            # Just wait for completion and handle errors
            for future in as_completed(futures):
                prompt_id = futures[future]
                try:
                    future.result()  # Result already saved, just check for exceptions
                except Exception as e:
                    print(f"Error processing {prompt_id}: {e}")
        
        print(f"Finished model: {model}")