import numpy as np
import ast
import requests
import json
import os
import pandas as pd


# BASE_PROMPT_IDCOMP = "You are a system specialized in solving object-transformation-based puzzles. " \
# "You are given several input–output pairs represented as ASCII grids, where each number corresponds to a color, which make up objects. " \
# "For each pair, a set of transformations is provided that explains how the input is converted into the output. " \
# "Your job is to infer the transformation logic from these examples and then apply a specified subset of these transformations " \
# "to the test input. Apply the transformation to the test input accordingly. "\
# "Output ONLY the resulting grid as a Python list of lists. No text before or after. No explanations. No markdown code blocks. Just the raw list.\n"

# BASE_PROMPT_OODCOMP = "You are a system specialized in solving object-transformation-based puzzles. " \
# "You are given several input–output pairs represented as ASCII grids, where each number corresponds to a color, which make up objects. " \
# "For each pair, a set of transformations is provided that explains how the input is converted into the output. " \
# "Your job is to infer the transformation logic from these examples and then apply a specified subset of these transformations " \
# "to the test input. Importantly, the test input has a different transformation sequence than the ones seen examples of. Apply the transformation " \
# "to the test input accordingly. " \
# "Output ONLY the resulting grid as a Python list of lists. No text before or after. No explanations. No markdown code blocks. Just the raw list.\n"

# BASE_PROMPT_OBJGEN = "You are a system specialized in solving object-transformation-based puzzles. " \
# "You are given several input–output pairs represented as ASCII grids, where each number corresponds to a color, which make up objects. " \
# "For each pair, a set of transformations is provided that explains how the input is converted into the output. " \
# "Your job is to infer how transformation impact objects from these examples and then apply the same transformation " \
# "to the test input. Importantly, the test input has more objects per grid than what you have seen examples of. Apply the transformation "\
# "to the test input accordingly. " \
# "Output ONLY the resulting grid as a Python list of lists. No text before or after. No explanations. No markdown code blocks. Just the raw list.\n"

# BASE_PROMPT_GRIDGEN = "You are a system specialized in solving object-transformation-based puzzles. " \
# "You are given several input–output pairs represented as ASCII grids, where each number corresponds to a color, which make up objects. " \
# "For each pair, a set of transformations is provided that explains how the input is converted into the output. " \
# "Your job is to infer how transformation impact objects from these examples and then apply the same transformation " \
# "to the test input. Importantly, the test input is of different size than what you have seen examples of. Apply the transformation "\
# "to the test input accordingly. " \
# "Output ONLY the resulting grid as a Python list of lists. No text before or after. No explanations. No markdown code blocks. Just the raw list.\n"



BASE_PROMPT_IDCOMP = """You are a system specialized in solving object-transformation puzzles on ASCII grids.

TASK:
1. You will see several input–output pairs as ASCII grids (numbers represent colors/objects)
2. Each pair includes labeled transformations explaining how input becomes output
3. You must infer what each transformation does from these examples
4. For the test input, you will be told which specific transformations to apply.
5. Apply ONLY the specified transformations to the test input in the order given

OUTPUT FORMAT:
Return ONLY the resulting grid as a Python list of lists.
- No explanations
- No markdown code blocks
- No text before or after
- Just the raw list, e.g.: [[0, 1], [1, 0]]
"""

BASE_PROMPT_OODCOMP = """You are a system specialized in solving object-transformation puzzles on ASCII grids.

TASK:
1. You will see several input–output pairs as ASCII grids (numbers represent colors/objects)
2. Each pair includes labeled transformations explaining how input becomes output
3. You must infer what each transformation does from these examples
4. For the test input, you will be told which specific transformations to apply (possibly in a new combination not seen during training).
"IMPORTANT: The test uses out-of-distribution composition. Each individual transformation appears in the training examples, but the test requires combining them in a way not seen during training. You must:
- Identify what each transformation does independently
- Apply the specified transformations to the test input, even though this exact combination is new"
5. Apply ONLY the specified transformations to the test input in the order given

OUTPUT FORMAT:
Return ONLY the resulting grid as a Python list of lists.
- No explanations
- No markdown code blocks
- No text before or after
- Just the raw list, e.g.: [[0, 1], [1, 0]]
"""

BASE_PROMPT_OBJGEN = """You are a system specialized in solving object-transformation puzzles on ASCII grids.

TASK:
1. You will see several input–output pairs as ASCII grids (numbers represent colors/objects)
2. Each pair includes labeled transformations explaining how input becomes output
3. You must infer how each transformation affects individual objects
4. Apply the same transformation(s) to the test input

IMPORTANT: The test evaluates object-count generalization. The transformation logic is identical to training, but the test input contains more objects than any training example. You must apply the learned transformation to each object, regardless of how many are present.

OUTPUT FORMAT:
Return ONLY the resulting grid as a Python list of lists.
- No explanations
- No markdown code blocks
- No text before or after
- Just the raw list, e.g.: [[0, 1], [1, 0]]
"""

BASE_PROMPT_GRIDGEN = """You are a system specialized in solving object-transformation puzzles on ASCII grids.

TASK:
1. You will see several input–output pairs as ASCII grids (numbers represent colors/objects)
2. Each pair includes labeled transformations explaining how input becomes output
3. You must infer how each transformation affects objects and the grid
4. Apply the same transformation(s) to the test input

IMPORTANT: The test evaluates grid-size generalization. The transformation logic is identical to training, but the test grid is a different size than any training example. You must apply the learned transformation correctly regardless of grid dimensions.

OUTPUT FORMAT:
Return ONLY the resulting grid as a Python list of lists.
- No explanations
- No markdown code blocks
- No text before or after
- Just the raw list, e.g.: [[0, 1], [1, 0]]
"""



def make_prompts(data, base_prompt, name, task_embedding = "original"):

    prompts = {}

    transformation_type = "transformation_suite" if task_embedding == "original" else "coded_transformation_suite"

    for experiment_id in list(data.keys()):

        for sample in list(data[experiment_id].keys()):

            prompt_id = f"{name}-{experiment_id}-{sample}"
            
            prompts[prompt_id] = {"prompt": base_prompt, "ground_truth": [], "test_input": []}

            sample = data[experiment_id][sample]

            for i in range(len(sample["in_context"])):
                input_grid = sample["in_context"][i]["input"]
                output_grid = sample["in_context"][i]["output"]
                transformations = sample["in_context"][i][transformation_type]
                
                prompts[prompt_id]["prompt"] += f"Here is a new example with transformations: {transformations}.\n"
                prompts[prompt_id]["prompt"] += f"Input {i + 1}" + str(input_grid) + "\n"
                prompts[prompt_id]["prompt"] += f"Output {i + 1}" + str(output_grid) + "\n"
            
            test_input = sample["test"][0]["input"]
            test_output = sample["test"][0]["output"]
            test_transformations = sample["test"][0][transformation_type]

            prompts[prompt_id]["prompt"] += f"Now, apply the following transformations: {test_transformations} to the test input below.\n"
            prompts[prompt_id]["prompt"] += "Test Input: " + str(test_input) + "\n"
            prompts[prompt_id]["test_input"].append(test_input)
            prompts[prompt_id]["ground_truth"].append(test_output)
    
    return prompts

# Convert string of list of lists to actual list of lists
def parse_llm_response(response_str):
    try:
        response_list = ast.literal_eval(response_str)
        return response_list
    except:
        return None


def check_response(ground_truth, llm_answer):
    gt = np.array(ground_truth)
    llm_answer = parse_llm_response(llm_answer)
    
    if llm_answer is None:
        return -1, -1, -1

    # Check if outputs match equally
    llm_ans = np.array(llm_answer)

    # Normalize llm_ans dimension by 1) assessing it's of the same shape as gt.
    if len(llm_ans.shape) == 3:
        llm_ans = llm_ans.squeeze()
    
    if len(gt.shape) == 3:
        gt = gt.squeeze()

    if llm_ans.shape != gt.shape:
        return -1, -1, -1

    correct_pixels = (gt == llm_ans)

    # Check overall accuracy
    accuracy = int(np.all(correct_pixels))
    per_pixel_accuracy = np.mean(correct_pixels)

    # Check overall accuracy of non-zero pixels
    non_zero_pixels = (gt != 0)
    object_accuracy = np.mean(correct_pixels[non_zero_pixels]) if np.any(non_zero_pixels) else 0.0

    return accuracy, per_pixel_accuracy, object_accuracy


def call_llm_api(prompt, model):
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_KEY')}",
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps({
        "model": model, # Optional
        "messages": [
        {
            "role": "user",
            "content": prompt,
        }
        ]
    })
    )
    return response.json()["choices"][0]["message"]["content"]