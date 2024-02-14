import os
import re
import sys
import argparse
import json
import openai
from tqdm import tqdm
from pathlib import Path
from utils.base import list_subdirectories, mkdir, split_list
from utils.listfunc import python_execution, post_process, parse_function, last_square_brackets

def execution(task_name, args):
    execution_lm(task_name, args)
    
def execution_py(task_name, args):

    if task_name == "results":
        return 
    
    with open(f'{args.data_dir}/{task_name}/test.json', encoding='utf-8') as f_examples:
        test_data = json.load(f_examples)

    with open(f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}/transformations.json', encoding='utf-8') as f_examples:
        transformations = json.load(f_examples)

    #openai.organization = openai_organization
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    scores = []
    
    for transformation in transformations:

        transformation = post_process(transformation)
        transformation = parse_function(transformation)

        score = 0
        
        for data in test_data:
            
            target = eval(data["target"])
            result = python_execution(transformation=transformation, input_=data["input"])
            
            if target == result:
                
                score += 1

        scores.append(score/len(test_data))
        
    avg_score = sum(scores)/len(scores)
    
    print(f"task {task_name} python execution score: {avg_score}")

    with open(f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}/execution.json', 'w', encoding='utf-8') as f_predictions:
        json.dump(avg_score, f_predictions)

def execution_lm(task_name, args):

    if task_name == "results":
        return
    
    with open(f'{args.data_dir}/{task_name}/test.json', encoding='utf-8') as f_examples:
        test_data = json.load(f_examples)

    with open(f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}/transformations.json', encoding='utf-8') as f_examples:
        transformations = json.load(f_examples)

    #openai.organization = openai_organization
    openai.api_key = os.getenv("OPENAI_API_KEY")

    scores = []
    
    for transformation in tqdm(transformations):
        
        if transformation.startswith("The transformation The transformation"):
            transformation = transformation.replace("The transformation", "", 1)
            
        if not transformation.startswith("The transformation"):
            transformation = "The transformation" + transformation
        
        print(transformation)
        
        score = 0
        i = 0
        
        while i < len(test_data):
            
            try:
                
                data = test_data[i]
                target = eval(data["target"])
                prompt = f"Instruction: {transformation}\nInput:{data['input']}\nOutput: "
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", 
                        messages=messages,
                        temperature=0,
                        max_tokens=100,
                        top_p=0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                )
                
                i += 1
                
            except (TimeoutError, openai.error.OpenAIError) as e:
                print(e)
                continue
            
            result = response.choices[0]["message"]["content"]
            result = last_square_brackets(result)
            
            try:
                result = eval(result)
            except:
                result = None
            
            if target == result:
                
                score += 1

        scores.append(score/len(test_data))
        
    avg_score = sum(scores)/len(scores)
    
    print(f"task {task_name} python execution score: {avg_score}")

    with open(f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}/execution.json', 'w', encoding='utf-8') as f_predictions:
        json.dump(avg_score, f_predictions)


def select_hypothesis(task_name, codes, samples, args):

    scores = []
    
    samples = [sample.replace("input: ", "").replace("output: ", "").split("\n") for sample in samples]
    
    for code in codes:
        
        code = post_process(code)
        code = parse_function(code)

        score = 0
        
        for data in samples:
            
            target = eval(data[0])
            
            if args.use_deductor_during_induction:
            
                result = python_execution(code, input_=data[1])
            
            else:
                META_PROMPT = """
You are a smart assistant, now please help me predict the output given the input and the python function.\n\
\n\
python function:\n\
def transform(input_list):\n\
    output_list = [9]*len(input_list)\n\
    return output_list\n\
input: [0, 8, 9, 3, 7, 5, 5]\n\
output: [9, 9, 9, 9, 9, 9, 9]\n\
\n\
python function:\n\
def transform(input_list):\n\
    output_list = []\n\
    for i in range(len(input_list)):\n\
        output_list.append(input_list[-(i+1)])\n\
    return output_list\n\
input: [1, 3, 7, 4, 2, 0, 8, 9]\n\
output: [9, 8, 0, 2, 4, 7, 3, 1]\n\
\n\
"""
                try:
                    if code:
                        output_ = args.deductor(META_PROMPT + f"python function: {code}\ninput: {str(data[1])}\noutput:")
                        output_ = eval(output_)
                    else:
                        output_ = False
                except:
                    output_ = False
                result = output_
            
            if target == result:
                
                score += 1
        
        scores.append(score)
        
    max_score = max(scores)
    max_index = scores.index(max_score)
    
    return codes[max_index]

def select_hypothesis_lm(task_name, transformations, samples, args, return_eval_results=False):

    scores = []
    eval_results = []
    
    samples = [sample.replace("input: ", "").replace("output: ", "").split("\n") for sample in samples]
    
    for transformation in transformations:
        
        score = 0
        eval_result = []
        
        for data in samples:
            
            input_ = eval(data[0])
            target = eval(data[1])
            
            META_PROMPT = """
You are a smart assistant, now please help me predict the output given the input and the transformation.\n\
\n\
transformation: Set all elements in the input list to 9.:\n\
input: [0, 8, 9, 3, 7, 5, 5]\n\
output: [9, 9, 9, 9, 9, 9, 9]\n\
\n\
transformation: Delete all odd number from the input list.:\n\
input: [3, 4, 8, 1, 0, 5, 3, 7, 9, 9]\n\
output: [4, 8, 0]\n\
\n\
transformation: Reverse the input list.:\n\
input: [1, 3, 7, 4, 2, 0, 8, 9]\n\
output: [9, 8, 0, 2, 4, 7, 3, 1]\n\
\n\
transformation: Remove the first element of the input list.:\n\
input: [7, 0, 3, 6]\n\
output: [0, 3, 6]\n\
\n\
"""

            
            while True:
                try:
                    output_ = args.deductor(META_PROMPT + f"transformation: {transformation}\ninput: {str(input_)}\noutput:")
                    output_ = last_square_brackets(output_)
                    result = eval(output_)
                    break  # If no error occurs, break out of the loop
                except openai.error.OpenAIError:
                    continue  # If an OpenAI error occurs, retry the code inside the try block
                except Exception:
                    result = False  # If an error occurs during eval function, set result to False
                    break
                
            case_score = target == result
            
            score += case_score
            
            eval_result.append((f"input: {input_}", f"output: {target}", output_, case_score))
        
        scores.append(score)
        eval_results.append(eval_result)
        
    max_score = max(scores)
    max_index = scores.index(max_score)

    if return_eval_results:
        return transformations[max_index], eval_results[max_index]
    else:
        return transformations[max_index]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="./exp/insin/llama2-1000-5", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default='./data/instruction-induction/data/induction_input', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    