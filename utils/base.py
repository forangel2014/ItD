import os
import re
import openai
import csv
import json
import random
from collections import Counter

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def list_subdirectories(directory):
    subdirectories = []
    try:
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                subdirectories.append(entry)
    except:
        pass
    return subdirectories

def list_files(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

def find_all_indices(string, target):
    indices = []
    index = 0

    while index < len(string):
        if string[index] == target:
            indices.append(index)
        index += 1

    return indices

def first_part(text):
    dot_index = find_all_indices(text, ".")
    for index in dot_index:
        last_word = text[:index].split(" ")[-1]
        if len(last_word) >= 3 or last_word.isdigit():
            return text[:index] + "."
    return text + "."

def replace_first_quote_content(string):
    pattern = r"['\"](.*?)['\"]" 
    replace_string = re.sub(pattern, r'the input', string, count=1)
    return replace_string


def split_list(lst, num_parts):
    avg = len(lst) // num_parts  # 每个子列表的平均长度
    remainder = len(lst) % num_parts  # 余数

    result = []
    start = 0

    for i in range(num_parts):
        length = avg + 1 if i < remainder else avg
        result.append(lst[start:start+length])
        start += length

    return result

class Instance:
    
    def __init__(self, f, input_, output_, keys):
        
        self.keys = keys
        self.f = f
        self.input = input_
        self.output = output_

    def __str__(self):
        
        return f"{self.keys[0]}: {self.f}\n{self.keys[1]}: {self.input}\n{self.keys[2]}: {self.output}\n"

    def to_dict(self):
        
        return {self.keys[0]: self.instruction, self.keys[1]: self.input, self.keys[2]: self.output}

    def check(self):
        
        return self.f and self.input and self.output

def parse(text, keys):
    instances = []
    loop = True
    while loop:
        instance_value = []
        for key in keys:
            key += ": "
            if key in text and "\n" in text:
                _, text = text.split(key, 1)
                index = text.find("\n")
                if index != -1:
                    value = text[:index]
                    text = text[index+1:]
                else:
                    value = text
                    text = ""
                instance_value.append(value)
            else:
                loop = False
                break
        if loop:
            instance = Instance(*instance_value)
            instances.append(instance)
    return instances

def deduplicate_list_of_dicts(list_of_dicts):
    seen_instructions = set()
    deduplicated_list = []

    for d in list_of_dicts:
        instruction = d.get('instruction')
        if instruction is not None and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            deduplicated_list.append(d)

    return deduplicated_list

def prepare_data(filename, ratio):
    all_instances = json.load(open(f"./finetune/{filename}", 'r'))
    all_instances = [Instance(instance['instruction'], instance['input'], instance['output']) for instance in all_instances]
    all_instances = [instance for instance in all_instances if instance.check()]
    #all_instances = deduplicate_list_of_dicts(all_instances)

    # 设置CSV文件的列名
    fieldnames = ['input', 'target']
    fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
    post_prompt = "The instruction was"

    instances = [{'input': fore_prompt + f"input: {instance.input}\noutput: {instance.output}\n" + post_prompt, 'target': instance.instruction} for instance in all_instances]
    #instances = [{'input': fore_prompt + f"input: {instance['input']}\noutput: {instance['output']}\n" + post_prompt, 'target': instance['instruction']} for instance in all_instances]

    random.shuffle(instances)
    num_instances = len(instances)
    num_train = round(ratio*num_instances)
    print(f"train samples: {num_train}")
    print(f"valid samples: {num_instances-num_train}")
    train_instances = instances[:num_train]
    valid_instances = instances[num_train:]

    # 指定CSV文件的路径
    train_file = './finetune/train.csv'
    valid_file = './finetune/valid.csv'

    # 使用'w'模式打开CSV文件
    with open(train_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入每一行数据
        for row in train_instances:
            writer.writerow(row)

    with open(valid_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入每一行数据
        for row in valid_instances:
            writer.writerow(row)

def sort_list_by_frequency(lst, log_probs):
    counter = Counter(lst)
    unique_list = list(set(lst))
    sorted_list = sorted(unique_list, key=lambda x: counter[x], reverse=True)
    sorted_log_probs = [log_probs[lst.index(elem)] for elem in sorted_list]
    final_list = [(sorted_list[i], sorted_log_probs[i]) for i in range(len(sorted_list))]
    return final_list

def generate_markdown_table(performance, order, tasks="all"):

    methods = order#list(list(performance.values())[0].keys())
    markdown_table = "| Task | " + " |".join(methods) + "\n"
    markdown_table += "| ---- | " + "-------- | "*len(methods) + "\n"

    for task in performance.keys():
        if task != "average" and (task in tasks or tasks == "all"):
            max_score_method = methods[0]
            max_score = performance[task][max_score_method]
            for method in methods:
                if performance[task][method] > max_score:
                    max_score = performance[task][method]
                    max_score_method = method
            row = f"| {task} "
            # for method, score in performance[task].items():
            #     row += f"| {score:.2f} "
            for method in methods:
                score = performance[task][method]*100
                if method == max_score_method:
                    row += f"| **{score:.2f}** "
                else:
                    row += f"| {score:.2f} "
            row += "|\n"
            markdown_table += row
    
    for task in performance.keys():
        if task == "average":
            max_score_method = methods[0]
            max_score = performance[task][max_score_method]
            for method in methods:
                if performance[task][method] > max_score:
                    max_score = performance[task][method]
                    max_score_method = method
            row = f"| {task} "
            # for method, score in performance[task].items():
            #     row += f"| {score:.2f} "
            for method in methods:
                score = performance[task][method]*100
                if method == max_score_method:
                    row += f"| **{score:.5f}** "
                else:
                    row += f"| {score:.5f} "
            row += "|\n"
            markdown_table += row
    
    return markdown_table

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):

        self.token_id_list = token_id_list
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list
    
def find_combinations(lst):
    n = len(lst)
    combinations = []
    
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                combination = "\n".join([lst[i], lst[j], lst[k]])
                combinations.append(combination)
    
    return combinations