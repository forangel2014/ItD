import json
import os
import numpy as np
import openai
import csv
import json
import random
from kid.lm import LLM
from tqdm import tqdm

class Instance:
    
    def __init__(self, instruction, input_, output_):
        
        self.instruction = instruction
        self.input = input_
        self.output = output_

    def __str__(self):
        
        return f"instruction: {self.instruction}\ninput: {self.input}\noutput: {self.output}\n"

    def to_dict(self):
        
        return {"instruction": self.instruction, "input": self.input, "output": self.output}

    def check(self):
        
        return self.instruction and self.input and self.output

def load_arc():
    directory = "data/ARC/evaluation"
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                json_data = json.load(file)
                data.append(json_data)
    return data

def covert_sample_to_text(samples):
    samples_text = []
    i = 0
    for sample in samples:
        text = ""
        text += f"Case {i}:\n"
        i += 1
        for part in ["input", "output"]:
            text += part + ":\n"
            text += str(np.array(sample[part]))
            text += "\n"
        samples_text.append(text)
    return samples_text

def parse_instruction(text):
    instructions = text.split("\n")[:-1]
    instructions = [instruction.lstrip("0123456789. ") for instruction in instructions]
    instructions = [instruction for instruction in instructions if not any(keyword in instruction for keyword in ["above", "rotat", "below", "column"])]

    return instructions

def parse_samples(text):
    samples = []
    lines = text.strip().split("\n")
    lines = [line.lstrip("0123456789. ") for line in lines]
    i = 0
    while i < len(lines)-1:
        if lines[i].startswith("input: ") and lines[i+1].startswith("output: ") and lines[i+1].endswith("]]"):
            input_ = lines[i].lstrip("input: ")
            output_ = lines[i+1].lstrip("output: ")
            samples.append((input_, output_))
            i += 2
        else:
            i += 1        
    return samples

def generate_data_sequentially(filename, num_inst, num_samples_per_inst):

    META_PROMPT = """
Please generate diverse natural language decribed possible transformation for 1D grids.\n\
The input grid will be numpy array like [[0, 2, 1, 8, 3, 2, 5, 7, 6, 8, 4, 9, 7, 3]].\n\
The transformation should transform the input grid to a different output grid of the same size.\n\
You should treat black cells as backgrounds and continuously colored cell as objects. The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown.\n\
You can design various types of transformations.\n\
Here are several examples:\n\
"""
    #llm = LLM(model_name_or_path="../llama2-cn/llama-2-7b-chat", max_token=200, device=0)
    llm = LLM(model_name_or_path="chatgpt", max_token=200, device=0)
    
    instructions = [
                    "fill all odd cells with the color of the first cell.",
                    "splice all objects to the right in front of the last object.",
                    "turn the leftmost grid of all objects into the background.",
                    "split each object into two parts by setting the grid of different color in each object as the background.",
                    "fill all backgrounds outside the object with the color of their left ojects."
                    ]

    while len(instructions) < num_inst:

        try:
            few_shot_instructions = random.sample(instructions, 5)
            random.shuffle(few_shot_instructions)
            few_shot_prompt = "\n".join(few_shot_instructions) + "\nNow please generate more different transformations:\n"
            response = llm(META_PROMPT + few_shot_prompt)

            new_instructions = parse_instruction(response)
            instructions.extend(new_instructions)

        except (TimeoutError, openai.error.OpenAIError) as e:
            print(e)
            pass
        
        print(f"{len(instructions)}/{num_inst}")

    instructions = list(set(instructions))

    json.dump(instructions, open(f"./finetune/instructions-{filename}", 'w'))

    META_PROMPT = """
Please generate corresponding input-output grids that satisfy the given transformation.\n\
Here are several examples:\n\
transformation: fill all odd cells with the color of the first cell.\n\
input: [[3, 0, 4, 4, 4, 0, 0, 7, 7, 8, 8, 0, 0, 0]]\n\
output: [[3, 0, 3, 4, 3, 0, 3, 7, 3, 8, 3, 0, 3, 0]]\n\
input: [[7, 6, 6, 6, 9, 9, 9, 9, 5, 0, 0,]]\n\
output: [[7, 6, 7, 6, 7, 9, 7, 9, 7, 0, 7]]\n\
input: [[6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8]]\n\
output: [[6, 6, 6, 6, 6, 5, 6, 5, 6, 5, 6, 8, 6, 8, 6, 8, 6, 8]]\n\
transformation: splice all objects to the right in front of the last object.\n\
input: [[2, 0, 3, 3, 3, 3, 0, 9, 9, 9, 9, 0, 0, 0, 6, 6, 0]]\n\
output: [[0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 9, 9, 9, 9, 6, 6, 0]]\n\
input: [[0, 3, 3, 0, 9, 9, 0, 5, 5,]]\n\
output: [[0, 0, 0, 3, 3, 9, 9, 5, 5,]]\n\
input: [[1, 1, 0, 0, 5, 5, 5, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0]]\n\
output: [[0, 0, 0, 0, 0, 1, 1, 5, 5, 5, 4, 4, 4, 4, 0, 0, 0, 0]]\n\
Now please generate examples for the following instruction:\n\
"""

    instances = []

    for instruction in tqdm(instructions):
        
        samples = []
        
        #while len(samples) < num_samples_per_inst:

        try:

            few_shot_prompt = f"instruction: {instruction}\n" #"\n".join(random.sample(instructions, 3))
            response = llm(META_PROMPT + few_shot_prompt)

            new_samples = parse_samples(response)
            samples.extend(new_samples)
                    
        except (TimeoutError, openai.error.OpenAIError) as e:
            print(e)
            pass
        
        instances.extend([Instance(instruction, *sample) for sample in samples]) 
            
    instances_json = [instance.to_dict() for instance in instances]
    json.dump(instances_json, open(f"./finetune/{filename}", 'w'))
    

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
            