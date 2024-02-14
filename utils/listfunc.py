import re
import os
import openai
import csv
import json
import random
import ast
import signal
from itd.lm import LLM
from tqdm import tqdm
from utils.base import list_subdirectories, mkdir

#     fore_prompt = """
# There is a transformation that transform the input list to the output list\n\
# please write a python function to describe the transformation.\n\
# """
#     post_prompt = "\nThe transformation is:\ndef transform(input_list):"
    
fore_prompt = """
There is a transformation that transform the input list to the output list\n\
please tell me the transformation in natural language.\n\
"""
#post_prompt = "\nThe transformation is:\n"
post_prompt = "\nThe transformation is:\nThe transformation"

def post_process(text):
    if not text.startswith("def transform(input_list):"):
        text = "def transform(input_list):" + text
    text = text.replace("\nreturn", "\n    return")
    if not text.endswith("\n"):
        text += "\n"
    if "return" not in text:
        text += "    return output_list\n"
    return text

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def execute_with_timeout(code, globals_dict, local_dict, timeout_seconds=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        exec(code, globals_dict, local_dict)
    except TimeoutError:
        print("Execution timed out")
    finally:
        signal.alarm(0)

def validate_function_syntax(function_str):
    try:
        ast.parse(function_str)
        return True
    except Exception:
        return False

def python_execution(transformation, input_):

    try:
        
        pattern = r"def\s+([\w_]+)\("

        match = re.search(pattern, transformation)
        if match:
            function_name = match.group(1)
        
        code = f"{transformation}\nresult={function_name}({input_})"

        local_vars = {}
        #exec(code, globals(), local_vars)
        execute_with_timeout(code, globals(), local_vars, timeout_seconds=1)

        result = local_vars.get('result')
        
        result = list(result)

    except Exception as e:
        result = False
        print(f"{e}")
        
    return result

def generate_random_list():
    length = random.randint(5, 10)
    random_list = [random.randint(0, 9) for _ in range(length)]
    return random_list

def last_square_brackets(string):
    if len(string) == 1 and string[0].isdigit():
        return f"[{string[0]}]"
    
    if len(string) >= 2:
        if string[-1].isdigit():
            string += "]"
        if not string[0] == '[':
            string = "[" + string
    
    for i in range(len(string)-1, -1, -1):
        if string[i] == "]":
            for j in range(i, -1, -1):
                if string[j] == "[":
                    return string[j:i+1]
    return string

def split_string(text):
    punctuation = [',', '.']
    for index, char in enumerate(text):
        if char in punctuation:
            text = text[:index]
            text = text.split(" if")[0].split(" that")[0].strip()
            return text
    return text

class Instance:
    
    def __init__(self, transformation, input_, output_):
        
        self.transformation = transformation
        self.input = input_
        self.output = output_

    def __str__(self):
        
        return f"transformation: {self.transformation}\ninput: {self.input}\noutput: {self.output}\n"

    def io_str(self):
        
        return f"input: {self.input}\noutput: {self.output}\n"

    def to_dict(self):
        
        return {"transformation": self.transformation, "input": self.input, "output": self.output}

    def check(self):
        
        return self.transformation and self.input and self.output

def parse(text):
    keys = ["transformation", "input", "output"]
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

def parse_transformation(text):
    transformations = text.split("\n")#[:-1]
    transformations = [transformation.lstrip("0123456789. ") for transformation in transformations]
    ban_patterns = ["random", "negat", "log", "sign", "-1", "sqrt", "square", "prime", "palindrome", "absolute", "module"]
    transformations = [transformation for transformation in transformations if not any(re.search(pattern, transformation, re.IGNORECASE) for pattern in ban_patterns)]
    transformations = [transformation for transformation in transformations if all(keyword in transformation.lower() for keyword in ["input list"])]

    return transformations

def parse_function(text):
    pattern = r"def\s.*?return\s+(.*?\n)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        function = match.group(0)
        return function
    else:
        return False

def parse_samples(text):
    samples = []
    lines = text.strip().split("\n")
    lines = [line.lstrip("0123456789. ") for line in lines]
    i = 0
    while i < len(lines)-1:
        if lines[i].startswith("input: ") and lines[i+1].startswith("output: "):
            input_ = lines[i].replace("input: ", "").replace("</s>", "")
            output_ = lines[i+1].replace("output: ", "").replace("</s>", "")
            input_ = last_square_brackets(input_)
            output_ = last_square_brackets(output_)
            samples.append((input_, output_))
            i += 2
        else:
            i += 1
    return samples

def parse_hypo(text):
    transformations = text.split("\n")
    transformations = [transformation.lstrip("0123456789. ") for transformation in transformations]
    ban_patterns = ["random"]
    transformations = [transformation for transformation in transformations if not any(re.search(pattern, transformation, re.IGNORECASE) for pattern in ban_patterns)]
    transformations = [transformation for transformation in transformations if all(keyword in transformation.lower() for keyword in ["the transformation"])]
    return transformations

def generate_few_shot_prompt(instances):
    instances_selected = random.sample(instances, 3)
#     few_shot_prompt = f"""
# Here are some examples:\n
# \n\
# {str(instances_selected[0])}
# \n\
# {str(instances_selected[1])}
# \n\
# {str(instances_selected[2])}
# \n\
# Now create different types of transformations as much as possible. Do not create same instances with the examples.
# """
    few_shot_prompt = f"""
\n\
{str(instances_selected[0])}
\n\
{str(instances_selected[1])}
\n\
{str(instances_selected[2])}
\n\
"""
    return few_shot_prompt

def generate_data_jointly(num):

    META_PROMPT = """
Please generate natural language transformations and corresponding input-output pairs that follow the instrcutions.\n\
Please generate diverse transformations as much as possible.
"""
# Each valid output instance should be in the form of:
# instrcution: <xxx>\n\
# input: <xxx>\n\
# output: <xxx>\n\
# """
    #llm = LLM(model_name_or_path="../llama2-cn/llama-2-7b-chat", max_token=100, device=0)
    #llm = LLM(model_name_or_path="chatgpt", max_token=200, device=0)
    
    instances = [
                Instance("output the larger number.",
                        "123 456",
                        "456"),
                Instance("output the color of the given object.",
                        "watermelon",
                        "green"),
                Instance("output the capital of the given country.",
                        "China",
                        "Beijing"),
                ]

    while len(instances) < num:

        try:

            few_shot_prompt = generate_few_shot_prompt(instances)
            response = llm(META_PROMPT + few_shot_prompt)

            new_instances = parse(response)
            instances.extend(new_instances)
        
        except (TimeoutError, openai.error.OpenAIError) as e:
            pass
        
        print(f"{len(instances)}/{num}")
        
    instances_json = [instance.to_dict() for instance in instances]
    json.dump(instances_json, open("./finetune/all_data.json", 'w'))

def load_induced_functions(args):
    
    tasks = list_subdirectories(args.load_from_induced)
    
    base_transformations = []
    for task in tasks:
        task_dir = f"{args.load_from_induced}/{task}"
        base_task_transformations = list(json.load(open(f"{task_dir}/transformations.json")))
        base_transformations.extend(base_task_transformations)
    
    base_transformations = list(set(base_transformations))
    
    valid_transformations = []
    for transformation in base_transformations:
        if transformation:
            transformation = post_process(transformation)
            transformation = parse_function(transformation)
        if validate_function_syntax(transformation):
            valid_transformations.append(transformation)
    
    return valid_transformations

def load_induced_transformations(args):
    
    tasks = list_subdirectories(args.load_from_induced)
    
    base_transformations = {}#[]
    for task in tasks:
        task_dir = f"{args.load_from_induced}/{task}"
        base_task_transformations = list(json.load(open(f"{task_dir}/transformations.json")))
        #base_task_transformations = [split_string(transformation) for transformation in base_task_transformations]
        
        #base_transformations.extend(base_task_transformations)
        base_task_transformations = list(set(base_task_transformations))
        
        for transformation in base_task_transformations:
            base_transformations[transformation] = task
    
    return base_transformations

def load_reference(args):
    
    tasks = list_subdirectories(args.data_dir)
    f = open(f"{args.exp_dir}/references.txt", "w")
    references = []
    for task in tasks:
        
        if task == "results":
            continue
        
        task_dir = f"{args.data_dir}/{task}"
        task_reference = json.load(open(f"{task_dir}/task.json"))["description"]
        
        task_score = json.load(open(f"exp/listfunc/mixtral/induction_out/io/{task}/execution.json"))
        
        task_reference = task_reference.split("The target function is \"")[-1].strip("\".")
        mkdir(f"{args.exp_dir}/{args.out_dir}/{args.mode}/{task}")
        json.dump([task_reference], open(f"{args.exp_dir}/{args.out_dir}/{args.mode}/{task}/transformations.json", "w"))

        references.append(task_reference)

        f.writelines(f"{task}: {task_reference}, {task_score}\n")
    
    return references

def load_x(args):
    
    tasks = list_subdirectories(args.data_dir)

    xs = {}

    for task in tasks:
        
        if task == "results":
            continue
        
        x_task = []
        
        task_dir = f"{args.data_dir}/{task}"
        data = json.load(open(f"{task_dir}/train.json"))        
        for train_batch in data:

            for sample in train_batch:

                x_task.append(sample["input"])
    
        xs[task] = x_task

    return xs

def generate_data_sequentially(args):

    if args.base_model == "chatgpt":
        llm = LLM(model_name_or_path="chatgpt", max_token=100, device=0)
    else:
        llm = LLM(model_name_or_path=args.base_model, max_token=100, device=0)

    if args.load_function or args.load_transformation or args.load_from_induced:
        pass

    elif args.load_transformation:
        
        transformations = json.load(open(f"{args.exp_dir}/transformations.json", 'r'))
        
    else:
    
        META_PROMPT = """
You are a smart assistant, your task is to help me write 10 transformations for the integer list.\n\
The transformation must take an input integer list as input and output another integer list.\n\
The transformations must be deterministic rather than stochastic.\n\
Here are the transformations:\n\
"""

        transformations = [
                        "Set all elements in the input list to 7.",
                        "Reverse the input list.",
                        "Subtract all elements from the input list by 1.",
                        "The i-th output is the sum of the input list which index are less or equal to i.",
                        "Remove the first element of the input list.",
                        "Delete all odd number from the input list.",
                        "Sort the input list.",
                        "Remove the 3rd element.",
                        "Append the sum of the inputs list to the end of the input list.",
                        "Remove all the numbers that are less than 10 in the input list.",
                        "Retain the latter half of the input list as output.",
                        "Always output the list [3, 6] no matter what the input list is.",
                        "Prepend the [5, 8] to the input list.",
                        "Append [3] to the input list.",
                        "Swap the 2rd and 5th elements in the input list.",
                        ]

        while len(transformations) < args.num_inst:

            try:
                few_shot_transformations = random.sample(transformations, 5)
                #few_shot_transformations = [f"transformation: {transformation}" for transformation in few_shot_transformations]
                random.shuffle(few_shot_transformations)
                few_shot_transformations = [f"{i+1}. {few_shot_transformations[i]}" for i in range(len(few_shot_transformations))]
                few_shot_prompt = "\n".join(few_shot_transformations) #+ "\nNow please generate more transformations:\n"
                response = llm(META_PROMPT + few_shot_prompt, do_sample=True)
                print(response)
                new_transformations = parse_transformation(response)
                print(new_transformations)
                transformations.extend(new_transformations)

            except (TimeoutError, openai.error.OpenAIError) as e:
                print(e)
                pass

            transformations = list(set(transformations))
            
            print(f"{len(transformations)}/{args.num_inst}")

        json.dump(transformations, open(f"{args.exp_dir}/transformations.json", 'w'))

    if args.load_from_induced: #args.exp_dir == "./exp/listfunc/base":
        
        functions = load_induced_functions(args)
        json.dump(functions, open(f"{args.exp_dir}/functions.json", 'w'))

    elif args.load_function:
        
        functions = json.load(open(f"{args.exp_dir}/functions.json", 'r'))
        
    else:

        META_PROMPT = """
You are a smart assistant, now please help me translate the given instruction to a python function.\n\
transformation: Set all elements in the input list to 9.\n\
def transform(input_list):\n\
    output_list = [9]*len(input_list)\n\
    return output_list\n\
transformation: Reverse the input list.\n\
def transform(input_list):\n\
    output_list = []\n\
    for i in range(len(input_list)):\n\
        output_list.append(input_list[-(i+1)])\n\
    return output_list\n\
"""

        functions = []

        for transformation in tqdm(transformations):

            few_shot_prompt = f"transformation: {transformation}\ndef transform(input_list):\n" #"\n".join(random.sample(transformations, 3))

            try_time = 0
            new_function = False
            
            while not new_function and try_time < 3:

                response = llm(META_PROMPT + few_shot_prompt, do_sample=True)

                response = post_process(response)
                
                new_function = parse_function(response)
                
                if not new_function:
                    try_time += 1
                    continue
                    
                elif validate_function_syntax(new_function):
                    print(new_function)
                    functions.append(new_function)
                
                else:
                    new_function = False
                    try_time += 1
                    continue
            

        json.dump(functions, open(f"{args.exp_dir}/functions.json", 'w'))

    llm.max_token = 20

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

    functions = list(set(functions))
    instances = []

    for function in tqdm(functions):
        
        samples = []
        try_time = 0
        
        while len(samples) < args.num_samples_per_inst and try_time < 3:

            input_ = generate_random_list()
            
            if args.use_deductor_during_induction:
            
                output_ = python_execution(function, input_)
            
            else:
                
                try:
                    output_ = llm(META_PROMPT + f"python function: {function}\ninput: {str(input_)}\noutput:")
                    output_ = eval(output_)
                except:
                    output_ = False
            
            if output_:
                samples.append((str(input_), str(output_)))
            
            else:
                try_time += 1

        
        instances.extend([Instance(function, *sample) for sample in samples]) 
            
    instances_json = [instance.to_dict() for instance in instances]
    json.dump(instances_json, open(f"{args.exp_dir}/instances.json", 'w'))

def generate_data_sequentially_lm(args):

    if args.base_model == "chatgpt" or args.use_deductor_during_induction:
        llm = LLM(model_name_or_path="chatgpt", max_token=100, device=0)
    else:
        llm = LLM(model_name_or_path=args.base_model, max_token=100, device=0)

    if args.load_from_induced:
        
        transformations = load_induced_transformations(args)
        json.dump(transformations, open(f"{args.exp_dir}/transformations.json", 'w'))

    elif args.load_transformation:
        
        transformations = json.load(open(f"{args.exp_dir}/transformations.json"))

    elif args.load_instance:
        pass

    else:
        
        META_PROMPT = """
You are a smart assistant, your task is to help me write 20 transformations for the integer list.\n\
The transformation must take an input integer list as input and output another integer list.\n\
The transformations must be deterministic rather than stochastic.\n\
Here are the transformations:\n\
"""

        transformations = [
                        "Set all elements in the input list to 7.",
                        "Reverse the input list.",
                        "Subtract all elements from the input list by 1.",
                        "The i-th output is the sum of the input list which index are less or equal to i.",
                        "Remove the first element of the input list.",
                        "Delete all odd number from the input list.",
                        "Sort the input list.",
                        "Remove the 3rd element of the input list.",
                        "Append the sum of the inputs list to the end of the input list.",
                        "Remove all the numbers that are less than 5 in the input list.",
                        "Retain the latter half of the input list as output.",
                        "Always output the list [5, 3, 6, 7] no matter what the input list is.",
                        "Always output the list [1, 8] no matter what the input list is.",
                        "Prepend the [5, 8] to the input list.",
                        "Append [3] to the input list.",
                        "Swap the 2rd and 5th elements in the input list.",
                        "Remove the last element of the input list.",
                        "Multiply each element in the input list with 3.",
                        "Insert [2, 7] after the second element of the input list.",
                        "Count the 0s in the input list.",
                        "Count from 0 to the length of the input list.",
                        ]

        while len(transformations) < args.num_inst:

            try:
                few_shot_transformations = random.sample(transformations, 10)
                #few_shot_transformations = [f"transformation: {transformation}" for transformation in few_shot_transformations]
                random.shuffle(few_shot_transformations)
                few_shot_transformations = [f"{i+1}. {few_shot_transformations[i]}" for i in range(len(few_shot_transformations))]
                few_shot_prompt = "\n".join(few_shot_transformations) #+ "\nNow please generate more transformations:\n"
                response = llm(META_PROMPT + few_shot_prompt + "\n", do_sample=True, stop="\n")
                print(response)
                new_transformations = parse_transformation(response)
                print(new_transformations)
                transformations.extend(new_transformations)

            except (TimeoutError, openai.error.OpenAIError) as e:
                print(e)
                pass

            transformations = list(set(transformations))
            
            print(f"{len(transformations)}/{args.num_inst}")

        json.dump(transformations, open(f"{args.exp_dir}/transformations.json", 'w'))

    if args.load_instance:
        
        instances = json.load(open(f"{args.exp_dir}/instances.json"))
        transformations = [instance["transformation"] for instance in instances]
        transformations = list(set(transformations))
        json.dump(transformations, open(f"{args.exp_dir}/transformations.json", 'w'))
    
    else:
        
        llm.max_token = 100

    
        META_PROMPT = """
You are a smart assistant, now please help me predict the output given the input and the transformation.\n\
\n\
transformation: Remove the first and the second element.\n\
input: [0, 8, 9, 3, 7, 5, 5]\n\
output: [9, 3, 7, 5, 5]\n\
input: [7, 3, 9, 6]\n\
output: [9, 6]\n\
input: [0, 0, 0, 7, 7, 7]\n\
output: [0, 7, 7, 7]\n\
input: [2, 5, 5, 6, 3]\n\
output: [5, 6, 3]\n\
input: [7, 3, 6, 8, 8, 5, 0]\n\
output: [6, 8, 8, 5, 0]\n\
\n\
transformation: Retain the elements that greater than 5.\n\
input: [3, 4, 8, 1, 0, 5, 3, 7, 9, 9]\n\
output: [8, 7, 9, 9]\n\
input: [0, 4, 5, 7, 7, 1, 2, 6]\n\
output: [7, 7]\n\
input: [1, 0, 0, 3, 7, 8, 5]\n\
output: [7, 8]\n\
input: [5, 1, 9, 3, 6, 1, 7, 3]\n\
output: [9, 6, 7]\n\
input: [2, 6, 8, 1, 7]\n\
output: [6, 8, 7]\n\
\n\
transformation: Reverse the input list.\n\
input: [1, 0, 3, 8]\n\
output: [8, 3, 0, 1]\n\
input: [1, 3, 7, 4, 2, 0, 8, 9]\n\
output: [9, 8, 0, 2, 4, 7, 3, 1]\n\
input: [8, 9, 0, 1, 3]\n\
output: [3, 1, 0, 9, 8]\n\
input: [5, 5, 6, 8, 0, 1, 3, 2]\n\
output: [2, 3, 1, 0, 8, 6, 5, 5]\n\
input: [2, 0, 8, 7, 5, 4]\n\
output: [4, 5, 7, 8, 0, 2]\n\
\n\
transformation: Append 5 to the input list.\n\
input: [7, 0, 3, 6]\n\
output: [7, 0, 3, 6, 5]\n\
input: [1, 2, 3, 7, 8, 5]\n\
output: [1, 2, 3, 7, 8, 5, 5]\n\
input: [2, 9, 6, 3, 7, 5, 4, 4]\n\
output: [2, 9, 6, 3, 7, 5, 4, 4, 5]\n\
input: [0, 0, 8, 6, 9]\n\
output: [0, 0, 8, 6, 9, 5]\n\
input: [7, 5, 6, 5, 3, 3, 2]\n\
output: [7, 5, 6, 5, 3, 3, 2, 5]\n\
\n\
"""

        instances = []

        if args.load_x:
            xs = load_x(args)

        for transformation in tqdm(transformations):
            
            samples = []
            try_time = 0
            
            while len(samples) < args.num_samples_per_inst and try_time < 3:

                sample = []

                if args.load_x:
                    task = transformations[transformation]
                    input_ = random.choice(xs[task])
                    
                else:

                    input_ = generate_random_list()
                    input_ = str(input_)
                    
                try:
                    prompt = META_PROMPT + f"transformation: {transformation}\ninput: {input_}\noutput: "
                    #prompt = META_PROMPT + f"transformation: {transformation}\n"
                    output_ = llm(prompt, stop="\n")
                    print(output_)
                    #sample = parse_samples(output_)
                    output_ = last_square_brackets(output_)
                    print("*"*10)
                    print(transformation)
                    #print(sample)
                    print(input_)
                    print(output_)
                    print("*"*10)
                
                except Exception as e:
                    print(e)
                    output_ = False
                
                if output_:
                    samples.append((input_, output_))
                
                else:
                    try_time += 1

            if try_time < 3:
                print(try_time)
                print(len(samples))
                instances.extend([Instance(transformation, *sample) for sample in samples]) 
                
        instances_json = [instance.to_dict() for instance in instances]
        json.dump(instances_json, open(f"{args.exp_dir}/instances.json", 'w'))


def deduplicate_list_of_dicts(list_of_dicts):
    seen_transformations = set()
    deduplicated_list = []

    for d in list_of_dicts:
        transformation = d.get('transformation')
        if transformation is not None and transformation not in seen_transformations:
            seen_transformations.add(transformation)
            deduplicated_list.append(d)

    return deduplicated_list

def prepare_data(args):
    all_transformations = json.load(open(f"{args.exp_dir}/transformations.json", 'r'))
    all_instances = json.load(open(f"{args.exp_dir}/instances.json", 'r'))
    all_instances = [Instance(instance['transformation'], instance['input'], instance['output']) for instance in all_instances]
    all_instances = [instance for instance in all_instances if instance.check()]
    #all_instances = deduplicate_list_of_dicts(all_instances)

    # num_transformations = len(all_transformations)
    # num_train = round(args.ratio*num_transformations)
    # train_transformations = random.sample(all_transformations, num_train)
    num_valid = max(round((1-args.ratio)*len(all_instances)), 1)
    train_instances = all_instances#[instance for instance in all_instances if instance.transformation in train_transformations]
    valid_instances = random.sample(all_instances, num_valid)#[instance for instance in all_instances if ((instance.transformation not in train_transformations) and (instance.transformation in all_transformations))]
    
    print(f"functions: {len(all_transformations)}")
    print(f"train samples: {len(train_instances)}")
    print(f"valid samples: {len(valid_instances)}")
    
    train_file = f'{args.exp_dir}/train.csv'
    valid_file = f'{args.exp_dir}/valid.csv'

    if args.mode == "gd":

        to_csv(train_instances, train_file)
        to_csv(valid_instances, valid_file)

    elif args.mode == "io":
        
        to_csv_io(train_instances, train_file, all_transformations)
        to_csv_io(valid_instances, valid_file, all_transformations)

    elif args.mode == "gd-2":
        
        to_csv_io(train_instances, train_file, all_transformations, num=2)
        to_csv_io(valid_instances, valid_file, all_transformations, num=2)

    elif args.mode == "gd-3":
        
        to_csv_io(train_instances, train_file, all_transformations, num=3)
        to_csv_io(valid_instances, valid_file, all_transformations, num=3)

def to_csv(instances, filename):
    
    fore_prompt = """
There is a transformation that transform the input list to the output list\n\
please tell me the transformation in natural language.\n\
"""
    post_prompt = "\nThe transformation is:\n"

    fieldnames = ['input', 'target']

    instances = [{'input': fore_prompt + f"input: {instance.input}\noutput: {instance.output}" + post_prompt, 'target': instance.transformation} for instance in instances]

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in instances:
            writer.writerow(row)

def to_csv_io(instances, filename, instructions, num=1):

    fore_prompt = """
There is a transformation that transform the input list to the output list\n\
please tell me the transformation in natural language.\n\
"""
    post_prompt = "\nThe transformation is:\n"

    fieldnames = ['input', 'target']

    instance_batches = []
    for instruction in tqdm(instructions):
        instance_batch = []
        for instance in instances:
            if instance.transformation == instruction:
                instance_batch.append(instance)
        if instance_batch:
            instance_batches.append(instance_batch)

    if num == 1:
        instances = [{'input': fore_prompt + "".join([instance.io_str() for instance in instance_batch]) + post_prompt, 'target': instance_batch[0].transformation} for instance_batch in instance_batches]

    else:
        instances = []
        for instance_batch in instance_batches:
            for i in range(5):
                try:
                    instances_tmp = random.sample(instance_batch, num)
                    instances.append({'input': fore_prompt + "".join([instance.io_str() for instance in instances_tmp]) + post_prompt, 'target': instance_batch[0].transformation})
                except:
                    pass
                
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in instances:
            writer.writerow(row)