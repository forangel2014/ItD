import re
import os
import openai
import csv
import json
import random
from itd.lm import LLM
from tqdm import tqdm
from utils.base import first_part, list_files, list_subdirectories, replace_first_quote_content

class Instance:
    
    def __init__(self, instruction, input_, output_):
        
        self.instruction = instruction
        self.input = input_
        self.output = output_

    def __str__(self):
        
        return f"instruction: {self.instruction}\nInput: {self.input}\nOutput: {self.output}\n"
    
    def io_str(self):
        
        return f"Input: {self.input}\nOutput: {self.output}\n"

    def to_dict(self):
        
        return {"instruction": self.instruction, "input": self.input, "output": self.output}

    def check(self):
        
        return self.instruction and self.input and self.output

def load_x(args):
    
    raw_dir = f"{args.data_dir}/induction_input"
    task_files = list_files(raw_dir)

    xs = {}
    for file in task_files:
        x = []
        task = file.replace(".json", "")
        task_dir = f"{raw_dir}/{file}"
        induction_data = json.load(open(task_dir, 'r'))["examples"]
        for id_ in induction_data:
            x_raw = induction_data[id_]["metadata"]["examples_seen"]
            x_raw = x_raw.split("\n\n")
            x_raw = [x.split("\n")[0] for x in x_raw]
            x.extend(x_raw)
            
        x = list(set(x))
        xs[task] = x
            
    return xs    

def parse(text):
    keys = ["instruction", "input", "output"]
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

def parse_instruction(text):
    instructions = text.split("\n")[:-1]
    instructions = [instruction.lstrip("0123456789. ") for instruction in instructions]
    ban_patterns = ["given the", "given a", "music", "picture", "art", "image", r".*\bgiven\b.*:\s*.+", "painting", 
                    "a new", "story", "creat", "a given list of", "song", "a play", "poem", "essay", "literature",
                    "photo", "a list of", r".*\bgiven\b.*\bis\b\s*.+", "conduct", "limerick", "passage", "design",
                    "summar", r".*\bwrite\b.*\bletter\b", "crossword", "puzzle", "another", "different language",
                    "including", "of your choice", "random", "haiku", "explain", "recommendation", "solve"]
    instructions = [instruction for instruction in instructions if not any(re.search(pattern, instruction, re.IGNORECASE) for pattern in ban_patterns)]
    instructions = [instruction for instruction in instructions if all(keyword in instruction.lower() for keyword in ["given"])]

    return instructions

def parse_samples(text):
    samples = []
    lines = text.strip().split("\n")
    lines = [line.lstrip("0123456789. ") for line in lines]
    i = 0
    while i < len(lines)-1:
        if lines[i].startswith("Input: ") and lines[i+1].startswith("Output: "):
            input_ = lines[i].replace("Input: ", "").replace("</s>", "")
            output_ = lines[i+1].replace("Output: ", "").replace("</s>", "")
            samples.append((input_, output_))
            i += 2
        else:
            i += 1
    return samples

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
# Now create different types of instructions as much as possible. Do not create same instances with the examples.
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
Please generate natural language instructions and corresponding input-output pairs that follow the instrcutions.\n\
Please generate diverse instructions as much as possible.
"""
# Each valid output instance should be in the form of:
# instrcution: <xxx>\n\
# input: <xxx>\n\
# output: <xxx>\n\
# """
    #llm = LLM(model_name_or_path="../llama2-cn/llama-2-7b-chat", max_token=100, device=0)
    llm = LLM(model_name_or_path="chatgpt", max_token=200, device=0)
    
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

def load_induced_instructions(args):
    
    tasks = list_subdirectories(args.load_from_induced)
    
    base_instructions = {}
    for task in tasks:
        task_dir = f"{args.load_from_induced}/{task}"
        base_task_instructions = list(json.load(open(f"{task_dir}/prediction_groups.json"))["metadata"]["unique_predictions_counter"].keys())
        #base_instructions.extend(base_task_instructions)
                
        instructions = []
        prefixes = []
        for instruction in base_task_instructions:
            # prefix = " ".join(instruction.split(" ")[:15])
            # if prefix not in prefixes:
            #     prefixes.append(prefix)
            instruction = first_part(instruction)
            instructions.append(instruction)
        
        cleaned_instructions = []    
        for instruction in instructions:
            instruction = replace_first_quote_content(instruction)
            ban_patterns = []#["alphabet"]
            if not any(re.search(pattern, instruction, re.IGNORECASE) for pattern in ban_patterns):
                cleaned_instructions.append(instruction)
                
        cleaned_instructions = list(set(cleaned_instructions))
            
        base_instructions.update(dict([(instruction, task) for instruction in cleaned_instructions]))

    return base_instructions

def generate_data_sequentially(args):

    if args.base_model == "chatgpt" or args.load_instance:
        llm = LLM(model_name_or_path="chatgpt", max_token=100, device=0)
    else:
        llm = LLM(model_name_or_path=args.base_model, max_token=100, device=0)

    if args.load_instruction or args.load_instance:
        
        instructions = json.load(open(f"{args.exp_dir}/instructions.json", 'r'))
        
    elif os.path.isdir(args.load_from_induced): #args.exp_dir == "./exp/insin/base":
        
        instructions = load_induced_instructions(args)
        json.dump(instructions, open(f"{args.exp_dir}/instructions.json", 'w'))

    else:
    
        META_PROMPT = """
You are a smart assistant, your task is to help me write 10 instructions.\n\
The instructions must need a language input in order to output language results.\n\
The expected output of the instructions should be deterministic rather than creative.\n\
Your instructions should be as diverse as possible.\n\
Here are the instructions:\n\
"""
#Here are several examples:\n\
#The instructions must contain the word \"given\".\n\
        
        instructions = [
                        #"output the larger number between the given two numbers.",
                        "Describe the major color of the given object.",
                        "Translate the given sentence into Chinese.",
                        "Answer the capital of the given country.",
                        "Output the last letter of the given string.",
                        "Choose the healthier one from the given two different foods.",
                        "Find all the verbs in the given sentence.",
                        "Convert the given word into the comparative degree.",
                        "Rewrite the given sentence using the past tense.",
                        "Multiply the given two numbers.",
                        "Write down the family to which the given animal belongs.",
                        ]

        while len(instructions) < args.num_inst:

            try:
                few_shot_instructions = random.sample(instructions, 5)
                #few_shot_instructions = [f"instruction: {instruction}" for instruction in few_shot_instructions]
                random.shuffle(few_shot_instructions)
                few_shot_instructions = [f"{i+1}. {few_shot_instructions[i]}" for i in range(len(few_shot_instructions))]
                few_shot_prompt = "\n".join(few_shot_instructions) #+ "\nNow please generate more instructions:\n"
                response = llm(META_PROMPT + few_shot_prompt, do_sample=True)
                print(response)
                new_instructions = parse_instruction(response)
                print(new_instructions)
                instructions.extend(new_instructions)

            except (TimeoutError, openai.error.OpenAIError) as e:
                print(e)
                pass

            instructions = list(set(instructions))
            
            print(f"{len(instructions)}/{args.num_inst}")

        json.dump(instructions, open(f"{args.exp_dir}/instructions.json", 'w'))

    #llm.max_token = 200

    if args.load_instance:

        instances = json.load(open(f"{args.exp_dir}/instances.json", 'r'))

    else:

        META_PROMPT = """
You are a smart assistant, now please help me generate corresponding input-output pairs that satisfy the given instruction.\n\
Do not repeat the instructions in the inputs.\n\
instruction: describe the major color of the given object.\n\
Input: watermelon.\n\
Output: green.\n\
Input: panda.\n\
Output: black and white.\n\
Input: ocean.\n\
Output: blue.\n\
Input: blood.\n\
Output: red.\n\
Input: snow.\n\
Output: white.\n\
instruction: answer the capital of the given country.\n\
Input: USA\n\
Output: Washington.\n\
Input: China.\n\
Output: Beijing.\n\
Input: Russia.\n\
Output: Moscow.\n\
Input: France.\n\
Output: Paris.\n\
Input: UK.\n\
Output: London.\n\
"""
#Here are several examples:\n\
#Now please generate examples for the following instruction:\n\

        if args.load_x:
            xs = load_x(args)

        if args.use_deductor_during_induction:
            args.deductor = LLM(model_name_or_path="chatgpt", max_token=args.max_tokens, device=0)
        else:
            args.deductor = llm

        instances = []

        for instruction in tqdm(instructions):
            
            samples = []
            try_time = 0
            
            while len(samples) < args.num_samples_per_inst and try_time < 10:

                try:

                    few_shot_prompt = f"instruction: {instruction}\n" #"\n".join(random.sample(instructions, 3))
                    
                    if args.load_x:
                        sampled_x = random.choice(xs[instructions[instruction]])
                        sampled_prompt = f"{sampled_x}\n"
                        few_shot_prompt += sampled_prompt
                    
                    if args.use_deductor_during_induction:
                        response = args.deductor(META_PROMPT + few_shot_prompt, temperature=1)
                    else:
                        response = args.deductor(META_PROMPT + few_shot_prompt, do_sample=True)
                    
                    if args.load_x:
                        response = sampled_prompt + response
                    new_samples = parse_samples(response)[:5]
                    samples.extend(new_samples)
                    
                    try_time += 1

                except (TimeoutError, openai.error.OpenAIError) as e:
                    print(e)
                    pass
            
            instances.extend([Instance(instruction, *sample) for sample in samples])
            print(instruction)
            print(samples)
                
        instances_json = [instance.to_dict() for instance in instances]
        json.dump(instances_json, open(f"{args.exp_dir}/instances.json", 'w'))

def deduplicate_list_of_dicts(list_of_dicts):
    seen_instructions = set()
    deduplicated_list = []

    for d in list_of_dicts:
        instruction = d.get('instruction')
        if instruction is not None and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            deduplicated_list.append(d)

    return deduplicated_list

def prepare_data(args):
    all_instructions_to_tasks = json.load(open(f"{args.exp_dir}/instructions.json", 'r'))
    all_instructions = list(all_instructions_to_tasks.keys())
    
    if args.ood_tasks:
        #["sum", "translation_en-fr", "synonyms", "first_word_letter"]
        ood_tasks = args.ood_tasks.split("+")
        json.dump(ood_tasks, open(f"{args.exp_dir}/ood_tasks.json", 'w'))
        all_instructions = [instruction for instruction in all_instructions if all_instructions_to_tasks[instruction] not in ood_tasks]
    
    all_instances = json.load(open(f"{args.exp_dir}/instances.json", 'r'))
    all_instances = [Instance(instance['instruction'], instance['input'], instance['output']) for instance in all_instances]
    all_instances = [instance for instance in all_instances if instance.check()]
    all_instances = [instance for instance in all_instances if instance.instruction in all_instructions]

    num_valid = max(round((1-args.ratio)*len(all_instructions)), 1)
    train_instances = all_instances#[instance for instance in all_instances if instance.transformation in train_transformations]
    valid_instructions = random.sample(all_instructions, num_valid)
    valid_instances = [instance for instance in all_instances if instance.instruction in valid_instructions]
    
    print(f"train samples: {len(train_instances)}")
    print(f"valid samples: {len(valid_instances)}")
    
    train_file = f'{args.exp_dir}/train.csv'
    valid_file = f'{args.exp_dir}/valid.csv'

    if args.mode == "gd":

        to_csv(train_instances, train_file)
        to_csv(valid_instances, valid_file)

    else:
        
        assert args.mode == "io"
        to_csv_io(train_instances, train_file, all_instructions)
        to_csv_io(valid_instances, valid_file, all_instructions)

def to_csv(instances, filename):

    fieldnames = ['input', 'target']
    fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
    post_prompt = "The instruction was"

    instances = [{'input': fore_prompt + f"Input: {instance.input}\nOutput: {instance.output}\n" + post_prompt, 'target': instance.instruction} for instance in instances]

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in instances:
            writer.writerow(row)

def to_csv_io(instances, filename, instructions):

    fieldnames = ['input', 'target']
    fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
    post_prompt = "The instruction was"
    
    instance_batches = []
    for instruction in instructions:
        instance_batch = []
        for instance in instances:
            if instance.instruction == instruction:
                instance_batch.append(instance)
        if instance_batch:
            instance_batches.append(instance_batch)

    instances = [{'input': fore_prompt + "".join([instance.io_str() for instance in instance_batch]) + post_prompt, 'target': instance_batch[0].instruction} for instance_batch in instance_batches]

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in instances:
            writer.writerow(row)


def filter_data(args):
    all_instructions = json.load(open(f"{args.exp_dir}/instructions.json", 'r'))
    all_instances = json.load(open(f"{args.exp_dir}/instances.json", 'r'))
    all_instances = [Instance(instance['instruction'], instance['input'], instance['output']) for instance in all_instances]
    all_instances = [instance for instance in all_instances if instance.check()]
    
    if args.base_model == "chatgpt" or args.load_instance:
        llm = LLM(model_name_or_path="chatgpt", max_token=100, device=0)
    else:
        llm = LLM(model_name_or_path=args.base_model, max_token=100, device=0)

    reasoner = LLM(model_name_or_path="chatgpt", max_token=100, device=0)
    
    filted_data = []
    for instance in all_instances:
        prompt = f"Instruction: {instance.instruction}\nInput: {instance.input}\nOutput:"
        target = instance.output
        
        target_loss, target_loss_given_prompt = llm.lm_loss(prompt, target)
        
        delta = target_loss - target_loss_given_prompt
        
        print(prompt)
        print(target)
        print(delta)
        