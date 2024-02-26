import os
import json
import random
import random
import argparse
import json
from urllib import response
import openai
from tqdm import tqdm
from pathlib import Path
from itd.model import load_base_model_and_tokenizer, load_chatglm, load_finetuned_model_and_tokenizer
from itd.induction import LLMInductor
from itd.prompt import Prompter
from itd.lm import LLM
from utils.base import list_subdirectories, mkdir, sort_list_by_frequency
from utils.listfunc import parse_function, post_process, validate_function_syntax, load_induced_transformations, parse_hypo
from listfunc.run_eval import select_hypothesis, select_hypothesis_lm

def induce(args):

    tasks = list_subdirectories(args.data_dir)

    for task in tqdm(tasks):
        
        transformations = []
        
        if task == "results": #or os.path.exists(f"{args.exp_dir}/{args.out_dir}/{args.mode}/{task}/transformations.json"):
            print(f"skip task {task}")
            continue
        
        task_dir = f"{args.data_dir}/{task}"
        task_datafile = f"{task_dir}/train.json"

        data = json.load(open(task_datafile))
        
        for train_batch in tqdm(data):

            samples = [f"input: {train_data['input']}\noutput: {train_data['target']}" for train_data in train_batch]

            if args.base_model in ["chatgpt", "gpt3", "gpt4"]:
                success = False
                while not success:
                    try:
                        prompt = args.prompter.IO_prompt(samples)
                        if args.mode == "io-sample":
                            for i in range(5):
                                text = args.chatgpt(prompt[0], temperature=1, top_p=1)
                                transformations.append(text)
                            success = True
                        elif args.mode == 'hs' or args.mode == 'hs+d':
                            prompt = args.prompter.IO_prompt(samples)
                            texts = []
                            scores = []
                            for i in range(5):
                                text = args.chatgpt(prompt[0], temperature=1, top_p=1)
                                text = text.split("\n")[0]
                                texts.append(text)
                            transformation = select_hypothesis_lm(task, texts, samples, args)
                            success = True
                            transformations.append(transformation)
                        elif args.mode == "io":
                            text = args.chatgpt(prompt[0], temperature=0, top_p=0)
                            success = True
                            print(text)
                            transformations.append(text)
                    except (TimeoutError, openai.error.OpenAIError) as e:
                        print(e)
                        success = False
            else:             
                if args.mode == 'gd':
                    prompt = args.prompter.GD_prompt(samples)
                    response = args.inductor.induce(prompt, num_beams=5)
                    # samples.append(samples[0])
                    # merged_samples = []
                    # i = 0
                    # while i < len(samples):
                    #     merged_samples.append("\n".join([samples[i], samples[i+1]]))
                    #     i += 2
                    # prompt = args.prompter.GD_prompt(merged_samples)
                    # response = args.inductor.induce(prompt, num_beams=5)
                elif args.mode == 'gd-sample':
                    prompt = args.prompter.GD_prompt(samples)
                    response = args.inductor.gd_sample(prompt, num_beams=5)
                elif args.mode == 'io':
                    prompt = args.prompter.IO_prompt(samples)
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'io-2':
                    prompt = args.prompter.IO_prompt(samples[:2])
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'io-8':
                    samples_extra = random.choice(train_batch)
                    samples_extra = [f"input: {train_data['input']}\noutput: {train_data['target']}" for train_data in samples_extra]
                    prompt = args.prompter.IO_prompt(samples+samples_extra[:3])
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'io-20':
                    for _ in range(3):
                        samples_extra = random.choice(train_batch)
                        samples_extra = [f"input: {train_data['input']}\noutput: {train_data['target']}" for train_data in samples_extra]
                        samples += samples_extra
                    prompt = args.prompter.IO_prompt(samples)
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'gd-2':
                    merged_samples = []
                    i = 0
                    while i < len(samples):
                        merged_samples.append("\n".join([samples[i], samples[i+1]]))
                        i += 2
                    prompt = args.prompter.GD_prompt(merged_samples)
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'gd-8':
                    samples_extra = random.choice(train_batch)
                    samples_extra = [f"input: {train_data['input']}\noutput: {train_data['target']}" for train_data in samples_extra]
                    samples += samples_extra[:3]
                    merged_samples = []
                    i = 0
                    while i < len(samples):
                        merged_samples.append("\n".join([samples[i], samples[i+1]]))
                        i += 2
                    prompt = args.prompter.GD_prompt(merged_samples)
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'gd-20':
                    for _ in range(3):
                        samples_extra = random.choice(train_batch)
                        samples_extra = [f"input: {train_data['input']}\noutput: {train_data['target']}" for train_data in samples_extra]
                        samples += samples_extra
                    merged_samples = []
                    i = 0
                    while i < len(samples):
                        merged_samples.append("\n".join([samples[i], samples[i+1]]))
                        i += 2
                    prompt = args.prompter.GD_prompt(merged_samples)
                    response = args.inductor.beamsearch(prompt, num_beams=5)
                elif args.mode == 'io-sample':
                    prompt = args.prompter.IO_prompt(samples)
                    response = args.inductor.self_consistency(prompt, num_beams=1)
                elif args.mode == 'hs' or args.mode == 'hs+d':
                    prompt = args.prompter.IO_prompt(samples)
                    texts = []
                    scores = []
                    for i in range(5):
                        text, score = args.inductor.self_consistency(prompt, num_beams=1)
                        text = text.split("\n")[0]
                        texts.append(text)
                        scores.append(score)
                    response = [(select_hypothesis_lm(task, texts, samples, args), 0)]
                elif args.mode == 'one-shot':
                    prompt = args.prompter.GD_prompt(samples)
                    response = args.inductor.induce([random.choice(prompt)], num_beams=5)
                elif args.mode == 'sample':
                    prompts = args.prompter.GD_prompt(samples)
                    for prompt in prompts:
                        response = args.inductor.sample([prompt], num_beams=1)
                        text, logprobs = response[0]
                        transformations.append(text)
                    continue
                elif args.mode == 'sc':
                    prompt = args.prompter.IO_prompt(samples)
                    texts = []
                    scores = []
                    for i in range(5):
                        text, score = args.inductor.self_consistency(prompt, num_beams=1)
                        texts.append(text)
                        scores.append(score)
                    response = sort_list_by_frequency(texts, scores)
                elif args.mode == 'all' or args.mode == 'all+d':
                    prompt = args.prompter.IO_prompt(samples)
                    texts = []
                    scores = []
                    for i in range(5):
                        text, score = args.inductor.self_consistency(prompt, num_beams=1)
                        text = "The transformation " + text.split("\n")[0]
                        texts.append(text)
                        scores.append(score)
                    sc_response = sort_list_by_frequency(texts, scores)
                    sc_response = sc_response[0][0]
                    hypothesis, eval_results = select_hypothesis_lm(task, texts, samples, args, return_eval_results=True)
                    hs_response = hypothesis
                    score = sum([result[-1] for result in eval_results])
                    prompt = f"Rewrite the following transformation to make it better in predicting the output based on the input.\n\nPrevious transformation: {hypothesis}\n\n"
                    for i in range(5):
                        flag = "correct" if eval_results[i][3] else "incorrect"
                        prompt = prompt + f"{eval_results[i][0]}\n{eval_results[i][1]}\nThe correct output should be：{eval_results[i][2]}\nYour answer is {flag}\n\n"
                    prompt = prompt + "\nAfter observing the input-output results of these samples, a better tranformation is:\nThe transformation"
                    new_hypothesis = args.inductor.beamsearch(prompt, num_beams=1)[0][0]
                    new_hypothesis = "The transformation " + new_hypothesis.split("\n")[0]
                    new_hypothesis, new_eval_results = select_hypothesis_lm(task, [new_hypothesis], samples, args, return_eval_results=True)
                    new_score = sum([result[-1] for result in new_eval_results])
                    print(hypothesis, score)
                    print(new_hypothesis, new_score)
                    hsr_response = new_hypothesis if new_score > score else hypothesis
                    
                    transformations.append({"io-sample": texts, "sc": sc_response, "hs": hs_response, "hsr": hsr_response})
                    
                    continue
                    #break
                    
                    
                text, logprobs = response[0]
                text = text.split("\n")[0]
                
                if not text.startswith("The transformation"):
                    text = "The transformation " + text
                print(text)
                
                transformations.append(text)
                
        if args.mode == "all" or args.mode == "all+d":
            
            io_sample_transformations = []
            for t in transformations:
                io_sample_transformations.extend(t["io-sample"])
            sc_transformations = [t["sc"] for t in transformations]
            hs_transformations = [t["hs"] for t in transformations]
            hsr_transformations = [t["hsr"] for t in transformations]

            io_sample_transformations = list(set(io_sample_transformations))
            sc_transformations = list(set(sc_transformations))
            hs_transformations = list(set(hs_transformations))
            hsr_transformations = list(set(hsr_transformations))
            
            for mode in ["io_sample", "sc", "hs", "hsr"]:
                mode_dir = mode + "+d" if mode == "hs" and args.use_deductor_during_induction else mode
                exp_task_dir = f"{args.exp_dir}/{args.out_dir}/{mode_dir}/{task}"
                mkdir(exp_task_dir)
                transformations = eval(f"{mode}_transformations")
                json.dump(transformations, open(f"{exp_task_dir}/transformations.json", "w"))
            
        else:
            transformations = list(set(transformations))
            
            exp_task_dir = f"{args.exp_dir}/{args.out_dir}/{args.mode}/{task}"
            mkdir(exp_task_dir)
            json.dump(transformations, open(f"{exp_task_dir}/transformations.json", "w"))