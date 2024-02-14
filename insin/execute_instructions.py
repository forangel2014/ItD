import os
import sys
import argparse
import json
import openai
from tqdm import tqdm
from pathlib import Path
from utils.base import first_part
from insin.execution_eval import eval_single_prediction

INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def execution(task_name, args):
    with open(f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)

    #openai.organization = openai_organization
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    output_ = dict()

    parameters = {
        'max_tokens': args.max_tokens,
        'top_p': 0,
        'temperature': 1,
        'logprobs': 5,
    }
    for instruction_id, instruction_data in tqdm(data.items()):
        d = {}
        d['instruction'] = instruction_data['instruction']
        d['prediction_counter'] = instruction_data['prediction_counter']
        instruction_outputs = {}
        test_examples = instruction_data['test_inputs']
        
        id_ = 0
        test_examples_ls = list(test_examples.items())
        num_test = len(test_examples_ls)
        
        while id_ < num_test:
            
            _, example = test_examples_ls[id_]
            
            try:
            
                prompt = example['prompt']
                parameters['prompt'] = prompt

                #response = openai.Completion.create(**parameters)
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

                instruction_outputs[id_] = dict()
                instruction_outputs[id_]['prompt'] = prompt
                instruction_outputs[id_]['prediction'] = response.choices[0]["message"]["content"]

                print(f'generated {id_} predictions with OpenAI', file=sys.stderr)
                    
                id_ += 1
                
            except (TimeoutError, openai.error.OpenAIError) as e:
                print(e)
                continue
            
            #break

        d['instruction_outputs'] = instruction_outputs
        output_[instruction_id] = d
        
        #break

    output_path = f'{args.exp_dir}/{args.out_dir}/{args.mode}'
    Path(output_path).mkdir(exist_ok=True)

    with open(f'{output_path}/{task_name}_execution.json', 'w', encoding='utf-8') as f_predictions:
        json.dump(output_, f_predictions, indent=2, ensure_ascii=False)

def select_hypothesis(task_name, instructions, samples, args, return_eval_results=False):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    output_ = dict()

    parameters = {
        'max_tokens': args.max_tokens,
        'top_p': 0,
        'temperature': 1,
        'logprobs': 5,
    }
    
    cleaned_instructions = []
    for instruction in instructions:
        instruction = first_part(instruction)
        instruction = instruction.split(", which")[0]
        cleaned_instructions.append(instruction)
        
    cleaned_instructions = list(set(cleaned_instructions))
    scores = []
    eval_results = []
    
    for instruction in cleaned_instructions:
        
        id_ = 0
        num_test = len(samples)
        score = 0
        eval_result = []
        
        while id_ < num_test:
            
            sample = samples[id_]
            input_, output_ = sample.split("\n")
            output_ = output_.strip("Output: ")
            
            try:
            
                prompt = f"Instruction: {instruction}\n{input_}Output: "
                parameters['prompt'] = prompt

                prediction = args.deductor(prompt)
                if not args.use_deductor_during_induction:
                    prediction = prediction.split("\n")[0]

                case_score = eval_single_prediction(task_name, prediction, answers=[output_])
                score += case_score
                eval_result.append((input_, f"Output: {prediction}", output_, case_score))
                    
                id_ += 1
                
            except (TimeoutError, openai.error.OpenAIError) as e:
                print(e)
                continue
        
        scores.append(score)
        eval_results.append(eval_result)
        
    max_score = max(scores)
    max_index = scores.index(max_score)

    if return_eval_results:
        return cleaned_instructions[max_index], eval_results[max_index]
    else:
        return cleaned_instructions[max_index]
