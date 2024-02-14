import random
import argparse
import json
from urllib import response
import openai
from tqdm import tqdm
from pathlib import Path
from utils.base import sort_list_by_frequency
from itd.model import load_base_model_and_tokenizer, load_chatglm, load_finetuned_model_and_tokenizer
from itd.induction import LLMInductor
from itd.prompt import Prompter
from itd.lm import LLM
from insin.execute_instructions import select_hypothesis

def generate_instructions_chatgpt(task_name, args):
    Path(f'{args.exp_dir}/{args.out_dir}/{args.mode}').mkdir(exist_ok=True)
    with open(f'{args.data_dir}/{args.train_dir}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)
    examples = data['examples']

    output = dict()

    for id_, example in tqdm(examples.items()):
        prompt = example['input']

        success = False    
        while not success:
            try:
                temperature = top_p = 1 if args.mode == "io-sample" else 0
                response = args.chatgpt(prompt, temperature=temperature, top_p=top_p)
                success = True

            except (TimeoutError, openai.error.OpenAIError) as e:
                pass

        text, logprobs = response, 0

        output[id_] = dict()
        output[id_]['input'] = prompt
        output[id_]['prediction'] = text

        metadata = dict()
        metadata['logprobs'] = logprobs#response.choices[0]['logprobs']
        # metadata['finish_reason'] = response.choices[0]['finish_reason']
        output[id_]['metadata'] = metadata

        if int(id_) % 100 == 0:
            print(f'generated {id_} predictions with engine')

    output_path = f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}'
    Path(output_path).mkdir(exist_ok=True)

    output_with_metadata_path = f'{output_path}/predictions_with_metadata.json'
    with open(output_with_metadata_path, 'w', encoding='utf-8') as f_predictions_with_metadata:
        json.dump(output, f_predictions_with_metadata, indent=2, ensure_ascii=False)

    for id_ in output:
        del output[id_]['metadata']

    output_no_metadata_path = f'{output_path}/predictions.json'
    with open(output_no_metadata_path, 'w', encoding='utf-8') as f_predictions:
        json.dump(output, f_predictions, indent=2, ensure_ascii=False)
        
def generate_instructions(task_name, args):
    Path(f'{args.exp_dir}/{args.out_dir}/{args.mode}').mkdir(exist_ok=True)
    with open(f'{args.data_dir}/{args.train_dir}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)
    examples = data['examples']

    output = dict()

    for id_, example in tqdm(examples.items()):

        if args.mode == 'gd':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.induce(prompt, num_beams=5)
        elif args.mode == 'gd-3':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")[:3]
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.induce(prompt, num_beams=5)
        elif args.mode == 'gd-7':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            _, samples_extra = random.choice(list(examples.items()))
            samples_extra = samples_extra['metadata']['examples_seen']
            samples_extra = samples_extra.split("\n\n")[:2]
            samples = samples + samples_extra
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.induce(prompt, num_beams=5)
        elif args.mode == 'gd-20':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            for _ in range(3):
                _, samples_extra = random.choice(list(examples.items()))
                samples_extra = samples_extra['metadata']['examples_seen']
                samples_extra = samples_extra.split("\n\n")
                samples = samples + samples_extra
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.induce(prompt, num_beams=5)
        elif args.mode == 'gd-sample':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.gd_sample(prompt, num_beams=5)
        elif args.mode == 'io':
            prompt = example['input']
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'io-2':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")[:2]
            prompt = args.prompter.IO_prompt(samples)
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'io-3':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")[:3]
            prompt = args.prompter.IO_prompt(samples)
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'io-5':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            prompt = args.prompter.IO_prompt(samples)
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'io-7':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            _, samples_extra = random.choice(list(examples.items()))
            samples_extra = samples_extra['metadata']['examples_seen']
            samples_extra = samples_extra.split("\n\n")[:2]
            samples = samples + samples_extra
            prompt = args.prompter.IO_prompt(samples)
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'io-8':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            _, samples_extra = random.choice(list(examples.items()))
            samples_extra = samples_extra['metadata']['examples_seen']
            samples_extra = samples_extra.split("\n\n")[:3]
            samples = samples + samples_extra
            prompt = args.prompter.IO_prompt(samples)
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'io-20':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            for _ in range(3):
                _, samples_extra = random.choice(list(examples.items()))
                samples_extra = samples_extra['metadata']['examples_seen']
                samples_extra = samples_extra.split("\n\n")
                samples = samples + samples_extra
            prompt = args.prompter.IO_prompt(samples)
            response = args.inductor.beamsearch(prompt, num_beams=5)
        elif args.mode == 'hs' or args.mode == 'hs+d':
            prompt = example['input']
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            texts = []
            scores = []
            for i in range(5):
                text, score = args.inductor.self_consistency(prompt, num_beams=1)
                texts.append(text)
                scores.append(score)
            response = [(select_hypothesis(task_name, texts, samples, args), 0)]
        elif args.mode == 'hs+r':
            prompt = example['input']
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            texts = []
            scores = []
            for i in range(5):
                text, score = args.inductor.self_consistency(prompt, num_beams=1)
                texts.append(text)
                scores.append(score)
            hypothesis, eval_results = select_hypothesis(task_name, texts, samples, args, return_eval_results=True)
            score = sum([result[-1] for result in eval_results])
            prompt = f"Rewrite the following instruction to make it better in predicting the output based on the input.\n\nPrevious Instruction: {hypothesis}\n\n"
            for i in range(5):
                flag = "correct" if eval_results[i][3] else "incorrect"
                prompt = prompt + f"{eval_results[i][0]}\n{eval_results[i][1]}\nThe correct output should be：{eval_results[i][2]}\nYour answer is {flag}\n\n"
            prompt = prompt + "\nAfter observing the input-output results of these samples, a better instruction is "
            new_hypothesis = args.inductor.beamsearch(prompt, num_beams=1)[0][0]
            new_hypothesis, new_eval_results = select_hypothesis(task_name, [new_hypothesis], samples, args, return_eval_results=True)
            new_score = sum([result[-1] for result in new_eval_results])
            print(hypothesis, score)
            print(new_hypothesis, new_score)
            response = [(new_hypothesis if new_score > score else hypothesis, 0)]
        elif args.mode == 'gdhs':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.induce(prompt, num_beams=5)
            texts = [text for text, score in response]
            response = [(select_hypothesis(task_name, texts, samples, args), 0)]
        elif args.mode == 'one-shot':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.induce([random.choice(prompt)], num_beams=5)
        elif args.mode == 'sample':
            samples = example['metadata']['examples_seen']
            samples = samples.split("\n\n")
            prompt = args.prompter.GD_prompt(samples)
            response = args.inductor.sample([random.choice(prompt)], num_beams=1)
        elif args.mode == 'io-sample':
            prompt = example['input']
            response = [args.inductor.self_consistency(prompt, num_beams=1)]
        elif args.mode == 'sc':
            prompt = example['input']
            texts = []
            scores = []
            for i in range(5):
                text, score = args.inductor.self_consistency(prompt, num_beams=1)
                texts.append(text)
                scores.append(score)
            response = sort_list_by_frequency(texts, scores)
        elif args.mode == 'sr':
            prompt = example['input']
            response = args.inductor.self_refine(prompt, num_beams=5)

        text, logprobs = response[0]

        output[id_] = dict()
        output[id_]['input'] = prompt
        output[id_]['prediction'] = text
        
        print(text)

        metadata = dict()
        metadata['logprobs'] = logprobs#response.choices[0]['logprobs']
        # metadata['finish_reason'] = response.choices[0]['finish_reason']
        output[id_]['metadata'] = metadata

        if int(id_) % 100 == 0:
            print(f'generated {id_} predictions with engine')


    output_path = f'{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}'
    Path(output_path).mkdir(exist_ok=True)

    output_with_metadata_path = f'{output_path}/predictions_with_metadata.json'
    with open(output_with_metadata_path, 'w', encoding='utf-8') as f_predictions_with_metadata:
        json.dump(output, f_predictions_with_metadata, indent=2, ensure_ascii=False)

    for id_ in output:
        del output[id_]['metadata']

    output_no_metadata_path = f'{output_path}/predictions.json'
    with open(output_no_metadata_path, 'w', encoding='utf-8') as f_predictions:
        json.dump(output, f_predictions, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--engine", type=str, default='text-davinci-002',
    #                    help='The OpenAI model that will be used to generate instructions.')
    #parser.add_argument('--organization', type=str, required=True, help='Organization for the OpenAI API.')
    #parser.add_argument('--api_key', type=str, required=True, help='API key for the OpenAI API')
    parser.add_argument('--exp_dir', type=str, default="./data/list_functions", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default='./data/instruction-induction/data/induction_input', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=50, help='Max number of tokens to generate.')
    #parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR, help='Tasks for instructions generation')
    args = parser.parse_args()

    Path(f'{args.exp_dir}/{args.out_dir}').mkdir(exist_ok=True)
    #task_list = args.tasks.split(',')

    INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                    'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                    'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                    'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                    'translation_en-fr', 'word_in_context']
    #INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)

    model_name_or_path = "../llama2-cn/llama-2-7b-chat"
    # model, tokenizer = load_base_model_and_tokenizer(model_name_or_path)
    finetune_model_path = args.exp_dir + "/ckpt/checkpoint-720"
    model, tokenizer = load_finetuned_model_and_tokenizer(model_name_or_path, finetune_model_path)

    fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
    post_prompt = "\nThe instruction was"
    prompter = Prompter(fore_prompt, post_prompt)
    inductor = LLMInductor(model, tokenizer, device='cuda:0')
    #chatgpt = LLM(model_name_or_path="chatgpt", max_token=20, device=0)

    for induction_task in tqdm(INDUCTION_TASKS):
        generate_instructions(task_name=induction_task,
                              exp_dir=args.exp_dir,
                              data_dir=args.data_dir,
                              out_dir=args.out_dir,
                              max_tokens=args.max_tokens,
                              mode=args.mode)