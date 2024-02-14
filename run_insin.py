import json
import torch
import argparse
import multiprocessing
from tqdm import tqdm
from utils.base import mkdir, split_list
from itd.model import load_base_model_and_tokenizer, load_chatglm, load_finetuned_model_and_tokenizer
from itd.induction import LLMInductor
from itd.prompt import Prompter
from itd.lm import LLM
from insin.induction import generate_instructions, generate_instructions_chatgpt
from insin.postprocess_instructions import group_instructions, load_reference
from insin.prepare_for_execution import create_cause_and_effect_examples, create_common_concept, create_task_examples, TASK_TO_ANSWERS
from insin.execute_instructions import execution
from insin.run_reference_eval import reference_eval
from insin.reference_eval import visualize_reference
from insin.execution_eval import visualize_execution, save_predictions_execution_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default="./exp/insin", help='Path of the input data.')
    parser.add_argument('--exp_dir', type=str, default="./exp/insin/induced1-iotraining", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default='./data/instruction-induction/data', help='Path of the input data.')
    parser.add_argument('--train_dir', type=str, default='induction_input', help='Path of the input data.')
    parser.add_argument('--test_dir', type=str, default='raw/execute', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="io", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    parser.add_argument('--device', type=int, default=0, help='Max number of tokens to generate.')
    parser.add_argument('--base_model', type=str, default="../llama2-cn/llama-2-7b-chat", help='Tasks for instructions generation')
    #parser.add_argument('--base_model', type=str, default="chatgpt", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='Tasks for instructions generation')
    parser.add_argument('--run_induction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--run_deduction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--multi_process', action="store_true", default=True, help='Tasks for instructions generation')
    parser.add_argument('--load_reference', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--use_deductor_during_induction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--sample_permutation', action="store_true", default=False, help='Tasks for instructions generation')

    args = parser.parse_args()

    if args.use_deductor_during_induction:
        args.mode += "+d"
    
    mode_dir = f"{args.exp_dir}/{args.out_dir}/{args.mode}"
    mkdir(mode_dir)
    args_dict = vars(args)
    output_file = f"{mode_dir}/args.json"
    with open(output_file, "w") as f:
        json.dump(args_dict, f, indent=4)

    INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                    'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                    'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                    'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                    'translation_en-fr', 'word_in_context']

    if args.run_induction:
    
        if args.load_reference:
            
            for induction_task in tqdm(INDUCTION_TASKS):
                load_reference(induction_task, args)
            
        else:
    
            if args.base_model == "chatgpt":
                if args.finetuned_model:
                    args.chatgpt = LLM(model_name_or_path="chatgpt", max_token=args.max_tokens, device=args.device, finetuned_model_id=args.finetuned_model)
                else:
                    args.chatgpt = LLM(model_name_or_path="chatgpt", max_token=args.max_tokens, device=args.device)
            else:   
                if args.finetuned_model is None or args.exp_dir == "./exp/insin/base":
                    model, tokenizer = load_base_model_and_tokenizer(args.base_model)
                else:
                    args.finetuned_model = f"{args.exp_dir}/{args.finetuned_model}"
                    model, tokenizer = load_finetuned_model_and_tokenizer(args.base_model, args.finetuned_model)
                args.inductor = LLMInductor(model, tokenizer, device=args.device)
            if args.use_deductor_during_induction:
                args.deductor = LLM(model_name_or_path="chatgpt", max_token=args.max_tokens, device=args.device)
            else:
                if args.base_model == "chatgpt":
                    args.deductor = args.chatgpt
                else:
                    base_model = model.base_model.model if args.finetuned_model else model
                    args.deductor = LLM(model_name_or_path=[base_model, tokenizer], max_token=args.max_tokens, device=args.device)

            fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
            post_prompt = "\nThe instruction was"
            args.prompter = Prompter(fore_prompt, post_prompt)


            print(f"inducing instructions")
            if args.base_model == "chatgpt":
                for induction_task in tqdm(INDUCTION_TASKS):
                    generate_instructions_chatgpt(induction_task, args)    
            else:
                for induction_task in tqdm(INDUCTION_TASKS):
                    generate_instructions(induction_task, args)

            print(f"grouping instructions")
            for induction_task in tqdm(INDUCTION_TASKS):
                group_instructions(induction_task, args)
        
        print(f"running reference-based evaluation")
        for induction_task in tqdm(INDUCTION_TASKS):
            reference_eval(induction_task, args) 

        #visualize_reference(args)
        
        try:
            del model, tokenizer, args.inductor, args.deductor
        except:
            pass

    if args.run_deduction:

        print(f"preparing data for execution")
        for induction_task in tqdm(INDUCTION_TASKS):
            if induction_task not in ['cause_and_effect', 'common_concept']:
                task_answers_key = TASK_TO_ANSWERS.get(induction_task)
                create_task_examples(induction_task, args)
            elif induction_task == 'cause_and_effect':
                create_cause_and_effect_examples(args)
            elif induction_task == 'common_concept':
                create_common_concept(args)

        print(f"chatgpt executing")

        if args.multi_process:
            
            def process_task(task_queue, args):
                for induction_task in tqdm(task_queue):
                    print(f"run {induction_task} execution")
                    execution(induction_task, args)

            num_processes = 8
            task_queue = split_list(INDUCTION_TASKS, num_processes)
            pool = multiprocessing.Pool(processes=num_processes)

            results = []
            for i in tqdm(range(num_processes)):
                pool.apply_async(process_task, args=(task_queue[i], args))

            pool.close()
            pool.join()

        else:
            for induction_task in tqdm(INDUCTION_TASKS):
                execution(induction_task, args)

        print(f"running execution-based evaluation")
        for induction_task in tqdm(INDUCTION_TASKS):
            save_predictions_execution_accuracy(induction_task, args) 

    visualize_execution(args)