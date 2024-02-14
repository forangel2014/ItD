import json
import argparse
import multiprocessing
from tqdm import tqdm
from itd.model import load_base_model_and_tokenizer, load_chatglm, load_finetuned_model_and_tokenizer
from itd.induction import LLMInductor
from itd.prompt import Prompter
from itd.lm import LLM
from utils.base import mkdir, list_subdirectories, split_list
from utils.listfunc import fore_prompt, post_prompt, load_reference
from listfunc.split_data import split_data
from listfunc.induction import induce
from listfunc.run_eval import execution, execution_lm
from listfunc.visualize import visualize

def post_process(texts):
    texts = [text.replace("<s>", "").replace("</s>", "") for text in texts]
    texts = [text.split("\n\n")[0] for text in texts]
    texts = [text.strip(' ".') for text in texts]
    #texts = [text + '.' for text in texts]
    return texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default="./exp/listfunc", help='Path of the input data.')
    parser.add_argument('--exp_dir', type=str, default="./exp/listfunc/mixtral_induced+d", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default="./data/list_functions", help='Path of the input data.')
    parser.add_argument('--train_dir', type=str, default='induction_input', help='Path of the input data.')
    parser.add_argument('--test_dir', type=str, default='raw/execute', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--device', type=int, default=9, help='Max number of tokens to generate.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    #parser.add_argument('--base_model', type=str, default="../llama2-cn/llama-2-7b-chat", help='Tasks for instructions generation')
    parser.add_argument('--base_model', type=str, default="/netcache/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1", help='Tasks for instructions generation')
    parser.add_argument('--multi_process', action="store_true", default=True, help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='Tasks for instructions generation')
    parser.add_argument('--use_deductor_during_induction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--run_induction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--run_deduction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--load_reference', action="store_true", default=False, help='Tasks for instructions generation')

    args = parser.parse_args()

    if args.use_deductor_during_induction:
        args.mode += "+d"

    mode_dir = f"{args.exp_dir}/{args.out_dir}/{args.mode}"
    mkdir(mode_dir)
    args_dict = vars(args)
    output_file = f"{mode_dir}/args.json"
    with open(output_file, "w") as f:
        json.dump(args_dict, f, indent=4)

    if args.base_model in ["chatgpt", "gpt3", "gpt4"]:
        if args.finetuned_model:
            args.chatgpt = LLM(model_name_or_path=args.base_model, max_token=args.max_tokens, device=args.device, finetuned_model_id=args.finetuned_model)
        else:
            args.chatgpt = LLM(model_name_or_path=args.base_model, max_token=args.max_tokens, device=args.device)
    else:   
        if args.finetuned_model is None:
            model, tokenizer = load_base_model_and_tokenizer(args.base_model)
        else:
            args.finetuned_model = f"{args.exp_dir}/{args.finetuned_model}"
            model, tokenizer = load_finetuned_model_and_tokenizer(args.base_model, args.finetuned_model)
        args.inductor = LLMInductor(model, tokenizer, device=args.device, post_process=post_process)
    if args.use_deductor_during_induction:
        args.deductor = LLM(model_name_or_path="chatgpt", max_token=args.max_tokens, device=args.device)
    else:
        if args.base_model in ["chatgpt", "gpt3", "gpt4"]:
            args.deductor = args.chatgpt
        else:
            base_model = model.base_model.model if args.finetuned_model else model
            args.deductor = LLM(model_name_or_path=[base_model, tokenizer], max_token=args.max_tokens, device=args.device)
    
    args.prompter = Prompter(fore_prompt, post_prompt)

    mkdir(f'{args.exp_dir}/{args.out_dir}')
    
    if args.run_induction:
    
        if args.load_reference:
            
            load_reference(args)
        
        else:
                     
            induce(args)
            
            try:
                del model, tokenizer, args.inductor, args.deductor
            except:
                pass
        
    if args.run_deduction:
        
        tasks = list_subdirectories(args.data_dir)
    
        if args.multi_process:
            
            def process_task(task_queue, args):
                for task_name in tqdm(task_queue):
                    print(f"run {task_name} execution")
                    execution(task_name, args)

            num_processes = 8
            task_queue = split_list(tasks, num_processes)
            pool = multiprocessing.Pool(processes=num_processes)

            results = []
            for i in tqdm(range(num_processes)):
                pool.apply_async(process_task, args=(task_queue[i], args))

            pool.close()
            pool.join()

        else:
            for task_name in tqdm(tasks):
                execution(task_name, args)

    visualize(args)