import csv
import json
import argparse
#from openai import OpenAI
from utils.base import mkdir

def prepare_data_for_chatgpt(args, split):
    
    chatgpt_data = []
    with open(f"{args.exp_dir}/{split}.csv", "r") as f, open(f"{args.exp_dir}/chatgpt_{split}.jsonl", "w") as g:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            message = {"messages": [{"role": "user", "content": row[0]}, {"role": "assistant", "content": row[1]}]}
            json_line = json.dumps(message)
            g.write(json_line + '\n')
            
    #json.dump(chatgpt_data, )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default="./exp/insin", help='Path of the input data.')
    parser.add_argument('--exp_dir', type=str, default="./exp/listfunc/induced_chatgpt+d", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default='./data/instruction-induction/data', help='Path of the input data.')
    parser.add_argument('--train_dir', type=str, default='induction_input', help='Path of the input data.')
    parser.add_argument('--test_dir', type=str, default='raw/execute', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    parser.add_argument('--base_model', type=str, default="../llama2-cn/llama-2-7b-chat", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default="ckpt/checkpoint-800", help='Tasks for instructions generation')


    args = parser.parse_args()

    mkdir(args.exp_dir)
    prepare_data_for_chatgpt(args, split="train")
    prepare_data_for_chatgpt(args, split="valid")
    