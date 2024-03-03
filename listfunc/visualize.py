import os
import argparse
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.base import list_subdirectories, generate_markdown_table

def task_performance(mode_dir):
    
    try:
        
        task_performance = {}
        tasks = list_subdirectories(mode_dir)
        
        for task in tasks:
        
            performance = json.load(open(f"{mode_dir}/{task}/execution.json", 'r'))
            task_performance[task] = performance
        
        task_performance["average"] = sum(task_performance.values())/len(task_performance)
        
        return task_performance
    
    except:
        return False

def invert_performance(performance):
    inverted_performance = {"average": {}}
    for setting, modes in performance.items():
        for mode, tasks in modes.items():
            setting_mode = f"{setting}-{mode}"
            num_tasks = len(tasks)
            for task, value in tasks.items():
                if task not in inverted_performance.keys():
                    inverted_performance[task] = {}
                inverted_performance[task][setting_mode] = value
                # if setting_mode not in inverted_performance["average"].keys():
                #     inverted_performance["average"][setting_mode] = 0
                # inverted_performance["average"][setting_mode] += value/num_tasks

    return inverted_performance

def visualize(args):
    
    order = ["mixtral-io", "mixtral-io_sample", "mixtral-sc", "mixtral-hs", "mixtral-hsr", "mixtral-hs+d", "mixtral_induced-gd", "mixtral_induced+d-gd", "mixtral-3-gd", "mixtral_io-io-2", "mixtral_io-io", "mixtral_io-io-8", "mixtral_io-io-20", "mixtral_induced-gd-2", "mixtral_induced-gd-8", "mixtral_induced-gd-20", "chatgpt-io", "induced_chatgpt-io", "reference-io"]

    performance = {}
    settings = list_subdirectories(args.task_dir)
    
    for setting in settings:
        
        setting_dir = f"{args.task_dir}/{setting}/{args.out_dir}"
        
        modes = list_subdirectories(setting_dir)
        
        for mode in modes:

            mode_dir = f"{args.task_dir}/{setting}/{args.out_dir}/{mode}"

            setting_mode_performance = task_performance(mode_dir)
            
            if setting_mode_performance:
                
                if setting not in performance.keys():
                    
                    performance[setting] = {}
                    
                performance[setting][mode] = setting_mode_performance

    inverted_performance = invert_performance(performance)
                
    #Plotting
    # fig, ax = plt.subplots(figsize=(10, 6))

    # colors = ['blue', 'red', 'green', 'orange']  # Define colors for each setting_mode

    # for idx, setting_mode in enumerate(performance.keys()):
    #     setting_mode_performance = performance[setting_mode]
    #     scores = list(setting_mode_performance.values())
    #     ax.plot(scores, color=colors[idx % len(colors)], label=setting_mode)

    # ax.set_xlabel('')  # Remove x-axis label
    # ax.set_ylabel('Task Performance')
    # ax.set_title('Performance by Task and Setting Mode')
    # ax.legend(loc='best')

    # ax.set_ylim([-0.1, 1])  # Set y-axis range to 0-1

    # plt.tight_layout()
    # plt.savefig(f"./exp/listfunc/execution.png")
    # plt.show()
    
    markdown_table = generate_markdown_table(inverted_performance, order, tasks="average")

    # Save markdown table to file
    with open(f"./exp/listfunc/execution.md", "w") as file:
        file.write(markdown_table)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="./exp/listfunc/base", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default="./data/list_functions", help='Path of the input data.')
    parser.add_argument('--train_dir', type=str, default='induction_input', help='Path of the input data.')
    parser.add_argument('--test_dir', type=str, default='raw/execute', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    parser.add_argument('--base_model', type=str, default="../llama2-cn/llama-2-7b-chat", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='Tasks for instructions generation')
    args = parser.parse_args()

    visualize(args)