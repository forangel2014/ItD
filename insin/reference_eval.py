import os
import argparse
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.base import list_subdirectories, generate_markdown_table

INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def task_performance(exp_dir):
    
    try:
        performance = {}
        
        for task in INDUCTION_TASKS:
            
            reference_eval = json.load(open(f"{exp_dir}/{task}/reference_eval.json"))
            performance[task] = reference_eval
            
        return performance
    
    except:

        return False

def invert_performance(performance):
    inverted_performance = {}
    for setting, modes in performance.items():
        for mode, tasks in modes.items():
            setting_mode = f"{setting}-{mode}"
            num_tasks = len(tasks)
            for task, metrics in tasks.items():
                for metric, value in metrics.items():
                    if metric not in inverted_performance.keys():
                        inverted_performance[metric] = {"average": {}}
                    if task not in inverted_performance[metric].keys():
                        inverted_performance[metric][task] = {}
                    inverted_performance[metric][task][setting_mode] = value
                    if setting_mode not in inverted_performance[metric]["average"].keys():
                        inverted_performance[metric]["average"][setting_mode] = 0
                    inverted_performance[metric]["average"][setting_mode] += value/num_tasks

    return inverted_performance


def visualize_reference(args):
    
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
                 
    performance = invert_performance(performance)

    json.dump(performance, open(f"./exp/insin/reference_eval.json", 'w'))

    order = ["base-io", "base-sc", "base-hs+d", "induced1-gd", "icl_prior-gd", "chatgpt-io", "induced_chatgpt-io", "reference-io"]

    for metric in performance.keys():
        m = len(performance[metric])
        fig, axs = plt.subplots(4, 6, figsize=(40, 60))
        fig.suptitle(metric)

        idx = 0
        for _, task in enumerate(performance[metric].keys()):
            if task != "average":
                i = idx // 6
                j = idx % 6
                idx += 1
                ax = axs[i][j]
                ax.set_title(task)
                ax.set_xlabel('Setting-Mode')
                ax.set_ylabel(metric)
                #ax.set_xticks(range(len(performance[metric][task])))
                #ax.set_xticklabels(performance[metric][task].keys())

                #values = performance[metric][task].values()
                ax.set_xticks(range(len(order)))
                #ax.set_xticklabels(performance[task].keys())
                ax.set_xticklabels(order)
                #values = performance[task].values()
                values = [performance[metric][task][method] for method in order]
                ax.bar(range(len(values)), values)
                ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f"./exp/insin/{metric}.png")
        plt.close()

        markdown_table = generate_markdown_table(performance[metric], order)

        # Save markdown table to file
        with open(f"./exp/insin/{metric}.md", "w") as file:
            file.write(markdown_table)


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="./exp/insin", help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR, help='Tasks to postprocess')
    args = parser.parse_args()

    visualize_reference(args)