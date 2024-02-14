import argparse
import json
import re
from bert_score import score
from nltk import bleu, meteor
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm
# import nltk
# nltk.download('wordnet')

from collections import Counter

scorer = RougeScorer(['rougeL'], use_stemmer=True)

def rouge(references, hypothesis):
    scores = []
    for reference in references:
        scores.append(
            scorer.score(
                reference, 
                hypothesis)['rougeL'][2]
        )
    
    return max(scores)

INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']

def reference_eval(task_name, args):
    references = json.load(open(f"{args.data_dir}/annotations/{task_name}.json", 'r'))["annotations"]
    hypotheses_metadata = json.load(open(f"{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}/prediction_groups.json", 'r'))["metadata"]
    hypotheses = hypotheses_metadata["unique_predictions_counter"]
    cnt = sum(hypotheses.values())
    
    weighted_bleu = weighted_meteor = weighted_rouge = weighted_bert_score = 0
    
    for hypothesis, cnt_hypothesis in hypotheses.items():
        weight = cnt_hypothesis / cnt
        weighted_bleu += bleu([reference.split() for reference in references], hypothesis.split(), weights=(0.25, 0.25, 0.25, 0.25)) * weight
        weighted_meteor += meteor([reference.split() for reference in references], hypothesis.split()) * weight
        weighted_rouge += rouge(references, hypothesis) * weight
        #P, R, F1 = score([hypothesis], [references], lang='en', verbose=True)
        # print(P, R, F1)
        # print(hypothesis)
        # print(references)
        #weighted_bert_score += F1.tolist()[0] * weight
    
    performance = {"weighted_bleu": weighted_bleu, "weighted_meteor": weighted_meteor, "weighted_rouge": weighted_rouge, "weighted_bert_score": weighted_bert_score}
    
    json.dump(performance, open(f"{args.exp_dir}/{args.out_dir}/{args.mode}/{task_name}/reference_eval.json", 'w'))
    #print(performance)

if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/instruction-induction/data/annotations', help='Path of the input data.')
    parser.add_argument('--exp_dir', type=str, default="./exp/insin/llama2test", help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR, help='Tasks to postprocess')
    args = parser.parse_args()

    task_list = args.tasks.split(',')

    for induction_task in task_list:
        eval(induction_task, args)