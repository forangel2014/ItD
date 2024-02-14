import argparse
from utils.base import mkdir
from utils.instruction_induction import generate_data_jointly, generate_data_sequentially, prepare_data, filter_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default="./exp/insin", help='Path of the input data.')
    parser.add_argument('--exp_dir', type=str, default="./exp/insin/induced1_ood1", help='Path of the input data.')
    parser.add_argument('--data_dir', type=str, default='./data/instruction-induction/data', help='Path of the input data.')
    parser.add_argument('--train_dir', type=str, default='induction_input', help='Path of the input data.')
    parser.add_argument('--test_dir', type=str, default='raw/execute', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='induction_out', help='Path for saving the predictions.')
    parser.add_argument('--mode', type=str, default="gd", help='Path of the input data.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    parser.add_argument('--base_model', type=str, default="../llama2-cn/llama-2-7b-chat", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='Tasks for instructions generation')
    parser.add_argument('--num_inst', type=int, default=500, help='Max number of tokens to generate.')
    parser.add_argument('--num_samples_per_inst', type=int, default=5, help='Max number of tokens to generate.')
    parser.add_argument('--ratio', type=float, default=0.99, help='Max number of tokens to generate.')
    parser.add_argument('--load_instruction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--load_instance', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--load_from_induced', type=str, default="./exp/insin/base/induction_out/io-sample", help='Tasks for instructions generation')
    parser.add_argument('--load_x', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--use_deductor_during_induction', action="store_true", default=False, help='Tasks for instructions generation')
    parser.add_argument('--ood_tasks', type=str, default=None, help='Tasks for instructions generation')

    args = parser.parse_args()

    if args.use_deductor_during_induction:
        args.exp_dir += "+d"
        
    mkdir(args.exp_dir)
    generate_data_sequentially(args)
    prepare_data(args)