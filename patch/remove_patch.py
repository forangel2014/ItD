import os
import shutil
import argparse

def remove_patch(transformers_path):

    utils_path = os.path.join(transformers_path, 'generation', 'utils.py')
    backup_path = os.path.join(transformers_path, 'generation', 'utils.py.bak')

    if os.path.exists(backup_path):

        if os.path.exists(utils_path):
            os.remove(utils_path)
            print(f"Removed existing file: {utils_path}")

        shutil.move(backup_path, utils_path)
        print(f"Restored backup: {utils_path}")

    link_path = os.path.join(transformers_path, 'generation', 'utils.py')

    if os.path.islink(link_path):
        os.remove(link_path)
        print(f"Removed symlink: {link_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformers_path', type=str, default=None, help='Path of the input data.')
    args = parser.parse_args()
    
    remove_patch(args.transformers_path)