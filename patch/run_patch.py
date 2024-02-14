import os
import shutil
import argparse

def apply_patch(transformers_path, patch_path):

    utils_path = os.path.join(transformers_path, 'generation', 'utils.py')
    backup_path = os.path.join(transformers_path, 'generation', 'utils.py.bak')

    if not os.path.exists(backup_path):
        shutil.move(utils_path, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        print("Patch already installed. Skipping the process.")

    link_path = os.path.join(transformers_path, 'generation', 'utils.py')

    if os.path.exists(link_path):
        if os.path.isfile(link_path):
            os.remove(link_path)
            print(f"Removed existing file: {link_path}")
        else:
            raise Exception(f"Existing path is not a file: {link_path}")

    os.symlink(patch_path, link_path)
    print(f"Created symlink: {link_path} -> {patch_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformers_path', type=str, default=None, help='Path of the input data.')
    args = parser.parse_args()
    
    transformers_path = "/data/sunwangtao/.conda/envs/kid/lib/python3.8/site-packages/transformers"
    patch_filename = "utils.py"

    work_dir = os.path.abspath(os.getcwd())

    patch_path = os.path.join(work_dir, patch_filename)

    apply_patch(transformers_path, patch_path)