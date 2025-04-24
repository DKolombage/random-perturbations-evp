# setup_path.py
import os
import sys
def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return "ZMQ" in shell or "Notebook" in shell
    except NameError:
        return False
    
def get_repo_root():
    if is_running_in_notebook():
        # For Jupyter Notebooks
        current_path = os.getcwd()
    else:
        # For Python scripts
        current_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_path, ".."))

def add_repo_paths():
    repo_root = get_repo_root()

    # List the relevant module folders
    paths = [
        os.path.join(repo_root, "random_perturbations"),
        os.path.join(repo_root, "gridlod"),
        os.path.join(repo_root, "eigen_problem")
    ]

    for path in paths:
        if path not in sys.path:
            sys.path.append(path)