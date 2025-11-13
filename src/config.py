# src/config.py
from pathlib import Path

CURR_PATH =  Path(__file__).resolve().parent        # current file path
REPO_PATH = CURR_PATH.parent                        # current repository path
NOTEBOOK_PATH = REPO_PATH / "notebooks"             # path for notebooks
DATA_PATH = REPO_PATH / "data"                      # path for saving the data
DEMO_PATH = DATA_PATH / "demo-data"                 # path for demo purpose 
SRC_PATH = REPO_PATH / "src"                        # path for other sources