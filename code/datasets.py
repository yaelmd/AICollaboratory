from typing import *

#from sklearn.model_selection import train_test_split
import api.results as bb
import pandas as pd
import numpy as np

'''
from datasets.splits import Split
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
'''

from log_handling import LogLoader


def to_dataframe(loader: LogLoader) -> pd.DataFrame:
    """
    Columns in the output dataframe:
    - input
    - targets
    - scores
    - target_values
    - correct
    - absolute_scores
    - normalized_scores
    - metrics
    - task
    - shots
    - model_name
    - model_family
    """
    tasks: List[bb.ResultsFileData] = list(loader.load_per_model())
    dfs: List[pd.DataFrame] = []
    for task in tasks:
        for query in (task.queries or []):
            df = pd.DataFrame(query.samples)
            df['model_name'] = task.model.model_name
            df['model_family'] = task.model.model_family
            df['task'] = task.task.task_name
            df['shots'] = query.shots
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f'No data found.')

    return pd.concat(dfs, ignore_index=True)

