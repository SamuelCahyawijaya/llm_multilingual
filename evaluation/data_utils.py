import pandas as pd
import datasets
from enum import Enum

NLU_TASK_LIST = [
    'cardiffnlp/tweet_topic_single',
    'SetFit/wnli',
    'SetFit/qnli',
    'sst2',
    # 'IndoStoryCloze',
    # 'haryoaw/COPAL'    
]
def load_nlu_datasets(lang='ind'):
    cfg_name_to_dset_map = {} # {config_name: (datasets.Dataset, task_name)

    # hack, add new Task
    class NewTasks(Enum):
        sst2 = "sst2"
        qnli = "qnli"
        wnli = "wnli"
        tweet_topic_single = "tweet_topic_single" 

    for task in NLU_TASK_LIST:
        if 'sst2' in task:
            dset =  datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.sst2)

        elif 'qnli' in task:
            dset =  datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.qnli)

        elif 'wnli' in task:
            dset = datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.wnli)

        elif 'tweet_topic_single' in task:
            dset = datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.tweet_topic_single)
        
        
    return cfg_name_to_dset_map

