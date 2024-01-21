import pandas as pd
import datasets
from enum import Enum

NLU_TASK_LIST = [
    'cardiffnlp/tweet_topic_single',
    'SetFit/wnli',
    'SetFit/qnli',
    'sst2',
    'IndoStoryCloze',
    'haryoaw/COPAL'    
]
def load_nlu_datasets(lang='ind'):
    cfg_name_to_dset_map = {} # {config_name: (datasets.Dataset, task_name)

    # hack, add new Task
    class NewTasks(Enum):
        COPA = "COPA"
        IndoStoryCloze = "IndoStoryCloze"
        sst2 = "sst2"
        qnli = "qnli"
        wnli = "wnli"
        tweet_topic_single = "tweet_topic_single" 

    for task in NLU_TASK_LIST:
        if 'COPAL' in task:
            dset = datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.COPA)
        elif 'IndoStoryCloze' in task:
            df = datasets.load_dataset('indolem/indo_story_cloze')['test'].to_pandas()

            # Preprocess
            df['premise'] = df.apply(lambda x: '. '.join([
                x['sentence-1'], x['sentence-2'], x['sentence-3'], x['sentence-4']
            ]), axis='columns')
            df = df.rename({'correct_ending': 'choice1', 'incorrect_ending': 'choice2'}, axis='columns')
            df = df[['premise', 'choice1', 'choice2']]
            df['label'] = 0

            dset = datasets.Dataset.from_pandas(df)
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.IndoStoryCloze)

        elif 'sst2' in task:
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

