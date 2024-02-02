import os, sys
from enum import Enum
import re                
import json

import gdown
import numpy as np
import pandas as pd
import datasets
from numpy import loadtxt
from datasets import load_dataset, Dataset

####
# Other Classification Tasks
####

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

####
# Reasoning Task from ChatGPT Evaluation
####
alphaonly = re.compile('[^a-zA-Z ?]')

def sparta_qa(num_dataset=30, output_gold=True, exp_type='1-reasoning-type'):
    """
        SpaRTQA
        SpartQA is a textual question answering benchmark for spatial reasoning on natural language text which contains more realistic spatial phenomena not covered by prior datasets and that is challenging for state-of-the-art language models (LM). SPARTQA is built on NLVR’s images containing more objects with richer spatial structures. SPARTQA’s stories are more natural, have more sentences, and richer in spatial relations in each sentence, and the questions require deeper reasoning and have four types: find relation (FR), find blocks (FB), choose object (CO), and yes/no (YN), which allows for more fine-grained analysis of models’ capabilities
        
        {
            "data_source": ["https://github.com/HLR/SpartQA_generation",
                "https://drive.google.com/file/d/12s2olGDV0ruywPtLhGL5M-1e4CbQrA8k/view"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "spatial-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
            "note": "Because each context has several corresponding questions for each type of reasoning, we select the first sample for each type and add it to our test set. That means, each context will usually have 4 questions in our test set. Also, we take the train split because it provides image to double check the gold answer."
        }
    """
    data = open('data/human_test.json', 'r').read()
    data = data.replace('false', 'False').replace('true', 'True')
    data = eval(data)['data']
    test_examples, test_golds, test_ids, candidate_choices = [], [], [], []

    if exp_type == '1-reasoning-type':
        num_context = num_dataset
        for i in range(num_context):
            story = data[i]['story']
            questions = data[i]['questions']
            for j, q in enumerate(questions):
                if not len(q['reasoning_type']) == 1:
                    continue

                q_type = q['q_type']
                if (q_type not in ['FR', 'YN', 'CO', 'FB']):
                    continue

                if 'candidate_answers' in q.keys():
                    candidate_str = ', '.join([f'{k}. {t}' for k, t in enumerate(q['candidate_answers'])])
                    candidate_str = candidate_str.replace('DK', "don't know")
                    candidate_choices.append([c.strip() for c in q['candidate_answers']])
                    if len(candidate_choices[-1]) == 0:
                        candidate_choices[-1] = ['Yes', 'No']
                    elif candidate_choices[-1][0] != 'A':
                        candidate_choices[-1] = [str(i) for i in range(len(candidate_choices[-1]))]
                else:
                    candidate_str = ''
                    candidate_choices.append(['Yes', 'No'])

                test_examples.append(f'Given the description: {story[0]}. {q["question"]} {candidate_str}')
                test_golds.append([str(a) for a in q['answer']])
                test_ids.append(f'{i}|{j}')

    elif exp_type == '2-reasoning-type':
        count_by_q_type = {x: 0 for x in ['FR', 'YN', 'CO', 'FB']}
        num_samples_each_q_type = (num_dataset-1)//4+1
        num_samples_taken = 0

        for i in range(len(data)):
            instance = data[i]
            story = instance['story']
            questions = instance['questions']
            question_type_taken = []

            for j, q in enumerate(questions):
                if not len(q['reasoning_type']) == 2:
                    continue

                q_type = q['q_type']
                if (q_type not in ['FR', 'YN', 'CO', 'FB']):
                    continue
                else:
                    test_ids.append(f'{i}|{j}')
                    question_type_taken.append(q_type)
                    count_by_q_type[q_type] += 1
                    num_samples_taken += 1

                if 'candidate_answers' in q.keys():
                    candidate_str = ', '.join([f'{k}. {t}' for k, t in enumerate(q['candidate_answers'])])
                    candidate_str = candidate_str.replace('DK', "don't know")
                    candidate_choices.append([c.strip() for c in q['candidate_answers']])
                    if len(candidate_choices[-1]) == 0:
                        candidate_choices[-1] = ['Yes', 'No']
                    elif candidate_choices[-1][0] != 'A':
                        candidate_choices[-1] = [str(i) for i in range(len(candidate_choices[-1]))]
                else:
                    candidate_str = ''
                    candidate_choices.append(['Yes', 'No'])
                test_examples.append(f'Given the description: {story[0]}. {q["question"]} {candidate_str}')
                test_golds.append([str(a) for a in q['answer']])

    if output_gold:
        return test_examples, candidate_choices, test_ids, test_golds,
    else:
        return test_examples, candidate_choices, test_ids

def timedial(num_dataset=30):
    """
        TimeDial
        TimeDial presents a crowdsourced English challenge set, for temporal commonsense reasoning, formulated as a multiple choice cloze task with around 1.5k carefully curated dialogs. The dataset is derived from the DailyDialog (Li et al., 2017), which is a multi-turn dialog corpus. We follow the format of the task in the BIG-Bench benchmark, which is multiple-choice (single correct answer). Note that the correct answer should be 0 or 1, say if answer of ChatGPT indicates the answer as 0 or 1, we mark the answer as True.

        {
            "data_source": "https://github.com/google-research-datasets/TimeDial/blob/main/test.json",
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "temporal-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
        }
    """
    timedial_data = json.load(open('data/timedial_test.json', 'r'))
    test_examples = []
    test_ids = list(range(num_dataset))
    test_golds = ['0 or 1 (if option 1 != none)']*num_dataset

    for ex in timedial_data[:num_dataset]:
        conversation = ex['conversation']
        choices = (ex['correct1'], ex['correct2'], ex['incorrect1'], ex['incorrect2']) # ex['correct2'] if ex['correct2'] != 'none' else ex['correct1']

        conversation = 'Given the conversation:\n' + '\n'.join(conversation) + '\n'
        choices = 'Candidate choices to fill in the <mask>: ' + ' '.join([f'{j}. {t},' for j, t in enumerate(choices)]) + '\n'
        caution = 'Note that there may not be enough information to certainly fill in the <mask>, but from commonsense reasoning, you can surely narrow down what are the most probable choices to fill in the <mask>. Please select the most propable choice from candidates and explain your choice.'
        final = conversation+choices+caution
        test_examples.append(final.replace('"', "'"))

    return test_examples, [['0','1','2','3'] for i in range(len(test_ids))], test_ids, test_golds

def pep_3k(num_dataset=30, output_gold=True):
    """
        Pep-3k
        Pep-3k is a dataset of physical semantic plausibility judgments of single events. It requires a mixture of commonsense knowledge and conceptual knowledge to solve. Each event consists of a subject, a verb, and an object, i.e it has the simple s-v-o format. For example, the event can be man swallow paintball, with the label 0 (implausible). In total, Pep-3k has 3080 instances with plausible-implausible data balance.

        {
            "data_source": "https://github.com/suwangcompling/Modeling-Semantic-Plausibility-NAACL18/tree/master/data",
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "commonse-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
            "note": "download two files pos-all.txt and neg-all.txt to ./data/pep-3k/"
        }
    """
    pos_data = open('data/pep-3k/pos-all.txt', 'r').read().splitlines()
    neg_data = open('data/pep-3k/neg-all.txt', 'r').read().splitlines()
    num_data_each = num_dataset // 2
    test_examples = pos_data[:num_data_each] + neg_data[:num_data_each]
    
    test_ids = list(range(num_data_each))*2
    test_golds = ['true']*num_data_each + ['false']*num_data_each

    if output_gold:
        return test_examples, [['true', 'false'] for i in range(len(test_ids))], test_ids, test_golds
    else:
        return test_examples, [['true', 'false'] for i in range(len(test_ids))], test_ids

def step_game(exp_type, num_dataset=30): # ['hard', 'basic', 'clock-position', 'basic-cardinal', 'diagonal']
    """
        StepGame - Spatial Reasoing & Question Answeing

        {
            "data_source": ["https://github.com/ZhengxiangShi/StepGame/blob/main/Dataset/CompleteVersion/clean/qa1_test.json",
                "https://github.com/ZhengxiangShi/StepGame/blob/main/Dataset/CompleteVersion/clean/qa9_valid.json"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "spatial-reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    if exp_type == 'hard':
        path = 'data/qa9_valid.json'
    elif exp_type in ['basic', 'clock-position', 'basic-cardinal', 'diagonal']:
        path = 'data/qa1_test.json'
    
    f = open(path)
    stepgame_data = json.load(f)

    MC_text = "Choose from: left, right, above, below, lower-left, lower-right, upper-left, upper-right."
    
    if exp_type == 'basic':
        test_ids = [i for i in range(num_dataset+1)]
        test_ids.remove(4)  # gold label is wrong
    elif exp_type == 'diagonal': 
        test_ids = [30, 31, 39, 40, 41, 47, 48, 50, 52, 53, 60, 61, 63, 65, 67, 69, 71, 72, 84, 87]
    elif exp_type == 'clock-position':
        test_ids = [62, 54, 119, 131, 143, 144, 189, 316, 426, 484, 697, 820, 960, 1045, 1163, 1607, 1618, 1620, 1736, 1778]
    elif exp_type == 'basic-cardinal':
        test_ids = [1, 2, 3, 9, 10, 16, 17, 18, 19, 22, 25, 27, 33, 34, 35, 42, 49, 51, 59, 66]
    elif exp_type == 'hard':
        test_ids = [i for i in range(num_dataset)]
    
    test_examples, test_golds = [], []
    for i in test_ids:
        # ex = [stepgame_data[str(i)]]
        ex = stepgame_data[str(i)]
        if exp_type == 'hard':
            ex['input'] = "Given the description: {}. {}".format(
                ' '.join(ex['story']), ex['question'].replace('relation', 'spatial relation, (e.g left, right, above lower-left, ..)'))
       
        else:
            ex['input'] = f"{ex['story'][0]} {ex['question']} {MC_text}"
        test_examples.append(ex['input'])
        test_golds.append(ex['label'])
    return test_examples, test_ids, test_golds

def babi(exp_type = 15, prompt_engineering=False, num_dataset=30, batching=True, output_gold=True, save_csv=False):
    """
        bAbI
        This basic induction bAbI tasks is taken from the (20) QA bAbI tasks that a set of proxy tasks that evaluate 
        reading comprehension via question answering. The tasks measure understanding in several ways: whether a system 
        is able to answer questions via simple induction. 
        The tasks are designed to be prerequisites for any system that aims to be capable of conversing with a human.
    
        {
            "data_source": ["http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "deductive-inductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    
    def make_samples(filename, exp_type, prompt_engineering=False, batching=True):
        df = pd.read_csv(filename, header=None)
    
        dataset = {}
        index = 0
        golds = {}
        for i, row in df.itertuples():
            if index not in dataset.keys():
                dataset[index] = ''
                golds[index] = ''

            if 'what' in row.lower():
                
                gold = row[row.find('\t'):]
                row = row[:row.find('\t')]
                
                if batching:
                    dataset[index] += row + '\n'
                    golds[index] += gold + '\n'
                    
                    if i+1 < df.shape[0]:
                        if prompt_engineering and \
                            'what' not in df.iloc[[i+1]].values[0][0].lower():
                            if exp_type == 15:
                                marker = 'deductive' 
                            elif exp_type == 16:
                                marker = 'inductive' 
                              
                            allprev = ''.join([i for i in dataset[index].replace('\n', '') if not i.isdigit()]).strip()
                            context = allprev[:allprev.lower().find('what')]
                            question = allprev[allprev.lower().find('what'):]

                            dataset[index] = 'Given facts: ' + context +\
                                             '\n\nThe most recent fact is the correct fact.\n\nBased on the given facts above, do a reasonable inference on this question using '+marker+' reasoning: ' + question
                        
                    if i+1 < df.shape[0]:
                        if 'what' not in df.iloc[[i+1]].values[0][0].lower():
                            index += 1
                    else:
                        index += 1
                else:
                    golds[index] = gold
                    if 'what' not in df.iloc[[i-1]].values[0][0].lower():
                        context = dataset[index]

                    if not prompt_engineering:
                        dataset[index] = context + row
                    else:
                        if exp_type == 15:
                            marker = 'deductive' 
                        elif exp_type == 16:
                            marker = 'inductive' 

                        dataset[index] = 'Given facts: ' +\
                                         ''.join([i for i in context.replace('\n', '') if not i.isdigit()]).strip() +\
                                         '\n\nThe most recent fact is the correct fact.\n\nBased on the given facts above, do a reasonable inference on this question using '+marker+' reasoning: ' +\
                                         alphaonly.sub('', row).strip()

                    index += 1
                    
            else:
                dataset[index] += row + '\n'
                
        ids = dataset.keys()
        dataset_in_list = [dataset[idx] for idx in ids]
        gold_in_list = [golds[idx] for idx in ids]
        
        return dataset_in_list, gold_in_list
    
    if exp_type == 15:
        filename = 'tasks_1-20_v1-2/en/qa15_basic-deduction_test.txt'
    elif exp_type == 16:
        filename = 'tasks_1-20_v1-2/en/qa16_basic-induction_test.txt'
    else:
        raise NotImplementedError('task_id: {} is not yet implemented'.\
                                            format(exp_type))
        
    filename = 'data/' + filename
    if not os.path.isfile(filename):
        os.system('wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
        os.system('tar -xf tasks_1-20_v1-2.tar.gz')
        os.system('rm -rf tasks_1-20_v1-2.tar.gz')
        os.system('mv tasks_1-20_v1-2 data/')
    
    prompts, golds = make_samples(filename, exp_type, prompt_engineering, batching)
    
    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts[:num_dataset], 'Golds': list(map(lambda x: x.split('\t')[1], golds[:num_dataset])) })
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('babi_task_'+str(exp_type)+'.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids

def alpha_nli(num_dataset=30, output_gold=True, save_csv=False):
    """
        αNLI
        αbductive Natural Language Inference (αNLI) is a new commonsense benchmark dataset designed to test 
        an AI system’s capability to apply abductive reasoning and common sense to form possible explanations for 
        a given set of observations. Formulated as a binary-classification task, the goal is to pick the most 
        plausible explanatory hypothesis given two observations from narrative contexts.
    
        {
            "data_source": ["https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip",
                            "http://abductivecommonsense.xyz/"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "abductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    
    if not os.path.isfile('anli/test.jsonl'):
        os.system('wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip')
        os.system('/usr/bin/unzip anli.zip')
        os.system('rm -rf anli.zip')
        os.system('mv anli data/')
    
    anli_dataset = load_dataset('json', data_files='data/anli/test.jsonl')
    lines = loadtxt('data/anli/test-labels.lst', comments="#", delimiter=",", unpack=False)
    labels = [int(line) for line in lines]
    
    prompts = []
    golds = []
    for dataset_id in range(num_dataset):
        data = anli_dataset['train'][dataset_id]

        prompt = 'Given: ' + data['obs1'] + ' Then: '+ data['obs2'] + \
                ' Select the most plausible explanation (hypothesis): A. ' + data['hyp1'] + \
                ' B. ' + data['hyp2']

        prompts.append(prompt)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':labels[:num_dataset]})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('anli.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids

def clutrr(num_dataset=30, output_gold=True, save_csv=False):
    """
        CLUTRR
        CLUTRR (Compositional Language Understanding and Text-based Relational Reasoning), 
        a diagnostic benchmark suite, is first introduced in (https://arxiv.org/abs/1908.06177) 
        to test the systematic generalization and inductive reasoning capabilities of NLU systems. 
        The CLUTRR benchmark allows us to test a model’s ability for systematic generalization by 
        testing on stories that contain unseen combinations of logical rules, and test for the 
        various forms of model robustness by adding different kinds of superfluous noise facts to the stories.
    
        {
            "data_source": ["https://huggingface.co/datasets/CLUTRR/v1"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "inductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    clutrr_dataset = load_dataset("CLUTRR/v1", "gen_train23_test2to10")
    
    prompts = []
    golds = []
    for dataset_id in range(num_dataset):
        data = clutrr_dataset['test'][dataset_id]
        first_p = eval(data['query'])[1]
        second_p = eval(data['query'])[0]

        prompt = data['clean_story'] + '. Who is ' + first_p + ' to ' + second_p + '?'
        gold = data['target_text']

        prompts.append(prompt)
        golds.append(gold)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('clutrr.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids

def commonsenseqa(num_dataset=30, output_gold=True, save_csv=False):
    """
        CommonsenseQA
        CommonsenseQA is a new multiple-choice question answering dataset that requires different types of 
        commonsense knowledge to predict the correct answers . It contains 12,102 questions with one correct 
        answer and four distractor answers. The dataset is provided in two major training/validation/testing 
        set splits: "Random split" which is the main evaluation split, and "Question token split", see paper for details.
    
        {
            "data_source": ["https://huggingface.co/datasets/commonsense_qa"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "commonsense reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    commonsenseqa_dataset = load_dataset("commonsense_qa")
    
    prompts = []
    golds = []
    for i in range(num_dataset):

        gold = commonsenseqa_dataset['validation'][i]['answerKey']
        prompt = commonsenseqa_dataset['validation'][i]['question']

        choice_str = ''
        for choice_id, choice in enumerate(commonsenseqa_dataset['validation'][i]['choices']['label']):
            choice_str += choice + '. ' + commonsenseqa_dataset['validation'][i]['choices']['text'][choice_id] + ', '
        prompt += ' ' + choice_str[:-2]
        prompt

        prompts.append(prompt)
        golds.append(gold)
    
    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('commonsense_qa.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids

def ecare(num_dataset=30, output_gold=True, save_csv=False):
    """
        E-Care
        Understanding causality has vital importance for various Natural Language Processing (NLP) applications. 
        Beyond the labeled instances, conceptual explanations of the causality can provide a deep understanding 
        of the causal fact to facilitate the causal reasoning process. We present a human-annotated explainable 
        CAusal REasoning dataset (e-CARE), which contains over 20K causal reasoning questions, together with 
        natural language formed explanations of the causal questions.
    
        {
            "data_source": ["https://huggingface.co/datasets/12ml/e-CARE"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "causal reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    ecare_dataset = load_dataset("12ml/e-CARE")
    
    prompts = []
    golds = []
    gold_explanations = []

    for i in range(num_dataset):

        gold = ecare_dataset['validation'][i]['label']
        gold_explanation = ecare_dataset['validation'][i]['conceptual_explanation']

        if ecare_dataset['validation'][i]['question'] == 'cause':
            prompt = 'Choices:\nA: ' + \
                        ecare_dataset['validation'][i]['choice1'] + ' B: ' + \
                        ecare_dataset['validation'][i]['choice2'] + '\nWhich one of the choices are causing the sentence: ' + \
                        ecare_dataset['validation'][i]['premise']
        elif ecare_dataset['validation'][i]['question'] == 'effect':
            prompt = 'If ' + ecare_dataset['validation'][i]['premise'] + ' Which one of the choices are caused by that? Choices:\nA: ' + \
                        ecare_dataset['validation'][i]['choice1'] + ' B: ' + \
                        ecare_dataset['validation'][i]['choice2']
        prompts.append(prompt)
        golds.append(gold)
        gold_explanations.append(gold_explanation)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':list(map(lambda x: str(x), golds)), 'Gold_explanations':gold_explanations})
    
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('ecare.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids

def covid_factchecking(exp_type='scientific'): # ['scientific', 'social']
    """
        Covid-factchecking - Hallucination & Factuality 
            - function for both Covid-social and Covid-scientific
        {
            "data_source": ["https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/fact_checker/covid19_scientific/task.json",
                            "https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/fact_checker/politifact/task.json"],
            "evaluation-method": "human-evaluation",
            "answer-type": "true/false",
            "evaluation-details": "Feed the example to the model without any instruction and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\
        }
    """
    if exp_type == 'scientific':
        path = 'data/covid19_scientific.json'
    elif exp_type =='social':
        path = 'data/politifact.json'
    
    f = open(path)
    covid_data = json.load(f)

    count_true, count_false = 0, 0
    test_examples, test_ids = [], []

    for ex in covid_data['examples']:
        label = 'true' if ex['target_scores']['true'] == 1 else 'false'
        if label == 'true':
            count_true+=1
        else:
            count_false+=1
        test_examples.append(ex)
        test_ids.append(ex['id'])
    
    return test_examples, test_ids

def load_chatgpt_eval_tasks():
    # SpartaQA 1 Reasoning
    queries, choices, ids, labels = sparta_qa(num_dataset=49, output_gold=True, exp_type='1-reasoning-type')
    sparta_qa_1reasoning_dset = Dataset.from_pandas(pd.DataFrame({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    }))

    # SpartaQA 2 Reasoning
    queries, choices, ids, labels = sparta_qa(num_dataset=49, output_gold=True, exp_type='2-reasoning-type')
    sparta_qa_2reasoning_dset = Dataset.from_pandas(pd.DataFrame({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    }))

    # TimeDial
    queries, choices, ids, labels = timedial(num_dataset=1446)
    for i, q in enumerate(queries):
        if '1. none' in q:
            labels[i] = ['0','1']
        else:
            labels[i] = ['0']            
    timedial_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # PEP-3k
    queries, choices, ids, labels = pep_3k(num_dataset=3080)
    queries = list(map(lambda x: f'Is it true that: {x}?', queries))
    pep_3k_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # Step Game Basic
    queries, ids, labels = step_game(exp_type='basic', num_dataset=1000)
    choices = [list(set(labels))] * 1000
    step_game_basic_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # Step Game Hard
    queries, ids, labels = step_game(exp_type='hard', num_dataset=1000)
    choices = [list(set(labels))] * 1000
    step_game_hard_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # bAbI 15
    queries, ids, labels = babi(exp_type=15, prompt_engineering=True, num_dataset=1000, batching=False, output_gold=True)
    choices = [list(set(labels))] * 1000
    babi15_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # bAbI 16
    queries, ids, labels = babi(exp_type=16, prompt_engineering=True, num_dataset=1000, batching=False, output_gold=True)
    choices = [list(set(labels))] * 1000
    babi16_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # Alpha NLI
    queries, ids, labels = alpha_nli(num_dataset = 3059)
    choices = [['A', 'B'] for i in range(3059)]
    labels = list(map(lambda x: 'A' if x == 1 else 'B', labels))
    alpha_nli_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # CLUTTR
    queries, ids, labels = clutrr(num_dataset=1146)
    choices = [list(set(labels))] * 1146
    clutrr_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # CommonsenseQA
    queries, ids, labels = commonsenseqa(num_dataset=1221)
    choices = [list(set(labels))] * 1221
    commonsenseqa_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # ECare
    queries, ids, labels = ecare(num_dataset=2122)
    choices = [['A', 'B']] * 2122
    ecare_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # Covid Scientific
    examples, ids = covid_factchecking('scientific')
    queries = [ex['input'] for ex in examples]
    labels = ['true' if ex['target_scores']['true'] else 'false' for ex in examples]
    choices = [['true', 'false'] for ex in examples]
    covid_fact_scientific_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })

    # Covid Social
    examples, ids = covid_factchecking('social')
    queries = [ex['input'] for ex in examples]
    labels = ['true' if ex['target_scores']['true'] else 'false' for ex in examples]
    choices = [['true', 'false'] for ex in examples]
    covid_fact_social_dset = Dataset.from_dict({
        'id': ids,
        'question': queries,
        'choice': choices,
        'label': labels
    })
    
    return {
        'sparta_qa_1reasoning': sparta_qa_1reasoning_dset,
        'sparta_qa_2reasoning': sparta_qa_2reasoning_dset,
        'timedial': timedial_dset,
        'pep_3k': pep_3k_dset,
        'step_game_basic': step_game_basic_dset,
        'step_game_hard': step_game_hard_dset,
        'babi15': babi15_dset,
        'babi16': babi16_dset,
        'alpha_nli': alpha_nli_dset,
        'clutrr': clutrr_dset,
        'commonsenseqa': commonsenseqa_dset,
        'ecare': ecare_dset,
        'covid_fact_scientific': covid_fact_scientific_dset,
        'covid_fact_social': covid_fact_social_dset
    }