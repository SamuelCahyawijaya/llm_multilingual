TASK_TO_PROMPT = {
    'eng': {
        # COPA-style (no nusacrowd Tasks yet)
        'COPA': [
            {'cause': '[PREMISE] This happened because...\nHelp me pick the more plausible option: - choice1: [OPTION_1], choice2: [OPTION_2]\n\n[LABELS_CHOICE]',
             'effect': '[PREMISE] As a consequence...\nHelp me pick the more plausible option: - choice1: [OPTION_1], choice2: [OPTION_2]\n\n[LABELS_CHOICE]'},
            {'cause': '[PREMISE]\n\nselect the most plausible cause:\n - [OPTION_1]\n - [OPTION_2]\n\n[LABELS_CHOICE]',
             'effect': '[PREMISE]\n\nselect the most plausible effect:\n - [OPTION_1]\n - [OPTION_2]\n\n[LABELS_CHOICE]'},
        ],
        # IndoStoryCloze-style (no nusacrowd Tasks yet)
        'IndoStoryCloze': [
            '[PREMISE]. [LABELS_CHOICE]',
            'Continue the following paragraph:\n[PREMISE]. [LABELS_CHOICE]',
        ],
        # sst2 
        'sst2': [
            '[PREMISE]. [LABELS_CHOICE]',
            'Determine whether phrase is of positive or negative sentiment:\n[PREMISE]. [LABELS_CHOICE]',
            'The folowing sentence: [PREMISE] is positive or negative?\n[LABELS_CHOICE]', # make new prompt - done
        ],
        #qnli
        'qnli': ['Determine whether the sentence [ANSWER] contains the information required to answer the question [QUESTION]\n[LABELS_CHOICE]',
                 'Given the following question: [QUESTION]. Does the sentence: [ANSWER] answer it?\n[LABELS_CHOICE]',
                 'Given the following sentence: [ANSWER]. Has the question: [QUESTION] been answered?\n[LABELS_CHOICE]',
            
        ],
        
        #wnli
        'wnli': ['Determine whether the sentence [SENTENCE2] contains is an entailment of the following sentence: [SENTENCE1]\n[LABELS_CHOICE]',
                 'Given the following sentence: [SENTENCE1]. Does it follow that [SENTENCE2]?\n[LABELS_CHOICE]',
                 'Suppose [SENTENCE1]. Can we infer that [SENTENCE2]?\n[LABELS_CHOICE]',
        ],
        
        #tweet_topic_single
        'tweet_topic_single': [
            '[TWEET]. [LABELS_CHOICE]',
            'Categorise the following tweets: \n[TWEET]. \nCategory: [LABELS_CHOICE]',
            'The following tweet: [TWEET] belongs to the following category: [LABELS_CHOICE]',
        ],
    },
}

LABEL_LANG_MAP ={
    'haryoaw/COPAL': {
        'eng': {0: '0', 1: '1'}
    },
    'IndoStoryCloze': {
        'eng': {0: '0', 1: '1'}
    },
    'sst2': {
        'eng': {0: '0', 1: '1'}
    },
    'SetFit/qnli': {
        'eng': {0: '0', 1: '1'} # Check because labelling is 'flipped'
    },
    'SetFit/wnli': {
        'eng': {0: '0', 1: '1'}
    },
    'cardiffnlp/tweet_topic_single': {
        'eng': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    }
        
}

LANG_MAP = {
    'eng': {
        'ind': 'Indonesian',
        'xdy': 'Dayak',
        'bug': 'Buginese',
        'mad': 'Madurese',
        'bjn': 'Banjarese',
        'tiociu': 'Tiociu',
        'jav': 'Javanese',
        'sun': 'Sundanese',
        'ace': 'Acehnese',
        'ban': 'Balinese',
        'min': 'Minangkabau'
    }
}

def get_label_mapping(dset_subset, prompt_lang):
    return LABEL_LANG_MAP[dset_subset][prompt_lang]

def get_lang_name(prompt_lang, lang_code):
    return LANG_MAP[prompt_lang][lang_code]

def get_prompt(prompt_lang):
    prompt_templates = {}
    for config, prompts in TASK_TO_PROMPT[prompt_lang].items():
        prompt_templates[config] = prompts
    return prompt_templates
