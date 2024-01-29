TASK_TO_PROMPT = {
    'eng': {
        # sst2 
        'sst2': [
            'Write a sentence with [LABELS_CHOICE] sentiment: [PREMISE].',
            'Determine whether phrase is of positive or negative sentiment:\n[PREMISE]. Sentiment: [LABELS_CHOICE]',
            'The folowing sentence: [PREMISE] is positive or negative?\n[LABELS_CHOICE]', # make new prompt - done
        ],
        #qnli
        'qnli': [
            'Determine whether the sentence [ANSWER] contains the information required to answer the question [QUESTION]\n[LABELS_CHOICE]',
             'Given the following question: [QUESTION]. Does the sentence: [ANSWER] answer it?\n[LABELS_CHOICE]',
             'Given the following sentence: [ANSWER]. Has the question: [QUESTION] been answered?\n[LABELS_CHOICE]',
        ],
        
        #wnli
        'wnli': [
            'Determine whether the sentence [SENTENCE2] contains is an entailment of the following sentence: [SENTENCE1]\n[LABELS_CHOICE]',
             'Given the following sentence: [SENTENCE1]. Does it follow that [SENTENCE2]?\n[LABELS_CHOICE]',
             'Suppose [SENTENCE1]. Can we infer that [SENTENCE2]?\n[LABELS_CHOICE]',
        ],
        
        #tweet_topic_single
        'tweet_topic_single': [
            'Write a sentence about [LABELS_CHOICE]: [PREMISE].'
            'Categorise the following tweets: \n[TWEET]. \nTopic: [LABELS_CHOICE]',
            'The following tweet: [TWEET] belongs to the [LABELS_CHOICE] category',
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
