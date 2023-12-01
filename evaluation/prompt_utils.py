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
    },
}

LABEL_LANG_MAP ={
    'haryoaw/COPAL': {
        'eng': {0: '0', 1: '1'}
    },
    'IndoStoryCloze': {
        'eng': {0: '0', 1: '1'}
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
