TASK_TO_PROMPT = {
    'eng': {
        # COPA-style (no nusacrowd Tasks yet)
        'COPA': [
            {'cause': '[PREMISE] This happened because...\nHelp me pick the more plausible option: - choice1: [OPTION_1], choice2: [OPTION_2]\n\n[LABELS_CHOICE]',
             'effect': '[PREMISE] As a consequence...\nHelp me pick the more plausible option: - choice1: [OPTION_1], choice2: [OPTION_2]\n\n[LABELS_CHOICE]'},
            {'cause': '[PREMISE]\n\nselect the most plausible cause:\n - [OPTION_1]\n - [OPTION_2]\n\n[LABELS_CHOICE]',
             'effect': '[PREMISE]\n\nselect the most plausible effect:\n - [OPTION_1]\n - [OPTION_2]\n\n[LABELS_CHOICE]'},
            {'cause': '[PREMISE_STRIP] because [LABELS_CHOICE]',
             'effect': '[PREMISE_STRIP] therefore [LABELS_CHOICE]'},
            {'cause': '[PREMISE_STRIP]. What was the cause? [LABELS_CHOICE]',
             'effect': '[PREMISE_STRIP]. What happened as a result? [LABELS_CHOICE]'},
        ],
        # MABL-style (no nusacrowd Tasks yet)
        'MABL': [
            'The sentence "[PREMISE]" implies [LABELS_CHOICE]',
            '"[PREMISE]" suggests that [LABELS_CHOICE]',
            '[PREMISE]\n\nThe statement above implies: [LABELS_CHOICE]',
            '[PREMISE]\n\nThe above statement entails [LABELS_CHOICE]'
        ],
        # MAPS-style (no nusacrowd Tasks yet)
        'MAPS': [
            'Question: What does the person mean by the proverb?\nProverb: [PREMISE]\nContext: [CONTEXT]\nChoices: A: [OPTION_1] B: [OPTION_2]\nAnswer: [LABELS_CHOICE]',
            'Question: What does the proverb means in this context?\nProverb: [PREMISE]\nContext: [CONTEXT]\nChoices: A: [OPTION_1] B: [OPTION_2]\nAnswer: [LABELS_CHOICE]',
            'Question: Which sense is more suitable given the following proverb and its context?\nProverb: [PREMISE]\nContext: [CONTEXT]\nChoices: A: [OPTION_1] B: [OPTION_2]\nAnswer: [LABELS_CHOICE]',
            'Question: Which interpretation is more likely to define the proverb?\nProverb: [PREMISE]\nContext: [CONTEXT]\nChoices: A: [OPTION_1] B: [OPTION_2]\nAnswer: [LABELS_CHOICE]',
        ],
        # IndoStoryCloze-style (no nusacrowd Tasks yet)
        'IndoStoryCloze': [
            '[PREMISE]. [LABELS_CHOICE]',
            'Continue the following paragraph:\n[PREMISE]. [LABELS_CHOICE]',
            '[PREMISE]\nMake a sentence to continue the paragraph above: [LABELS_CHOICE]',
            '[PREMISE]\nWhat sentence is suitable to follow the paragraph above? Answer: [LABELS_CHOICE]',
        ],
    },
    'ind': {
        # COPA-style (no nusacrowd Tasks yet)
        'COPA': [
            {'cause': '[PREMISE] Ini terjadi karena...\nBantu saya memilih opsi yang paling mungkin: - opsi1: [OPTION_1], opsi2: [OPTION_2]\n\n[LABELS_CHOICE]',
             'effect': '[PREMISE] Konsekuensinya...\nBantu saya memilih opsi yang paling mungkin: -opsi1: [OPTION_1], opsi2: [OPTION_2]\n\n[LABELS_CHOICE]'},
            {'cause': '[PREMISE]\n\npilih penyebab yang paling mungkin:\n - [OPTION_1]\n - [OPTION_2]\n\n[LABELS_CHOICE]',
             'effect': '[PREMISE]\n\npilih efek yang paling mungkin:\n - [OPTION_1]\n - [OPTION_2]\n\n[LABELS_CHOICE]'},
            {'cause': '[PREMISE_STRIP] karena [LABELS_CHOICE]',
             'effect': '[PREMISE_STRIP] maka [LABELS_CHOICE]'},
            {'cause': '[PREMISE_STRIP]. Apa penyebabnya? [LABELS_CHOICE]',
             'effect': '[PREMISE_STRIP]. Apa akibatnya? [LABELS_CHOICE]'},
        ],
        # MABL-style (no nusacrowd Tasks yet)
        'MABL': [
            'Dari kalimat "[PREMISE]" bisa disimpulkan bahwa [LABELS_CHOICE]',
            '"[PREMISE]" menunjukan [LABELS_CHOICE]',
            '[PREMISE]\n\nKalimat diatas menyatakan bahwa: [LABELS_CHOICE]',
            '[PREMISE]\n\nKalimat tersebut menyimpulkan [LABELS_CHOICE]'
        ],
        # MAPS-style (no nusacrowd Tasks yet)
        'MAPS': [
            'Pertanyaan: Apa yang orang itu maksud dengan peribahasa berikut?\nPeribahasa [PREMISE]\nKonteks: [CONTEXT]\nPilihan: A: [OPTION_1] B: [OPTION_2]\nJawaban: [LABELS_CHOICE]',
            'Pertanyaan: Apa arti peribahasa dalam konteks berikut?\nPeribahasa [PREMISE]\nKonteks: [CONTEXT]\nPilihan: A: [OPTION_1] B: [OPTION_2]\nJawaban: [LABELS_CHOICE]',
            'Pertanyaan: Makna apa yang lebih tepat untuk mengartikan peribahasa dari konteks berikut?\nPeribahasa [PREMISE]\nKonteks: [CONTEXT]\nPilihan: A: [OPTION_1] B: [OPTION_2]\nJawaban: [LABELS_CHOICE]',
            'Pertanyaan: Interpretasi mana yang lebih memungkinkan untuk mendefinisikan peribahasa berikut?\nPeribahasa [PREMISE]\nKonteks: [CONTEXT]\nPilihan: A: [OPTION_1] B: [OPTION_2]\nJawaban: [LABELS_CHOICE]',
        ],    
        # IndoStoryCloze-style (no nusacrowd Tasks yet)
        'IndoStoryCloze': [
            '[PREMISE]. [LABELS_CHOICE]',
            'Lanjutkan paragraf berikut:\n[PREMISE]. [LABELS_CHOICE]',
            '[PREMISE]\nBuat kalimat untuk melanjutkan paragraf diatas: [LABELS_CHOICE]',
            '[PREMISE]\nKalimat apa yang sesuai untuk menyambung paragraf diatas? Jawaban: [LABELS_CHOICE]',
        ]
    }
}

LABEL_LANG_MAP ={
    'haryoaw/COPAL': {
        'eng': {0: '0', 1: '1'},
        'ind': {0: '0', 1: '1'}
    },
    'MABL/id': {
        'eng': {0: '0', 1: '1'},
        'ind': {0: '0', 1: '1'}
    },
    'MABL/jv': {
        'eng': {0: '0', 1: '1'},
        'ind': {0: '0', 1: '1'}
    },
    'MABL/su': {
        'eng': {0: '0', 1: '1'},
        'ind': {0: '0', 1: '1'}
    },
    'MAPS': {
        'eng': {'a': 'A', 'b': 'B'},
        'ind': {'a': 'A', 'b': 'B'}
    },
    'MAPS/figurative': {
        'eng': {'a': 'A', 'b': 'B'},
        'ind': {'a': 'A', 'b': 'B'}
    },
    'MAPS/non_figurative': {
        'eng': {'a': 'A', 'b': 'B'},
        'ind': {'a': 'A', 'b': 'B'}
    },
    'IndoStoryCloze': {
        'eng': {0: '0', 1: '1'},
        'ind': {0: '0', 1: '1'}
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
