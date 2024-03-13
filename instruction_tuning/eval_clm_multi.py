import re
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import sys
import glob
import math
import logging
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass, field

import torch
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, HfArgumentParser, TrainingArguments
from datasets import load_dataset, concatenate_datasets, DatasetDict
from accelerate import Accelerator
from peft import PeftModel

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    # Base model parameters
    base_model_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(
                default=False, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )
    # LoRA parameters
    use_lora: bool = field(default=False, metadata={"help": "Whether to use Lora or not."})
    lora_r: int = field(default=8, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    lora_target_modules: str = field(default="q_proj,v_proj", metadata={"help": "Names of the modules to apply Lora to."})


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default='bigscience/xP3all', metadata={"help": "Path to the training file."})
    model_max_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    val_set_size: Optional[int] = field(default=2000, metadata={"help": "The validation set size. For loss checking."})

@dataclass
class CendolTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
    fp16: bool = field(
        default=False, metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    lang: str = field(default="zh", metadata={"help": "The language or language list separated by `,`, dataset will be downlaoded from HF Hub."})
    evaluation_strategy: str = field(default="steps", metadata={"help": ""})
    save_strategy: str = field(default="steps", metadata={"help": ""})
    wandb_project: str = field(default="bactrian", metadata={"help": "Weight & Bias (W&B) project name."})

# Copied from https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def eval():
    accelerator = Accelerator()
    
    # HF parser
    parser = HfArgumentParser((ModelArguments, DataArguments, CendolTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check save file
    if model_args.model_name_or_path == 'none':
        mdl_name = 'falcon-7b'
        ckpt = 0
    else:
        mdl_name, ckpt = model_args.model_name_or_path.split('/')[-2:]
        if len(ckpt) == 0:
            ckpt = 28000
        else:
            ckpt = int(ckpt.split('-')[1])
        
    # if os.path.exists(f'eval_loss/{mdl_name}${ckpt}.csv'):
    #     print(f'Skipping evaluation for {mdl_name}${ckpt}')
    #     exit()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if 'wandb' in training_args.report_to:
         os.environ["WANDB_PROJECT"] = training_args.wandb_project

    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        device_map='auto' if model_args.load_in_8bit else None,
    )

    if model_args.model_name_or_path != 'none':
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        padding_side="right"
    )
    
    # llama has no pad_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    # if model_args.load_in_8bit:
    #     model = prepare_model_for_int8_training(model)
    # else:
    #     model.enable_input_require_grads()

    # Load dataset from HF Hub
    num_lang_to_size_map = {
        1: 3600000,
        2: 1800000,
        3: 1200000,
        5: 720000,
        10: 360000,
        20: 180000,
        45: 80000
    }
    
    # Load English Eval Dataset
    eval_langs = training_args.lang.split(',')
    eval_datasets = []    
    for lang in eval_langs:
        dset = datasets.load_dataset(data_args.data_path, lang)['train']
        size = num_lang_to_size_map[len(eval_langs)]
        eval_datasets.append(dset.train_test_split(test_size=size, shuffle=True, seed=14045)['test'])
    eval_datasets = datasets.concatenate_datasets(eval_datasets)
    
    # Splitting
    if data_args.val_set_size > 0:
        eval_dataset = eval_datasets.train_test_split(
            test_size=data_args.val_set_size, shuffle=True, seed=42
        )['test']
    else:
        raise ValueError("val_set_size must large than 0.")

    # Determine model_max_length for truncation
    model_max_length = data_args.model_max_length

    def generate_and_tokenize_prompt(data_point):
        full_prompt = f'{data_point["inputs"]} {data_point["targets"]}'
        user_prompt = f'{data_point["inputs"]} '
        user_prompt_len = len(tokenizer(user_prompt, truncation=True, max_length=model_max_length)["input_ids"])
        tokenized_full_prompt = tokenizer(full_prompt + tokenizer.eos_token, truncation=True, max_length=model_max_length)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"]
        tokenized_full_prompt.pop('attention_mask')
        return tokenized_full_prompt

    #with training_args.main_process_first(desc="dataset map tokenization"):
    eval_dataset = eval_dataset.map(
        generate_and_tokenize_prompt,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        desc="preprocess val data set",
    )

    trainer = Trainer(
        model = model,
        eval_dataset = eval_dataset,
        args = training_args,
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    with torch.no_grad():
        predictions = trainer.predict(eval_dataset)

    test_loss = predictions.metrics['test_loss']
    eval_data = { 'model': [mdl_name], 'checkpoint': [ckpt], 'loss': [test_loss] }
    pd.DataFrame(eval_data).to_csv(f'eval_loss_multi/{mdl_name}${ckpt}.csv', index=False)

if __name__ == "__main__":
    eval()
