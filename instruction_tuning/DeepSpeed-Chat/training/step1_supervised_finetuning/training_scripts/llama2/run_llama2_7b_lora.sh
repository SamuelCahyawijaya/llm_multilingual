#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# # DeepSpeed Team
# OUTPUT=$1
# ZERO_STAGE=$2
# if [ "$OUTPUT" == "" ]; then
#     OUTPUT=./output_step1_llama2_7b_lora
# fi
# if [ "$ZERO_STAGE" == "" ]; then
#     ZERO_STAGE=3
# fi
# mkdir -p $OUTPUT

# deepspeed main.py \
#    --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
#    --data_split 2,4,4 \
#    --model_name_or_path meta-llama/Llama-2-7b-hf \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --max_seq_len 512 \
#    --learning_rate 9.65e-6 \
#    --weight_decay 0. \
#    --num_train_epochs 4  \
#    --gradient_accumulation_steps 1 \
#    --lr_scheduler_type cosine \
#    --num_warmup_steps 0 \
#    --seed 1234 \
#    --gradient_checkpointing \
#    --zero_stage $ZERO_STAGE \
#    --deepspeed \
#    --lora_dim 128 \
#    --lora_module_name "layers." \
#    --output_dir $OUTPUT \
#    &> $OUTPUT/training.log

# IndoNLP Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b_lora
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path indonlp/nusa_t2t \
   --data_split 10,0,0 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 768 \
   --learning_rate 2e-5 \
   --weight_decay 0 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type linear \
   --num_warmup_steps 0 \
   --dtype fp16 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --only_optimize_lora \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   &> $OUTPUT/training-7b-lora.log
