#! /usr/bin/bash
#SBATCH --job-name=finetune_clm_7b
#SBATCH --mail-user=scahyawijaya@connect.ust.hk
#SBATCH --output=/project/emllmerobo/out.log
#SBATCH --error=/project/emllmerobo/err.log
#SBATCH -N 1 -n 56 --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=project
#SBATCH --time=72:00:00

# module load cuda12.2/blas/12.2.2
# module load cuda12.2/fft/12.2.2
# module load cuda12.2/toolkit/12.2.2

OMP_NUM_THREADS=64
batch_size=64
grad_accum=1

# Zho Only
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-only \
    --lang zho \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 2 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-only

# Zho + Multi 0.1x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_10p \
    --lang multi \
    --multilingual_size 0.1 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_10p

# Zho + Multi 0.2x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_20p \
    --lang multi \
    --multilingual_size 0.2 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_20p

# Zho + Multi 0.5x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_50p \
    --lang multi \
    --multilingual_size 0.5 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_50p

# Zho + Multi 1x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_100p \
    --lang multi \
    --multilingual_size 1 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_100p
    
# Zho + Multi 2x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_200p \
    --lang multi \
    --multilingual_size 2 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_200p
    
# Zho + Multi 3x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_300p \
    --lang multi \
    --multilingual_size 3 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_300p
    
# Zho + Multi 5x
CUDA_VISIBLE_DEVICES=0,1 accelerate launch finetune_clm_zh.py \
    --model_name_or_path tiiuae/falcon-7b \
    --output_dir output/falcon-7b-lang-zh-multi_500p \
    --lang multi \
    --multilingual_size 5 \
    --data_path ./cache \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --val_set_size 5000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'query_key_value,dense' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi_zho \
    --run_name falcon-7b-lang-zho-multi_500p