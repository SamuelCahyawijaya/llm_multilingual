OMP_NUM_THREADS=128
WORLD_SIZE=4

# 70B Full Fine-Tuning
batch_size=32
grad_accum=1

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
    --model_name_or_path mosaicml/mpt-30b \
    --output_dir output/mpt-30b \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --sample_size 15000 \
    --val_set_size 5000 \
    --save_steps 5000 \
    --eval_steps 5000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --gradient_checkpointing \
    --use_lora True \
    --lora_r 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules 'Wqkv' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project samcah \
    --run_name samcah-mpt-30b