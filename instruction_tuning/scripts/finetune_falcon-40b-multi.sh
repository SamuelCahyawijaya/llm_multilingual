OMP_NUM_THREADS=64
WORLD_SIZE=4
batch_size=32
grad_accum=1

# En Only
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
    --model_name_or_path tiiuae/falcon-40b \
    --output_dir output/falcon-40b-lang-1 \
    --lang en \
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
    --lora_target_modules 'query_key_value' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi \
    --run_name falcon-40b-lang-1

# # En-Fr
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
#     --model_name_or_path tiiuae/falcon-40b \
#     --output_dir output/falcon-40b-lang-2 \
#     --lang en,fr \
#     --overwrite_output_dir \
#     --learning_rate 2e-4 \
#     --bf16 \
#     --per_device_train_batch_size ${batch_size} \
#     --per_device_eval_batch_size ${batch_size} \
#     --gradient_accumulation_steps ${grad_accum} \
#     --num_train_epochs 1 \
#     --model_max_length 768 \
#     --val_set_size 5000 \
#     --save_steps 10000 \
#     --eval_steps 10000 \
#     --logging_steps 100 \
#     --preprocessing_num_workers 64 \
#     --dataloader_num_workers 64 \
#     --gradient_checkpointing \
#     --use_lora True \
#     --lora_r 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --lora_target_modules 'query_key_value' \
#     --ddp_find_unused_parameters False \
#     --torch_compile \
#     --save_total_limit 3 \
#     --group_by_length \
#     --report_to wandb \
#     --wandb_project llm_multi \
#     --run_name falcon-40b-lang-2

# # En-Fr-Zh
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
#     --model_name_or_path tiiuae/falcon-40b \
#     --output_dir output/falcon-40b-lang-3 \
#     --lang en,fr,zh \
#     --overwrite_output_dir \
#     --learning_rate 2e-4 \
#     --bf16 \
#     --per_device_train_batch_size ${batch_size} \
#     --per_device_eval_batch_size ${batch_size} \
#     --gradient_accumulation_steps ${grad_accum} \
#     --num_train_epochs 1 \
#     --model_max_length 768 \
#     --val_set_size 5000 \
#     --save_steps 10000 \
#     --eval_steps 10000 \
#     --logging_steps 100 \
#     --preprocessing_num_workers 64 \
#     --dataloader_num_workers 64 \
#     --gradient_checkpointing \
#     --use_lora True \
#     --lora_r 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --lora_target_modules 'query_key_value' \
#     --ddp_find_unused_parameters False \
#     --torch_compile \
#     --save_total_limit 3 \
#     --group_by_length \
#     --report_to wandb \
#     --wandb_project llm_multi \
#     --run_name falcon-40b-lang-3

# # En-Fr-Zh-Es-Id
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
#     --model_name_or_path tiiuae/falcon-40b \
#     --output_dir output/falcon-40b-lang-5 \
#     --lang en,fr,zh,es,id \
#     --overwrite_output_dir \
#     --learning_rate 2e-4 \
#     --bf16 \
#     --per_device_train_batch_size ${batch_size} \
#     --per_device_eval_batch_size ${batch_size} \
#     --gradient_accumulation_steps ${grad_accum} \
#     --num_train_epochs 1 \
#     --model_max_length 768 \
#     --val_set_size 5000 \
#     --save_steps 10000 \
#     --eval_steps 10000 \
#     --logging_steps 100 \
#     --preprocessing_num_workers 64 \
#     --dataloader_num_workers 64 \
#     --gradient_checkpointing \
#     --use_lora True \
#     --lora_r 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --lora_target_modules 'query_key_value' \
#     --ddp_find_unused_parameters False \
#     --torch_compile \
#     --save_total_limit 3 \
#     --group_by_length \
#     --report_to wandb \
#     --wandb_project llm_multi \
#     --run_name falcon-40b-lang-5

# 10 langs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
    --model_name_or_path tiiuae/falcon-40b \
    --output_dir output/falcon-40b-lang-10 \
    --lang en,fr,zh,es,id,ar,vi,hi,sw,ig \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --model_max_length 640 \
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
    --lora_target_modules 'query_key_value' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi \
    --run_name falcon-40b-lang-10

# # 20 langs
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
#     --model_name_or_path tiiuae/falcon-40b \
#     --output_dir output/falcon-40b-lang-20 \
#     --lang en,fr,zh,es,id,ar,vi,hi,sw,ig,yo,lg,nso,ny,rw,tn,ts,xh,zu,sn \
#     --overwrite_output_dir \
#     --learning_rate 2e-4 \
#     --bf16 \
#     --per_device_train_batch_size ${batch_size} \
#     --per_device_eval_batch_size ${batch_size} \
#     --gradient_accumulation_steps ${grad_accum} \
#     --num_train_epochs 1 \
#     --model_max_length 768 \
#     --val_set_size 5000 \
#     --save_steps 10000 \
#     --eval_steps 10000 \
#     --logging_steps 100 \
#     --preprocessing_num_workers 64 \
#     --dataloader_num_workers 64 \
#     --gradient_checkpointing \
#     --use_lora True \
#     --lora_r 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --lora_target_modules 'query_key_value' \
#     --ddp_find_unused_parameters False \
#     --torch_compile \
#     --save_total_limit 3 \
#     --group_by_length \
#     --report_to wandb \
#     --wandb_project llm_multi \
#     --run_name falcon-40b-lang-20


# 45 langs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_clm.py \
    --model_name_or_path tiiuae/falcon-40b \
    --output_dir output/falcon-40b-lang-45 \
    --lang en,fr,zh,es,id,ar,vi,hi,sw,ig,yo,lg,nso,ny,rw,tn,ts,xh,zu,sn,ur,te,bn,mr,ta,ln,wo,gu,pa,rn,ne,eu,ca,pt,ml,ak,as,bm,fon,ki,kn,or,st,tum,tw \
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
    --lora_target_modules 'query_key_value' \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --save_total_limit 3 \
    --group_by_length \
    --report_to wandb \
    --wandb_project llm_multi \
    --run_name falcon-40b-lang-45
