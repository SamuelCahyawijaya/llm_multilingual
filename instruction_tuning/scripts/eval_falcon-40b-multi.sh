OMP_NUM_THREADS=64
WORLD_SIZE=4
batch_size=32
grad_accum=1

# En Only
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-40b \
    --model_name_or_path none \
    --output_dir output/eval_falcon \
    --lang en \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --load_in_8bit \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --model_max_length 768 \
    --val_set_size 5000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --ddp_find_unused_parameters False \
    --group_by_length \
    --prediction_loss_only \
    --report_to none
    
# En Only
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-40b-lang-1/ \
    --output_dir output/eval_falcon \
    --lang en \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --load_in_8bit \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --model_max_length 768 \
    --val_set_size 5000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --ddp_find_unused_parameters False \
    --group_by_length \
    --prediction_loss_only \
    --report_to none
    
# # 10 langs
# CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
#     --base_model_name_or_path tiiuae/falcon-7b \
#     --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-40b-lang-10/ \
#     --output_dir output/eval_falcon \
#     --lang en \
#     --overwrite_output_dir \
#     --learning_rate 2e-4 \
#     --load_in_8bit \
#     --per_device_train_batch_size ${batch_size} \
#     --per_device_eval_batch_size ${batch_size} \
#     --model_max_length 768 \
#     --val_set_size 5000 \
#     --logging_steps 100 \
#     --preprocessing_num_workers 64 \
#     --dataloader_num_workers 64 \
#     --ddp_find_unused_parameters False \
#     --group_by_length \
#     --prediction_loss_only \
#     --report_to none

# 45 langs
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-40b-lang-45/ \
    --output_dir output/eval_falcon \
    --lang en \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --load_in_8bit \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --model_max_length 768 \
    --val_set_size 5000 \
    --logging_steps 100 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --ddp_find_unused_parameters False \
    --group_by_length \
    --prediction_loss_only \
    --report_to none