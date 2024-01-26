OMP_NUM_THREADS=64
WORLD_SIZE=1
batch_size=128

# Baseline
CUDA_VISIBLE_DEVICES=5 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
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
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-1/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-1/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-1/checkpoint-20000 \
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
    
# En-Fr
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-2/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-2/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-2/checkpoint-20000 \
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

# En-Fr-Zh
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-3/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-3/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-3/checkpoint-20000 \
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

# En-Fr-Zh-Es-Id
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-5/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-5/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-5/checkpoint-20000 \
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


# 10 langs
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-10/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-10/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-10/checkpoint-20000 \
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

# 20 langs
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-20/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-20/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-20/checkpoint-20000 \
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
    
# 45 langs
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-45/ \
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
    
CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-45/checkpoint-10000 \
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

CUDA_VISIBLE_DEVICES=4 python eval_clm_en.py \
    --base_model_name_or_path tiiuae/falcon-7b \
    --model_name_or_path /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-45/checkpoint-20000 \
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