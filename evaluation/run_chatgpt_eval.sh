# Baseline 7B
CUDA_VISIBLE_DEVICES=2 python main_chatgpt_eval_prompt.py tiiuae/falcon-7b &

# Falcon 7B
CUDA_VISIBLE_DEVICES=2 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-1/checkpoint-20000 &
CUDA_VISIBLE_DEVICES=2 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-3/checkpoint-20000 &
CUDA_VISIBLE_DEVICES=2 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-2/checkpoint-20000 &
CUDA_VISIBLE_DEVICES=1 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-5/checkpoint-20000 &
CUDA_VISIBLE_DEVICES=1 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-10/checkpoint-20000 &
CUDA_VISIBLE_DEVICES=1 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-20/checkpoint-20000 &
CUDA_VISIBLE_DEVICES=1 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-7b-lang-45/checkpoint-20000 &

wait
wait
wait
wait
wait
wait
wait
wait

# Baseline 40B
CUDA_VISIBLE_DEVICES=2 python main_chatgpt_eval_prompt.py tiiuae/falcon-40b &

# Falcon 40B
CUDA_VISIBLE_DEVICES=1 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-40b-lang-1/checkpoint-20000 &

wait
wait

CUDA_VISIBLE_DEVICES=2 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-40b-lang-10/checkpoint-10000 &
CUDA_VISIBLE_DEVICES=1 python main_chatgpt_eval_prompt.py /home/scahyawijaya/multilingual-cognition/instruction_tuning/output/falcon-40b-lang-45/checkpoint-20000 &

wait
wait
