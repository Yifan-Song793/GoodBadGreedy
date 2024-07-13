device=cuda:0
batch_size=32

reward_model=RLHFlow/ArmoRM-Llama3-8B-v0.1
completion_model=$1
completion_path=$2

python -m get_benchmark_rewards.get_gsm_sample_rewards \
    --reward_model_path $reward_model \
    --completion_model_name $completion_model \
    --completion_path $completion_path \
    --sample_num 128 \
    --batch_size $batch_size \
    --max_length 4096 \
    --device $device