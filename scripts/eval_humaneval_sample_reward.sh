sample_nums=(4 8 16 32 64 128)
data_path=$1

for sample_num in ${sample_nums[@]}; do
    python -m eval_bench.eval_humaneval_sample_reward \
        --model Yi-1.5-6B-Chat \
        --data_path $data_path \
        --sample_num $sample_num
done
