sample_nums=(4 8 16 32)
data_path=$1

for sample_num in ${sample_nums[@]}; do
    python -m eval_bench.eval_mmlu_sample_reward \
        --model Meta-Llama-3-8B-Instruct \
        --data_path $data_path \
        --sample_num $sample_num
done
