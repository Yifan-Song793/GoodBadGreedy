data_path=$1

python -m eval_bench.eval_mmlu_sample_baseline \
    --model Meta-Llama-3-8B-Instruct \
    --data_path $data_path \
    --sample_num 32
