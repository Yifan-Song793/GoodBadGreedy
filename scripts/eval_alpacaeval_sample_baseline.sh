data_path=$1

python -m eval_bench.eval_alpacaeval_sample_baseline \
    --model Yi-1.5-6B-Chat \
    --data_path $data_path \
    --sample_num 16
