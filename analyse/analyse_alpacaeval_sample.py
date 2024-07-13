import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import tiktoken

from best_of_n_eval import evaluate_alpacaeval


models = [
    "Meta-Llama-3-8B-Instruct",
    "Yi-1.5-6B-Chat",
    "Yi-1.5-9B-Chat",
    "Yi-1.5-34B-Chat",
    "Qwen2-7B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    "gpt-4-turbo-2024-04-09"
]


def main(args):
    sample_num = args.sample_num

    encoder = tiktoken.get_encoding("cl100k_base")

    for model in models:
        cur_res = []
        cur_len_list = []
        for sample_idx in tqdm(range(sample_num)):
            try:
                raw = json.load(open(f"{args.data_path}/{model}_{sample_idx}/weighted_alpaca_eval_gpt4_turbo/annotations.json"))
            except:
                print(f"Failed to load {model}_{sample_idx}")
                continue
            metric = evaluate_alpacaeval(raw)
            cur_res.append(metric['length_controlled_winrate'])
            cur_len = []
            for item in raw:
                try:
                    cur_len.append(len(encoder.encode(item['output_2'])))
                except:
                    print(f"Failed to encode {model}_{sample_idx}")
                    continue
            cur_len_list.append(sum(cur_len) / len(cur_len))
        cur_res = np.array(cur_res)
        print(f"{model}:")
        print(f"Mean: {cur_res.mean()}")
        print(f"Std: {cur_res.std()}")
        print(f"Max: {cur_res.max()}")
        print(f"Min: {cur_res.min()}")
        print(f"Average Length: {sum(cur_len_list) / len(cur_len_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='benchmark_res',
        type=str,
    )
    parser.add_argument(
        '--sample_num',
        default=16,
        type=int,
    )
    args = parser.parse_args()

    main(args)
