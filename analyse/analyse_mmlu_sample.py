import json
import argparse
import os
from tqdm import tqdm

import numpy as np
import tiktoken


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
                cur_raw_annotation = json.load(open(os.path.join(args.data_path, f"{model}_{sample_idx}.json")))
            except:
                continue
            metric_res = []
            cur_len = []
            for idx, item in enumerate(cur_raw_annotation):
                idx = str(idx)
                metric_res.append(item['correct'])
                cur_len.append(len(encoder.encode(item['rounds'][0]['response'])))
            cur_res.append(sum(metric_res) / len(metric_res) * 100)
            cur_len_list.append(sum(cur_len) / len(cur_len))
        cur_res = np.array(cur_res)
        print(f"{model}:")
        print(f"Mean: {cur_res.mean()}")
        print(f"Std: {cur_res.std()}")
        print(f"Max: {cur_res.max()}")
        print(f"Min: {cur_res.min()}")
        print(f"Length: {sum(cur_len_list) / len(cur_len_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='benchmark_res',
        type=str,
    )
    parser.add_argument(
        '--sample_num',
        default=32,
        type=int,
    )
    args = parser.parse_args()

    main(args)
