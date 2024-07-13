import json
import os
import argparse
from tqdm import tqdm

import numpy as np


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

    for model in models:
        cur_res = []
        for sample_idx in tqdm(range(sample_num)):
            with open(os.path.join(args.data_path, f"{model}_nochat_sample_{sample_idx}", "codex_eval_predictions.jsonl_results.jsonl")) as fr:
                cur_raw_annotation = [json.loads(line) for line in fr]
            metric_res = []
            for idx, item in enumerate(cur_raw_annotation):
                idx = str(idx)
                metric_res.append(item['passed'])
            cur_res.append(sum(metric_res) / len(metric_res) * 100)
        cur_res = np.array(cur_res)
        print(f"{model}:")
        print(f"Mean: {cur_res.mean()}")
        print(f"Std: {cur_res.std()}")
        print(f"Max: {cur_res.max()}")
        print(f"Min: {cur_res.min()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='benchmark_res',
        type=str,
    )
    parser.add_argument(
        '--sample_num',
        default=128,
        type=int,
    )
    args = parser.parse_args()

    main(args)
