import json
import os
import argparse
from tqdm import tqdm

import numpy as np


models = [
    "llama_3_8b_instruct",
    "mistral_7b_instruct_v02",
    "qwen_2_7b_instruct",
    "yi_15_6b_chat",
    "yi_15_9b_chat",
    "gpt_4_turbo_2024_04_09",
    "yi_15_34b_chat",
]


def main(args):
    sample_num = args.sample_num

    for model in models:
        cur_res = []
        for sample_idx in tqdm(range(sample_num)):
            metric = json.load(open(os.path.join(args.data_path, f"{model}", "mixeval", f"{sample_idx}", "score.json")))
            cur_res.append(metric['overall score (final score)'] * 100)
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
        default=16,
        type=int,
    )
    args = parser.parse_args()

    main(args)
