import os
import json
import random
import argparse

import pandas as pd


def main(args):
    model = args.model
    raw_annotations = {}

    for sample_idx in range(args.sample_num):
        cur_raw_annotation = json.load(open(os.path.join(args.data_path, f"{model}_{sample_idx}.json")))
        metric_res = []
        for idx, item in enumerate(cur_raw_annotation):
            idx = str(idx)
            if idx not in raw_annotations:
                raw_annotations[idx] = []
            raw_annotations[idx].append(item)
            metric_res.append(item['correct'])

    max_metric = []
    for key in raw_annotations:
        max_metric.append(any([item['correct'] for item in raw_annotations[key]]))
    print("Oracle Sample:", sum(max_metric) / len(max_metric))

    for file in os.listdir(os.path.join("benchmark_rewards", model, "mmlu")):
        reward_name = file.split('.')[0]
        rewards = json.load(open(os.path.join("benchmark_rewards", model, "mmlu", file)))

        reward_max_metric = []

        random.seed(42)
        for key in raw_annotations:
            cur_rewards = rewards[str(key)]['rewards']
            # select the max reward from the first args.sample_num items
            cur_max_rewards, cur_max_idx = -999999, -1
            for sample_idx in cur_rewards:
                if int(sample_idx) >= args.sample_num:
                    continue
                if cur_rewards[sample_idx] > cur_max_rewards:
                    cur_max_rewards = cur_rewards[sample_idx]
                    cur_max_idx = sample_idx
            cur_max_idx = int(cur_max_idx)
            reward_max_metric.append(raw_annotations[key][cur_max_idx]['correct'])

        print(f"{reward_name} Reward Max Sample:", sum(reward_max_metric) / len(reward_max_metric))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='Meta-Llama-3-8B-Instruct',
        type=str,
    )
    parser.add_argument(
        '--data_path',
        default='',
        type=str,
    )
    parser.add_argument(
        '--sample_num',
        default=32,
        type=int,
    )
    args = parser.parse_args()

    main(args)
