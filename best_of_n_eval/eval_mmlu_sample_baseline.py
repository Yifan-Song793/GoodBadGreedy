import os
import json
import random
import argparse

import pandas as pd


def main(args):
    model = args.model
    raw_annotations = {}
    raw_metrics = []

    for sample_idx in range(args.sample_num):
        cur_raw_annotation = json.load(open(os.path.join(args.data_path, f"{model}_{sample_idx}.json")))
        metric_res = []
        for idx, item in enumerate(cur_raw_annotation):
            idx = str(idx)
            if idx not in raw_annotations:
                raw_annotations[idx] = []
            raw_annotations[idx].append(item)
            metric_res.append(item['correct'])
        raw_metrics.append(sum(metric_res) / len(metric_res))

    random_metric = []
    max_metric = []

    random.seed(42)
    for key in raw_annotations:
        random_metric.append(random.choice(raw_annotations[key])['correct'])
        max_metric.append(any([item['correct'] for item in raw_annotations[key]]))

    print("Best Run:", max(raw_metrics))
    print("Worst Run:", min(raw_metrics))
    print("Average:", sum(raw_metrics) / len(raw_metrics))
    print("Random Sample:", sum(random_metric) / len(random_metric))
    print("Oracle Sample:", sum(max_metric) / len(max_metric))


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
