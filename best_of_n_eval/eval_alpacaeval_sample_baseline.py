import os
import json
import random
import argparse

from best_of_n_eval import evaluate_alpacaeval


def main(args):
    model = args.model
    raw_annotations = {}
    raw_metrics = []
    for sample_idx in range(args.sample_num):
        cur_raw_annotation = json.load(open(os.path.join(args.data_path, f"{model}_{sample_idx}", "weighted_alpaca_eval_gpt4_turbo/annotations.json")))
        for idx, item in enumerate(cur_raw_annotation):
            idx = str(idx)
            if idx not in raw_annotations:
                raw_annotations[idx] = []
            if raw_annotations[idx]:
                assert item['instruction'] == raw_annotations[idx][0]['instruction']
            raw_annotations[idx].append(item)
        raw_metrics.append(evaluate_alpacaeval(cur_raw_annotation)['length_controlled_winrate'])

    random_annotations = []
    max_annotations = []

    random.seed(42)
    for key in raw_annotations:
        random_annotations.append(random.choice(raw_annotations[key]))
        max_annotations.append(max(raw_annotations[key], key=lambda x: x['preference']))

    print("Best Run:", max(raw_metrics))
    print("Worst Run:", min(raw_metrics))
    print("Average:", sum(raw_metrics) / len(raw_metrics))
    for item in random_annotations:
        item['generator_2'] = "random"
    print("Random Sample:", evaluate_alpacaeval(random_annotations)['length_controlled_winrate'])
    for item in max_annotations:
        item['generator_2'] = "max"
    print("Oracle Sample:", evaluate_alpacaeval(max_annotations)['length_controlled_winrate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='Meta-Llama-3-8B-Instruct',
        type=str,
    )
    parser.add_argument(
        '--data_path',
        default='benchmark_res',
        type=str,
    )
    parser.add_argument(
        '--sample_num',
        default=4,
        type=int,
    )
    args = parser.parse_args()

    main(args)
