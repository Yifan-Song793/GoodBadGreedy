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

    max_annotations = []
    for key in raw_annotations:
        max_annotations.append(max(raw_annotations[key], key=lambda x: x['preference']))
    for item in max_annotations:
        item['generator_2'] = "max"
    print("Oracle Sample:", evaluate_alpacaeval(max_annotations)['length_controlled_winrate'])

    for file in os.listdir(os.path.join("benchmark_rewards", model, "alpacaeval")):
        reward_name = file.split('.')[0]
        rewards = json.load(open(os.path.join("benchmark_rewards", model, "alpacaeval", file)))

        reward_max_annotations = []

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
            reward_max_annotations.append(raw_annotations[key][cur_max_idx])

        for item in reward_max_annotations:
            item['generator_2'] = "reward_max"
        print(f"{reward_name} Reward Max Sample:", evaluate_alpacaeval(reward_max_annotations)['length_controlled_winrate'])


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
        default=16,
        type=int,
    )
    args = parser.parse_args()

    main(args)
