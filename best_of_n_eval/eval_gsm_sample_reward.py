import argparse
import json
import os
import random


def main(args):
    model = args.model
    raw_annotations = {}
    for sample_idx in range(args.sample_num):
        with open(os.path.join(args.data_path, "gsm", f"{model}-cot-8shot-sample-{sample_idx}", "predictions.jsonl")) as fr:
            cur_raw_annotation = [json.loads(line) for line in fr]
        metric_res = []
        for idx, item in enumerate(cur_raw_annotation):
            idx = str(idx)
            if idx not in raw_annotations:
                raw_annotations[idx] = []
            if raw_annotations[idx]:
                assert item['question'] == raw_annotations[idx][0]['question']
            raw_annotations[idx].append(item)
            metric_res.append(item['prediction'] == item['answer'])

    max_metric = []
    for key in raw_annotations:
        max_metric.append(any([item['prediction'] == item['answer'] for item in raw_annotations[key]]))
    print("Oracle Sample:", sum(max_metric) / len(max_metric))

    for file in os.listdir(os.path.join("benchmark_rewards", model, "gsm")):
        reward_name = file.split('.')[0]
        rewards = json.load(open(os.path.join("benchmark_rewards", model, "gsm", file)))

        reward_max_annotations = []

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
            cur_item = raw_annotations[key][cur_max_idx]
            reward_max_annotations.append(cur_item['prediction'] == cur_item['answer'])

        print(f"{reward_name} Reward Max Sample:", sum(reward_max_annotations) / len(reward_max_annotations))


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
        default=128,
        type=int,
    )
    args = parser.parse_args()

    main(args)
