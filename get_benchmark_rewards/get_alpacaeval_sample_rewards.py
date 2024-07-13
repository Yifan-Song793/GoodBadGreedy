import os
import json
import math
import argparse
from tqdm import tqdm
from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel


def get_reward(
    args, 
    model: Union[AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel],
    tokenizer: AutoTokenizer, 
    samples: List[str],
    reward_model_name=None,
):
    input_ids = []
    attention_masks = []
    encodings_dict = tokenizer(
        samples,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    ).to(args.device)
    input_ids = encodings_dict["input_ids"]
    attention_masks = encodings_dict["attention_mask"]

    mbs = args.batch_size
    out = []
    with torch.inference_mode():
        for i in tqdm(range(math.ceil(len(samples) / mbs)), desc="Computing rewards"):
            rewards = model(
                input_ids=input_ids[i * mbs : (i + 1) * mbs],
                attention_mask=attention_masks[i * mbs : (i + 1) * mbs]
            )   # [B, 1]
            if "ArmoRM" in reward_model_name:
                rewards = rewards.score
            if "FsfairX" in reward_model_name:
                rewards = rewards.logits
            out.extend(rewards.squeeze().cpu().tolist())
    return out


def main(args):
    ## Load the model and tokenizer
    reward_model_name = args.reward_model_path.split('/')[-1]
    if "Starling-RM-34B" in reward_model_name:
        from models import StarlingForSequenceClassification

        reward_model = StarlingForSequenceClassification.from_pretrained(
            args.reward_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device
        )

        reward_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        reward_tokenizer.truncation_side = "left"
    elif "Starling-RM-7B-alpha" in reward_model_name:
        from models import StarlingAlphaForSequenceClassification

        reward_model = StarlingAlphaForSequenceClassification("meta-llama/Llama-2-7b-chat-hf").to(args.device)
        checkpoint_path = os.path.join(args.reward_model_path, "pytorch_model.bin")
        reward_model.load_state_dict(torch.load(checkpoint_path), strict=False)

        reward_tokenizer = reward_model.tokenizer
        reward_tokenizer.truncation_side = "left"
    elif "Eurus" in reward_model_name:
        reward_model = AutoModel.from_pretrained(
            args.reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    elif "ArmoRM" in reward_model_name or "FsfairX" in reward_model_name:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, use_fast=True)
    else:
        raise NotImplementedError

    model = args.completion_model_name
    output_res = {}
    for sample_idx in range(args.sample_num):
        raw_completions = json.load(open(os.path.join(args.completion_path, f"{model}_{sample_idx}", "weighted_alpaca_eval_gpt4_turbo/annotations.json")))
        test_samples = []
        for item in raw_completions:
            prompt = item['instruction']
            completion = item['output_2']
            cur_msg = [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": completion,
                }
            ]
            test_samples.append(reward_tokenizer.apply_chat_template(
                conversation=cur_msg,
                tokenize=False,
                add_generation_prompt=False,
            ))

        rewards = get_reward(
            args,
            reward_model,
            reward_tokenizer,
            test_samples,
            reward_model_name=reward_model_name,
        )
        
        for idx, (item, reward) in enumerate(zip(raw_completions, rewards)):
            idx = str(idx)
            if idx not in output_res:
                output_res[idx] = {
                    "prompt": item['instruction'],
                    "rewards": {},
                }
            output_res[idx]['rewards'][sample_idx] = reward
            output_res[idx]['best_sample'] = max(output_res[idx]['rewards'], key=output_res[idx]['rewards'].get)
            output_res[idx]['best_reward'] = max(output_res[idx]['rewards'].values())

    if not os.path.exists(os.path.join(args.output_path, model)):
        os.mkdir(os.path.join(args.output_path, model))
    if not os.path.exists(os.path.join(args.output_path, model, "alpacaeval")):
        os.mkdir(os.path.join(args.output_path, model, "alpacaeval"))

    output_path = os.path.join(args.output_path, model, "alpacaeval", f"{reward_model_name}.json")
    
    json.dump(output_res, open(output_path, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reward_model_path',
        default='RLHFlow/ArmoRM-Llama3-8B-v0.1',
        type=str,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
    )
    parser.add_argument(
        '--completion_model_name',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--completion_path',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--output_path',
        default="benchmark_rewards",
        type=str,
    )
    parser.add_argument(
        "--sample_num",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--max_length",
        default=4096,
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    args = parser.parse_args()

    print(args)
    main(args)
