# Evaluation of LLMs Should Not Ignore Non-Determinism

[**🤗 Dataset**](https://huggingface.co/datasets/goodbadgreedy/GoodBadGreedy) | [**📖 arXiv**](https://arxiv.org/abs/2407.10457)

Official repo for [The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism](https://arxiv.org/abs/2407.10457)

Authors: [Yifan Song](https://github.com/Yifan-Song793), Guoyin Wang, [Sujian Li](http://123.56.88.210/), [Bill Yuchen Lin](https://yuchenlin.xyz/).


> **TLDR:** Are there performance differences between greedy decoding and sampling methods for LLM generation? The answer is YES!

Current evaluations of large language models (LLMs) often overlook non-determinism, typically focusing on a single output per example. This limits our understanding of LLM performance variability in real-world applications. 
Our study addresses this issue by exploring key questions about the **non-determinism of LLM generations**, identifying benchmarks’ consistency regarding non-determinism, and examining unique model behaviors. 

Here are our findings:
- A notable performance gap is observed between **greedy decoding** and **sampling generation**.
- **Greedy decoding outperforms sampling** on most evaluated benchmarks, except for AlpacaEval.
- Math reasoning and code generation were most impacted by sampling variance.
- The above findings remain consistent across different sizes and families of LLMs.
- Alignment methods, e.g., DPO, can significantly reduce the sampling variance for most benchmarks.
- High temperature will significantly harm the reasoning and code generation capabilities of LLMs, while higher repetition penalty leads to improved performance on AlpacaEval.
- In the best-of-N sampling setting, 7B-level LMs have the potential to outperform GPT-4-Turbo.



## 🧩 Structure of This Project

There are two parts in this project: LLM non-determinism analysis, best-of-N experiments

`analyse`: analyse the results of non-determinism generation on seven benchmarks, including [AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval), [Arena-Hard](https://github.com/lm-sys/arena-hard-auto), [WildBench v2](https://github.com/allenai/WildBench), [MixEval](https://github.com/Psycoy/MixEval), [MMLU-Redux](http://arxiv.org/abs/2406.04127), [GSM8K](https://arxiv.org/abs/2110.14168), and [HumanEval](https://arxiv.org/abs/2107.03374).

`get_benchmark_rewards`: prepare reward data for best-of-N experiments. Using [cutting-edge reward models](https://huggingface.co/spaces/allenai/reward-bench) to generate rewards for sampled model responses.

`best_of_n_eval`: unveiling the potential of LLM non-determinism generation by using [best-of-N strategy](https://arxiv.org/abs/2306.02561), selecting the best response from N sampled generations.


## 🛠️ Setup

1. Download LLM samples from [Huggingface](https://huggingface.co/datasets/goodbadgreedy/GoodBadGreedy).
2. `pip install -r requirements.txt`

## 📊 LLM Non-Determinism Analysis

We evaluate non-determinism generation of LLMs on seven benchmarks: [AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval), [Arena-Hard](https://github.com/lm-sys/arena-hard-auto), [WildBench v2](https://github.com/allenai/WildBench), [MixEval](https://github.com/Psycoy/MixEval), [MMLU-Redux](http://arxiv.org/abs/2406.04127), [GSM8K](https://arxiv.org/abs/2110.14168), and [HumanEval](https://arxiv.org/abs/2107.03374).

| Dataset      | Instance Num. | Sample Num. | Metric   |
|--------------|---------------|-------------|----------|
| AlpacaEval 2 | 805           | 16          | LC       |
| Arena-Hard   | 500           | 16          | Win Rate |
| WildBench v2 | 1024          | 16          | WB-Score |
| MixEval      | 4000          | 16          | Score    |
| MMLU-Redux   | 3000          | 32          | Acc      |
| GSM8K        | 1319          | 128         | EM       |
| HumanEval    | 164           | 128         | Pass@1   |

Take AlpacaEval for example, you can analyse the 16 sampled generations:
```bash
bash scripts/eval_alpacaeval_sample_baseline.sh <DATA_PATH>/alpaca_eval
```

<p align="center">
<img src=assets/main.png width=800/>
</p>

From the results, we observe a consistent performance gap between greedy decoding and the sampling method.
Greedy decoding generally proves more effective for most tasks, except for AlpacaEval.

## 🚀 Best-of-N Evaluation

First, employ off-the-shelf reward models to get the rewards for LLM generations. We have implement reward modeling code for [Starling-RM](https://huggingface.co/Nexusflow/Starling-RM-34B), [Eurus-RM](https://huggingface.co/openbmb/Eurus-RM-7b), [FsfairX](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1), and [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1). Take AlpacaEval for example:
```bash
bash scripts/get_alpacaeval_sample_rewards.sh Meta-Llama-3-8B-Instruct <DATA_PATH>/alpaca_eval
```

The reward information will be saved in `benchmark_rewards/{model_name}/{dataset}/{reward_model_name}.json`.
Then, evaluate in a best-of-N setting:
```bash
bash scripts/eval_alpacaeval_sample_reward.sh <DATA_PATH>/alpaca_eval
```

<p align="center">
<img src=assets/reward.png width=800/>
</p>

With the oracle selection, even smaller LLMs like Llama-3-8B-Instruct can outperform GPT-4-Turbo on MMLU, GSM8K, and HumanEval.
Furthermore, cutting-edge reward models can also select superior responses from multiple generations, and can outperform GPT-4-Turbo on GSM8K with only 8 samples.


## 📖 Citation

If you find this repo helpful, please cite out paper:

```
@article{song2024good,
    author={Yifan Song and Guoyin Wang and Sujian Li and Bill Yuchen Lin},
    title={The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism},
    year={2024},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
