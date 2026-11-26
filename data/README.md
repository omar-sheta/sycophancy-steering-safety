# Data

Everything the notebooks read is in this folder, so the repository runs from a fresh clone. The files are copies from the sources below, unchanged.

| File | What it is | Source | Licence |
|---|---|---|---|
| `jbb/harmful-behaviors.csv` | 100 harmful behaviours (goal, target, category), used for ASR | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT |
| `jbb/benign-behaviors.csv` | 100 paired benign behaviours, used for FRR | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT |
| `sycophancy/generate_dataset.json` | 1,000 contrastive A/B pairs, used to extract the steering vector | [yetiiil/analyse-sv-safety](https://github.com/yetiiil/analyse-sv-safety) (`datasets/generate/sycophancy/`), originally from [nrimsky/CAA](https://github.com/nrimsky/CAA) | MIT |
| `sycophancy/test_dataset_ab.json` | 50 held-out A/B items, used for the steering sanity check | [yetiiil/analyse-sv-safety](https://github.com/yetiiil/analyse-sv-safety) (`datasets/test/sycophancy/`), originally from [nrimsky/CAA](https://github.com/nrimsky/CAA) | MIT |

The sycophancy pairs are a mix of two Anthropic model-written evals (CC BY 4.0):

- [Sycophancy on NLP Survey](https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/sycophancy/sycophancy_on_nlp_survey.jsonl)
- [Sycophancy on Political Typology Quiz](https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl)

Papers: [JailbreakBench](https://arxiv.org/abs/2404.01318), [CAA](https://arxiv.org/abs/2312.06681), [Li et al.](https://arxiv.org/abs/2603.24543).

## Not included

- **The model.** `meta-llama/Llama-2-7b-chat-hf` is gated. Request access on [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
- **Raw model replies.** Some are completions of harmful prompts. Aggregated results and per-prompt verdict labels are in `results/`.
