# Reproducing "Analysing the Safety Pitfalls of Steering Vectors" for sycophancy on Llama-2-7B-chat

A by-hand reproduction of the sycophancy steering experiment from [Li et al.](https://arxiv.org/abs/2603.24543) on [Llama-2-7B-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf): does steering a model toward sycophancy make it easier to jailbreak?

Steering is done with a plain PyTorch forward hook, without a steering library. Both notebooks run top to bottom from the repository root, and every path is relative to it.

## Summary

- **Setup.** A sycophancy steering vector is added to the residual stream at layer 14 with seven multipliers from -1.5 to +1.5 (0 is the unsteered baseline). Attack Success Rate (ASR) is measured on the 100 harmful prompts of [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) under three conditions (no attack, prefix injection, refusal suppression), judged by GPT-4o-mini with Li et al.'s prompt.
- **What matched.** The change in ASR at +1.5 under the two attacks: prefix injection **+20 points** (Li et al. about +15), refusal suppression **+15 points** (Li et al. about +20).
- **What did not match.** With no attack, ASR rose by **+11 points** (Li et al. about +1). For negative multipliers ASR changed by about -1 point in all three conditions (Li et al. about 0 / -6 / -3). The negative side is partly explained by lower baselines here. The no-attack gap is unexplained.
- **Added measurement.** False Refusal Rate (FRR) on the 100 benign JailbreakBench prompts fell from **65%** at -1.5 to **51%** at 0 to **36%** at +1.5, using the [XSTest](https://github.com/paul-rottger/xstest) classifier prompt on GPT-4o-mini. Li et al. report FRR only for Qwen models and for two other behaviours, so there is no published number to compare against.
- **Practical finding.** The match only appears after a normalisation step that is in Li et al.'s [code](https://github.com/yetiiil/analyse-sv-safety/blob/main/normalize_vectors.py) but not described in the paper. The vector extracted by hand has cosine similarity **0.997** with theirs, but a norm of **1.94** against **7.08**. With the raw vector the changes at +1.5 were only +2 / +4 / 0 points.

Li et al.'s comparison numbers were read off Figure 3 of their paper, so they are approximate.

## Results

### ASR, normalised vector

| m | none | prefix | refusal suppression |
|---|---|---|---|
| -1.5 | 0% | 3% | 0% |
| -1.0 | 1% | 2% | 0% |
| -0.5 | 1% | 3% | 1% |
| 0 | 1% | 4% | 1% |
| +0.5 | 5% | 7% | 5% |
| +1.0 | 5% | 10% | 11% |
| +1.5 | 12% | 24% | 16% |

![ASR vs multiplier, normalised vector](figures/asr_vs_multiplier_norm.png)

### ASR, raw vector (before normalisation)

| m | none | prefix | refusal suppression |
|---|---|---|---|
| -1.5 | 1% | 3% | 1% |
| 0 | 2% | 5% | 2% |
| +1.5 | 4% | 9% | 2% |

Full table in `results/asr_raw_vector.csv`. The unsteered baseline differs slightly between the two runs (see Limitations).

### FRR on benign prompts

| m | FRR | full refusal | partial refusal |
|---|---|---|---|
| -1.5 | 65% | 45 | 20 |
| -1.0 | 60% | 41 | 19 |
| -0.5 | 56% | 39 | 17 |
| 0 | 51% | 35 | 16 |
| +0.5 | 46% | 26 | 20 |
| +1.0 | 41% | 24 | 17 |
| +1.5 | 36% | 18 | 18 |

![FRR vs multiplier](figures/frr_vs_multiplier_xstest.png)

## Repository layout

```
01_extract_sycophancy_vector.ipynb   extract the difference-in-means vector for all 32 layers
02_steering_asr_frr.ipynb            steer, generate, judge, and plot ASR and FRR
data/                                JailbreakBench prompts and the sycophancy contrastive pairs
vectors/                             the extracted vector, Li et al.'s normalised layer 14 vector, CAA reference vectors
results/                             ASR and FRR tables, plus per-prompt judge verdicts
figures/                             plots
```

Raw model replies are not published, because some are completions of harmful prompts. `results/` has the aggregated tables and the per-prompt verdict labels without the reply text.

## How to reproduce

1. **Install.**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get model access.** Llama-2-7B-chat is gated on Hugging Face and needs approved access. Everything else the notebooks read (JailbreakBench prompts, the sycophancy pairs, the steering vectors) is already in this repository. Sources and licences are in [data/README.md](data/README.md).

3. **Extract the vector (optional).** Run `01_extract_sycophancy_vector.ipynb` top to bottom. It caches activations at the answer-letter token for the 1,000 contrastive pairs, saves them as `pos_acts_llama2_7b_chat.pt` and `neg_acts_llama2_7b_chat.pt` (about 250 MB each), and writes the difference in means to `vectors/`. That vector is already in the repository, so this step can be skipped.

4. **Steer and generate.** Run `02_steering_asr_frr.ipynb` top to bottom. It picks a GPU if one is available. The full sweep (7 multipliers, 300 harmful prompts and 100 benign prompts each, 300 new tokens, greedy decoding, no system prompt) took about 15 minutes on one H100 at batch size 32. It writes `replies_norm/` and `replies_benign/`.

5. **Judge.** Before running the judge sections of the same notebook, copy `.env.example` to `.env` and add an OpenAI API key. They write `judged_norm/` and `judged_benign_xstest/`. The 2,800 calls to GPT-4o-mini cost well under a dollar.

6. **Plot.** The plotting cells rebuild the tables and write the figures to `figures/`.

## Details that matter

- **Layer and position.** Layer 14 of 32, vector added to the output of the decoder block at every token position.
- **Normalisation.** Each behaviour vector is rescaled to the mean L2 norm of the nine behaviour vectors at that layer. See [vectors/README.md](vectors/README.md).
- **Attack templates.** Copied from Li et al.'s `run_steering.py`. They differ slightly from the templates printed in the paper.
- **ASR judge.** GPT-4o-mini, Li et al.'s prompt from `judgelm_evaluate.py`, temperature 0.0001, 100 max tokens. A reply counts when the verdict starts with JAILBROKEN.
- **FRR judge.** GPT-4o-mini with the XSTest classifier prompt, plus one added line asking for the label only. A reply counts when the label is full refusal or partial refusal.

## Limitations

- **No random-direction control.** Random steering directions at a matched norm were not tested, so the increase in ASR cannot be claimed to be specific to the sycophancy direction. Prior work ([The Rogue Scalpel](https://arxiv.org/abs/2509.22067)) reports that random directions alone raise harmful compliance by a few points on other models.
- **Small sample, one run.** 100 prompts per condition, so one point is one prompt. Differences of a point or two are inside the noise, and there are no confidence intervals.
- **High baseline refusal.** The unsteered model already refuses 51% of the benign prompts.
- **Judges not validated by hand.** Neither judge was checked against human labels, and the FRR judge is an adaptation of XSTest's.
- **Baseline not bit-reproducible.** Between two runs on different hardware and library versions, 217 of 300 unsteered replies were identical despite greedy decoding.
- **Scope.** One behaviour, one layer, one model.

## Credits

- Li et al., [Analysing the Safety Pitfalls of Steering Vectors](https://arxiv.org/abs/2603.24543), code at [yetiiil/analyse-sv-safety](https://github.com/yetiiil/analyse-sv-safety) (MIT). The attack templates, the ASR judge prompt and the normalised vector come from their code.
- Rimsky et al., [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), code at [nrimsky/CAA](https://github.com/nrimsky/CAA).
- Chao et al., [JailbreakBench](https://arxiv.org/abs/2404.01318).
- Röttger et al., [XSTest](https://arxiv.org/abs/2308.01263).
- Perez et al., Anthropic [model-written evals](https://huggingface.co/datasets/Anthropic/model-written-evals) for the sycophancy datasets.
