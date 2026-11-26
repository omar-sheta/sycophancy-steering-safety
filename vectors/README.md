# Steering vectors

All files are for `meta-llama/Llama-2-7b-chat-hf`, residual stream, output of each decoder block.

| File | Shape | What it is |
|---|---|---|
| `sycophancy_diff_in_means_llama2_7b_chat.pt` | `[32, 4096]`, float32 | Difference in means for all 32 layers, extracted by hand in notebook 01 (activation at the answer-letter token, 1,000 contrastive pairs). Not normalised. Layer 14 norm: 1.94. |
| `caa_reference/vec_layer_{k}_Llama-2-7b-chat-hf.pt` | `[4096]` each, 32 files | Sycophancy vectors released with [CAA](https://github.com/nrimsky/CAA) (`vectors/sycophancy/`, MIT). Used at the end of notebook 01 to check the by-hand vector layer by layer. |
| `sycophancy_layer14_li_normalized.pt` | `[4096]`, float32 | Layer 14 sycophancy vector produced by Li et al.'s code (`generate_vectors.py` then `normalize_vectors.py`). Norm: 7.08. This is the vector used for the reported ASR and FRR results. |

Cosine similarity between layer 14 of the first file and the second file: 0.997.

```python
import torch as t
raw = t.load("vectors/sycophancy_diff_in_means_llama2_7b_chat.pt")[14]   # by hand, raw
li  = t.load("vectors/sycophancy_layer14_li_normalized.pt")              # Li et al., normalised
```

## The normalisation step

Li et al. rescale every behaviour vector at a layer to the mean L2 norm of all nine behaviour vectors at that layer. The step is in their code (`normalize_vectors.py`) and is not described in the paper. For sycophancy at layer 14 it scales the vector by about 3.5 (norm 2.00 to 7.08 for their vector).

To regenerate the normalised vector from their repository (needs a GPU, all nine behaviours):

```bash
git clone https://github.com/yetiiil/analyse-sv-safety
cd analyse-sv-safety
MODEL_FAMILY=llama MODEL_SIZE=7b bash scripts/01_generate_vectors.sh
MODEL_FAMILY=llama MODEL_SIZE=7b bash scripts/02_normalize_vectors.sh
# result: normalized_vectors/sycophancy/vec_layer_14_meta_llama_llama_2_7b_chat_hf.pt
```
