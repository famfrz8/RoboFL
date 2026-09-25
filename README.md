# InternVLA-A1 — RoboTwin LoRA-MoE Release

This repository contains the code needed to **train and evaluate InternVLA-A1-3B on the
RoboTwin 2.0 benchmark**, including the full **LoRA / LoRA-MoE** and **federated learning (FL)**
pipeline.

InternVLA-A1 unifies scene **understanding**, visual **generation**, and **action** execution
in a single three-expert architecture:

- **Understanding expert** — Qwen3-VL vision-language backbone.
- **Generation expert** — Cosmos tokenizer based visual foresight.
- **Action expert** — diffusion-style action head conditioned on state and time.

On top of the base model this release adds:

- **LoRA fine-tuning** for RoboTwin.
- **LoRA-MoE**: per-client LoRA experts combined by a learned router (soft / top-k /
  noisy-top-k), with load-balancing (aux) loss.
- **Federated learning** over RoboTwin tasks: `fedavg`, `fedprox`, `scaffold`, task/episode/
  Dirichlet non-IID partitioning, and router-weighted expert aggregation.
- Optional **TCR** (prototype-guided regularization) and **Affordance** heads.

> **Note on licensing:** this code is a derivative of InternVLA-A1 and is distributed under
> **CC BY-NC-SA 4.0**. See [License and attribution](#license-and-attribution) below.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/lerobot/` | Core library (policies, datasets, federated utilities, training scripts). |
| `src/lerobot/policies/InternVLA_A1_3B/` | InternVLA-A1-3B policy, transforms, Cosmos tokenizer, `transformers_replace` patches. |
| `src/lerobot/scripts/lerobot_train.py` | Standard single-model training entry point. |
| `src/lerobot/scripts/lerobot_fl_robotwin*.py` | Federated RoboTwin training entry points (LoRA / LoRA-MoE / router-weighted / FARD). |
| `launch/` | Launch scripts for fine-tuning and federated training. |
| `evaluation/RoboTwin/` | RoboTwin 2.0 evaluation and inference. |
| `tutorials/` | Installation and fine-tuning guides. |
| `util_scripts/` | Dataset statistics and utility scripts. |
| `third_party/RoboTwin/` | RoboTwin 2.0 simulator, added as a git submodule (upstream is not vendored). |
| `third_party/RoboTwin_custom.patch` | Custom changes applied on top of the pinned RoboTwin revision. |

---

## Installation

Tested with **Python 3.10**, **CUDA 12.8**, and **PyTorch 2.7.1**. See
[`tutorials/installation.md`](tutorials/installation.md) for the full guide.

```bash
conda create -y -n internvla_a1 python=3.10
conda activate internvla_a1

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install -e .
```

### Patch HuggingFace Transformers

The model relies on patched `transformers` modules. After installing dependencies:

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/

cp -r src/lerobot/policies/pi0/transformers_replace/models          ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models ${TRANSFORMERS_DIR}
```

Re-run this step after any reinstall of `transformers`.

---

## Data

Download the preprocessed RoboTwin dataset in LeRobot v3.0 format and link the cache:

```bash
hf download hxma/RoboTwin-LeRobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robotwin

ln -s ${HF_HOME}/lerobot data   # if not already linked
```

Compute the delta-action normalization statistics used by training:

```bash
DATASET_REPO_ID="$(
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-agilex*" 2>/dev/null \
  | while read -r d; do
        if [[ -d "$d/meta" && -d "$d/videos" ]]; then echo "${d#data/}"; fi
    done | sort -u
)"

python util_scripts/compute_norm_stats_multi.py \
  --action_mode delta --chunk_size 50 --repo_id "${DATASET_REPO_ID}"
```

Set `HF_HOME`, `HF_TOKEN`, and (optionally) `WANDB_API_KEY` before launching. Launch
scripts no longer hardcode credentials or absolute paths; `PRETRAINED_PATH`,
`HF_HOME`, `CONDA_ROOT`, and output directories can all be overridden with environment
variables.

See [`tutorials/finetune_internvla_a1_with_robotwin.md`](tutorials/finetune_internvla_a1_with_robotwin.md)
for the complete RoboTwin fine-tuning walkthrough.

---

## Training

### Fine-tune on RoboTwin (single model, LoRA)

```bash
bash launch/internvla_a1_3b_finetune_robotwin.sh
```

### Federated learning on RoboTwin

```bash
# FedAvg + LoRA
bash launch/internvla_a1_3b_fl_robotwin.sh

# FedAvg + LoRA-MoE
bash launch/internvla_a1_3b_fl_robotwin_moe.sh

# LoRA-MoE with full options (router, shared-A/expert-B, visual token pruning)
bash launch/internvla_a1_3b_fl_lora_moe.sh

# LoRA-MoE with FARD/PCEA affinity and router-weighted expert aggregation
bash launch/fardpcea.sh

# LoRA-MoE + TCR
bash launch/internvla_a1_3b_fl_robotwin_moe_tcr.sh
```

Common overrides:

```bash
NUM_GPUS=4 FL_NUM_CLIENTS=8 FL_NUM_ROUNDS=50 \
  bash launch/internvla_a1_3b_fl_robotwin_moe.sh
```

Key federated parameters: `FL_NUM_CLIENTS`, `FL_LOCAL_STEPS`, `FL_NUM_ROUNDS`,
`FL_PARTITION` (`by_task` / `by_episode` / `dirichlet`), `FL_AGGREGATION`
(`fedavg` / `fedprox` / `scaffold`), `MOE_NUM_EXPERTS`, `MOE_K_EXPERTS`,
`MOE_STEPS`, `LORA_RANK`. See the header of each launch script for details.

---

## Evaluation

The RoboTwin 2.0 simulator is tracked as a git submodule pinned to a known revision, plus a
small patch with the custom changes required by this pipeline. The simulator itself (and its
~17 GB of assets) is **not** vendored here; it is fetched from its upstream repository
(see its own MIT license).

```bash
git submodule update --init third_party/RoboTwin
cd third_party/RoboTwin
git apply ../RoboTwin_custom.patch
cp ../../evaluation/RoboTwin/requirements.txt script/requirements.txt
bash script/_install.sh
bash script/_download_assets.sh
cd ../..
```

Then run the evaluation scripts:

```bash
bash evaluation/RoboTwin/eval.sh          # full/fine-tuned checkpoint
bash evaluation/RoboTwin/eval_lora.sh     # LoRA checkpoint
bash evaluation/RoboTwin/eval_lora_moe.sh # LoRA-MoE checkpoint
```

Set `PRETRAINED_CKPT`, `LORA_CKPT`, or `LORAMOE_CKPT` (and `BASE_MODEL_PATH`) to point at
your checkpoints. See [`evaluation/RoboTwin/README.md`](evaluation/RoboTwin/README.md).

---

## License and attribution

This project is a derivative work based on **InternVLA-A1** and is released under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)** license. You must give appropriate credit, provide a link to the
license, and distribute your contributions under the same license. Commercial use is not
permitted. See [`LICENSE`](LICENSE) for the full notice.

This codebase also builds on open-source projects including
[LeRobot](https://github.com/huggingface/lerobot),
[openpi](https://github.com/Physical-Intelligence/openpi),
[InternVL](https://github.com/OpenGVLab/InternVL),
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), and
[NVIDIA Cosmos](https://github.com/nvidia-cosmos). Their respective licenses apply to
those components.

```bibtex
@article{internvla_a1_contributors_2026,
  title={InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation},
  author={InternVLA-A1 contributors},
  journal={arXiv preprint arXiv:2601.02456},
  year={2026}
}
```
