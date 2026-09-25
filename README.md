<div align="center">

# BLUR

### A Bi-Level Optimization Approach for LLM Unlearning

[![arXiv](https://img.shields.io/badge/arXiv-2506.08164-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.08164)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/OptimAI-Lab/blur)
[![EACL 2026](https://img.shields.io/badge/EACL-2026-8A2BE2)](#citation)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-MUSE%20%7C%20WMDP%20%7C%20TOFU-2E86C1)](#running-experiments)

**Forget first, then preserve utility: LLM unlearning as a bi-level problem.**

[Abstract](#abstract) •
[Repository Layout](#repository-layout) •
[Installation](#installation) •
[Running Experiments](#running-experiments) •
[Models](#released-models) •
[Citation](#citation)

</div>

---

## Abstract

Enabling large language models (LLMs) to unlearn knowledge and capabilities acquired during training has proven vital for ensuring compliance with data regulations and promoting ethical practices in generative AI. Although there are growing interests in developing various unlearning algorithms, it remains unclear how to best formulate the unlearning problem. The most popular formulation uses a weighted sum of forget and retain loss, but it often leads to performance degradation due to the inherent trade-off between forget and retain losses.

In this work, we argue that it is important to model the **hierarchical structure** of the unlearning problem, where the forget problem (which *unlearns* certain knowledge and/or capabilities) takes priority over the retain problem (which preserves model utility). This hierarchical structure naturally leads to a **bi-level optimization** formulation where the **lower-level** objective focuses on minimizing the forget loss, while the **upper-level** objective aims to maintain the model's utility. Based on this new formulation, we propose a novel algorithm, termed **Bi-Level UnleaRning (BLUR)**, which not only possesses strong theoretical guarantees but more importantly, delivers superior performance. In particular, our extensive experiments demonstrate that BLUR consistently outperforms all the state-of-the-art algorithms across various unlearning tasks, models, and metrics.

## Highlights

- **Hierarchical formulation**: forgetting is the lower-level problem; utility preservation is the upper-level problem, instead of a single weighted sum of losses.
- **Three benchmarks**: code and configs for **MUSE** (News & Books), **WMDP** (bio & cyber hazardous knowledge), and **TOFU** (fictitious author QA).
- **Drop-in baselines**: MUSE and TOFU folders also contain standard baselines (GA, GradDiff, NPO, SimNPO, DPO, RMU, Task Vector, etc.) for comparison.
- **Released checkpoints** on Hugging Face.

## Repository Layout

```text
BLURLLMUnlearning/
├── MUSE/                         # MUSE benchmark (News / Books)
│   ├── environment.yml           # conda env: muse_env
│   ├── load_data.py              # downloads MUSE data into MUSE/data/
│   ├── eval.py                   # VerbMem / KnowMem / PrivLeak evaluation
│   ├── news_eval.sh, books_eval.sh
│   └── baselines/
│       ├── unlearn.py            # entry point (BLUR = --algo BLO_forget_lower_*)
│       ├── unlearn_news.sh, unlearn_books.sh
│       └── baselines/iterative.py  # BLUR bi-level update is implemented here
├── WMDP/
│   └── rmu/
│       ├── unlearn_bi.py         # BLUR on top of RMU
│       └── unlearn.py            # original RMU baseline
└── TOFU/                         # built on open-unlearning (Hydra configs)
    ├── configs/trainer/BLURNPO.yaml
    ├── src/trainer/unlearn/grad_diff.py   # BLUR_NPO trainer
    └── scripts/tofu_unlearn.sh
```

## Installation

Each benchmark uses its own environment. Clone the repo first:

```bash
git clone https://github.com/iirmz1997-rgb/BLURLLMUnlearning.git
cd BLURLLMUnlearning
```

### MUSE environment

```bash
cd MUSE
conda env create -f environment.yml
conda activate muse_env
```

### WMDP environment

```bash
conda create -n wmdp_env python=3.9.21 -y
conda activate wmdp_env

conda install pytorch=2.1.1 torchvision=0.16.1 torchaudio=2.1.1 \
  pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install datasets==3.2.0 wandb==0.19.2 transformers==4.37.2 \
  sentencepiece==0.1.99 sentence-transformers==2.5.1

pip install terminaltables==3.1.10 sacrebleu==2.4.0 rouge-score==0.1.2 \
  matplotlib==3.8.3 seaborn==0.13.2 scikit-learn==1.4.0

# for evaluation
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

### TOFU environment

The TOFU code is based on [open-unlearning](https://github.com/locuslab/open-unlearning) and requires **Python ≥ 3.11**. The model configs use `flash_attention_2`, so `flash-attn` is needed.

```bash
cd TOFU
conda create -n tofu_env python=3.11 -y
conda activate tofu_env
pip install .
pip install --no-build-isolation flash-attn==2.6.3

# download eval logs of the retain models and auxiliary data
python setup_data.py
```

## Running Experiments

### MUSE

**1. Download the data** (from `MUSE/`):

```bash
cd MUSE
python load_data.py      # writes MUSE/data/{news,books}/...
```

**2. Run BLUR** (from `MUSE/baselines/`). The ready-made scripts are `unlearn_news.sh` and `unlearn_books.sh`; the News example is:

```bash
cd baselines

CORPUS="news"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="muse-bench/MUSE-News_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"
MAX_LEN=2048
EPOCHS=10
LR='2.5e-5'                  # Books uses 1e-5
PER_DEVICE_BATCH_SIZE=4      # with 8 GPUs
GAMA=1.0
OUT_DIR="./out_dir"          # change to your own path

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for algo in 'BLO_forget_lower_npo_gdr'; do
  python unlearn.py \
    --algo "$algo" \
    --model_dir "$TARGET_DIR" \
    --tokenizer_dir "$LLAMA_DIR" \
    --data_file "$FORGET" \
    --retain_data_file "$RETAIN" \
    --out_dir "$OUT_DIR/$CORPUS/$algo" \
    --max_len "$MAX_LEN" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gama "$GAMA"
done
```

> `--algo` values containing `BLO_forget_lower` run BLUR (forget loss at the lower level); `--gama` sets the weight γ in the bi-level update.

**3. Evaluate** (from `MUSE/`), pointing to a saved checkpoint:

```bash
cd ..
python eval.py \
  --model_dirs "./baselines/out_dir/news/BLO_forget_lower_npo_gdr/checkpoint-102" \
  --names "checkpoint-102" \
  --corpus news \
  --out_file "./result/out.csv"
```

Metrics: `verbmem_f`, `knowmem_f`, `knowmem_r`, `privleak`.

### WMDP

Run from the `WMDP/` folder. Forget corpora are read from `files/data/<name>.jsonl` at the repository root (`../files/data/` relative to `WMDP/`), so place the WMDP corpora there first; the names passed to `--forget_corpora` must match the file names. The WMDP bio forget corpus is gated and must be requested from the [WMDP authors](https://www.wmdp.ai/). The default base model is `HuggingFaceH4/zephyr-7b-beta`.

```bash
cd WMDP
CUDA_VISIBLE_DEVICES=0 python3 -m rmu.unlearn_bi \
  --max_num_batches 150 \
  --batch_size 4 \
  --retain_corpora wikitext,wikitext \
  --forget_corpora bio_remove_dataset,cyber-forget-corpus \
  --steering_coeffs 6.5,6.5 \
  --alpha 800,800 \
  --lr 5e-5 \
  --seed 0 \
  --output_dir models/bi_unlearn
```

Evaluate with `lm-evaluation-harness`:

```bash
lm-eval --model hf \
  --model_args pretrained=models/bi_unlearn \
  --tasks wmdp,mmlu \
  --batch_size 16
```

### TOFU

Run from the `TOFU/` folder. The script unlearns `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` on the `forget05` split with the `BLURNPO` trainer and then evaluates it:

```bash
cd TOFU
bash scripts/tofu_unlearn.sh
```

Or run a single job directly:

```bash
accelerate launch --config_file configs/accelerate/default_config.yaml \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=BLURNPO \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget05 retain_split=retain95 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json \
  task_name=tofu_Llama-3.2-1B-Instruct_forget05_BLURNPO
```

Results are saved to `TOFU/saves/unlearn/<task_name>/`. See `TOFU/docs/` for more on the Hydra configs.

## Released Models

Unlearned checkpoints are available in the Hugging Face collection:

- [🤗 OptimAI-Lab/BLUR](https://huggingface.co/collections/OptimAI-Lab/blur)

## Citation

If you find BLUR useful in your research, please cite:

```bibtex
@inproceedings{reisizadeh2026blur,
  title={Blur: A bi-level optimization approach for llm unlearning},
  author={Reisizadeh, Hadi and Jia, Jinghan and Bu, Zhiqi and Vinzamuri, Bhanukiran and Ramakrishna, Anil and Chang, Kai-Wei and Cevher, Volkan and Liu, Sijia and Hong, Mingyi},
  booktitle={Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7043--7058},
  year={2026}
}
```

## Acknowledgements

This code builds on the following open-source projects:

- [MUSE](https://github.com/jaechan-repo/muse_bench): MUSE benchmark and baselines
- [WMDP / RMU](https://github.com/centerforaisafety/wmdp): WMDP benchmark and RMU
- [open-unlearning](https://github.com/locuslab/open-unlearning): TOFU framework

Please also follow the licenses and terms of these projects, the benchmark datasets, and the base models (e.g., Llama).
