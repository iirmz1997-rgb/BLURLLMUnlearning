<div align="center">

# BLUR

### A Bi-Level Optimization Approach for LLM Unlearning

[![Paper](https://img.shields.io/badge/arXiv-2506.08164-B31B1B)](https://arxiv.org/abs/2506.08164)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow?logo=huggingface)](https://huggingface.co/collections/OptimAI-Lab/blur)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)
[![EACL 2026](https://img.shields.io/badge/EACL-2026-8A2BE2)](#citation)

**A research framework for targeted LLM unlearning with improved forgetting, retention, and controllability.**

</div>

## Abstract

Enabling large language models (LLMs) to unlearn knowledge and capabilities acquired during training is vital for regulatory compliance, privacy, and responsible AI. However, removing targeted information while preserving general model utility remains a challenging optimization problem.

**BLUR** formulates LLM unlearning as a **bi-level optimization** problem: an inner optimization focuses on forgetting the target data, while an outer optimization helps preserve model behavior on knowledge that should be retained. The repository provides implementations and experiment configurations for evaluating BLUR across established unlearning benchmarks.

## Highlights

- **Bi-level optimization** for balancing forgetting and retention.
- **Benchmark support** for both MUSE and WMDP experiments.
- **Reproducible configurations** for model unlearning experiments.
- **Released checkpoints** through the OptimAI-Lab Hugging Face collection.
- **Research-ready evaluation** for studying effectiveness and side effects.

## Installation

### MUSE environment

```bash
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

pip install terminaltables==3.1.10 sacrebleu==2.4.0 \
  rouge-score==0.1.2 matplotlib==3.8.3 seaborn==0.13.2 \
  scikit-learn==1.4.0

git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## Running Experiments

### MUSE

The following example runs BLUR on the MUSE News corpus. Adjust paths, hyperparameters, and GPU settings for your environment.

```bash
CORPUS="news"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="muse-bench/MUSE-News_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"
MAX_LEN=2048
EPOCHS=10
LR='2.5e-5'
PER_DEVICE_BATCH_SIZE=4
GAMA=1.0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for algo in 'BLO_forget_lower_npo_gdr'; do
  python unlearn.py \
    --algo "$algo" \
    --model_dir "$TARGET_DIR" \
    --tokenizer_dir "$LLAMA_DIR" \
    --data_file "$FORGET" \
    --retain_data_file "$RETAIN" \
    --out_dir "/home/mhong/shared/hadir/out_dir/$CORPUS/$algo" \
    --max_len "$MAX_LEN" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gama "$GAMA"
done
```

### WMDP

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m rmu.unlearn_bi \
  --max_num_batches 150 \
  --batch_size=4 \
  --retain_corpora wikitext,wikitext \
  --forget_corpora bio_remove_dataset,cyber-forget-corpus \
  --steering_coeffs 6.5,6.5 \
  --alpha 800,800 \
  --lr 5e-5 \
  --seed 0 \
  --output_dir models/bi_unlearn
```

## Released Models

Pretrained and unlearned models are available in the Hugging Face collection:

- [🤗 OptimAI-Lab/BLUR](https://huggingface.co/collections/OptimAI-Lab/blur)

## Evaluation

When reporting results, evaluate both the success of forgetting and the preservation of unrelated capabilities. We recommend documenting:

| Dimension | Question |
| --- | --- |
| Forgetting effectiveness | Is the target knowledge or capability removed? |
| Retention utility | Does the model preserve unrelated behavior? |
| Robustness | Does forgetting persist across prompts and paraphrases? |
| Generalization | Does unlearning extend beyond observed examples? |
| Efficiency | What are the compute, memory, and time requirements? |

For reproducibility, report the base model, dataset versions, preprocessing, random seed, hyperparameters, hardware, and evaluation protocol.

## Citation

If BLUR is useful in your research, please cite:

```bibtex
@inproceedings{reisizadeh2026blur,
  title={Blur: A bi-level optimization approach for llm unlearning},
  author={Reisizadeh, Hadi and Jia, Jinghan and Bu, Zhiqi and Vinzamuri, Bhanukiran and Ramakrishna, Anil and Chang, Kai-Wei and Cevher, Volkan and Liu, Sijia and Hong, Mingyi},
  booktitle={Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7043--7058},
  year={2026}
}
```

## Responsible Use

Unlearning behavior can depend on the base model, data, prompts, optimization settings, and evaluation procedure. A successful benchmark score should not be interpreted as proof of irreversible removal in every context. Use this code responsibly and report limitations alongside results.

## Contributing

Contributions, replications, and discussion are welcome. Please include:

1. A clear description of the motivation and change.
2. Reproducible commands or configuration files.
3. Forgetting, retention, and efficiency results where applicable.
4. Relevant documentation and tests.

## License

Please review the repository for the applicable license and usage terms.
