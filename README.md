<div align="center">

# LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling

</div>

[![arXiv](https://img.shields.io/badge/arXiv-2604.11748-b31b1b.svg)](https://arxiv.org/abs/2604.11748) [![HuggingFace](https://img.shields.io/badge/HuggingFace-langflow--owt-yellow)](https://huggingface.co/Continuous-Rivals-Discrete/langflow-owt) [![HuggingFace](https://img.shields.io/badge/HuggingFace-langflow--lm1b-yellow)](https://huggingface.co/Continuous-Rivals-Discrete/langflow-lm1b) [![Blog](https://img.shields.io/badge/Blog-Post-blue)](https://caradryanl.github.io/blog/2026/langflow/)

By Yuxin Chen*, Chumeng Liang*, Hangke Sui*, Ruihan Guo, Chaoran Cheng, Jiaxuan You, Ge Liu.

**The first continuous diffusion language model that rivals discrete counterparts on standard language modeling benchmarks like LM1B and OpenWebText**.

<div align="center">
  <img src="assets/pipe.png" alt="LangFlow pipeline" width="65%">
  <img src="assets/table_eval.png" alt="Evaluation results" width="30%">
</div>

## TODO

- [x] Inference code
- [x] PPL evaluation code
- [x] OpenWebText checkpoint on [HuggingFace](https://huggingface.co/Continuous-Rivals-Discrete/langflow-owt)
- [x] LM1B checkpoint on [HuggingFace](https://huggingface.co/Continuous-Rivals-Discrete/langflow-lm1b)
- [ ] Training code (after paper acceptance)  
- [ ] All trainable checkpoints (after paper acceptance)

## Quick Start

### 1. Install dependencies

```bash
conda create -n langflow python=3.12
conda activate langflow
# Driver 550 on the training host supports CUDA 12.x, not CUDA 13.x.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Download the checkpoint

Download only the safetensors weights file from HuggingFace — no need to clone the HF repo:

```bash
# Using huggingface-hub CLI
hf download Continuous-Rivals-Discrete/langflow-owt model.safetensors --local-dir ./checkpoints
```

### 3. Run inference

```bash
python inference.py \
    --checkpoint ./checkpoints/model.safetensors \
    --num_samples 5 \
    --batch_size 1 \
    --num_steps 1024 \
    --seq_length 1024 \
    --seed 42 \
    --output samples.txt
```

## Train on TinyStoriesV2

`train.py` implements Algorithm 1 from the paper: CE denoising on the VP
embedding path, a learned Gumbel log-NSR scheduler and its detached fitting
loss, 25% self-conditioning, low-discrepancy noise sampling, the 5K-step bias
warmup, and EMA. The first launch converts raw documents to reusable packed
memmaps beside the dataset. Documents are split before packing, tokenized
without truncation, and terminated with EOS. Following the released dataloader,
each batch of 1,000 documents is concatenated, chunked into `L-2` token payloads
(dropping its incomplete tail), and wrapped in BOS/EOS. Every position is valid.

Sampling always denoises a full fixed-length block. EOS is learned by the same
token CE as the rest of the vocabulary: it does not stop diffusion, select a
length, or get forced into the output. Decoding for Gen-PPL retains special
tokens and does not truncate at EOS.

The expected dataset layout is:

```text
datasets/TinyStoriesV2/
├── TinyStoriesV2-GPT4-train.txt
└── TinyStoriesV2-GPT4-valid.txt
```

Single GPU (a plain `python` launch is supported):

```bash
python train.py \
    --data_dir ../../datasets/TinyStoriesV2 \
    --output_dir results/langflow-tinystories \
    --micro_batch_size 2 \
    --global_batch_size 512 \
    --epochs 100
```

Four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --data_dir ../../datasets/TinyStoriesV2 \
    --output_dir results/langflow-tinystories \
    --micro_batch_size 2 \
    --global_batch_size 512 \
    --epochs 100
```

`global_batch_size` includes every GPU and gradient-accumulation step. It must
be divisible by `world_size * micro_batch_size`; accumulation is derived
automatically. To cap a run by optimizer updates, pass `--max_steps` (the paper
uses 1M updates for LM1B/OWT). Resume with either
`--resume path/to/step-XXXXXXX.pt` or `--resume_latest`. Checkpoints contain the
model, EMA, optimizer, LR scheduler, scaler, data position, and each rank's RNG
state. Exact resume checks the data fingerprint, model configuration, world
size and training hyperparameters. Legacy story-aligned checkpoints/caches
cannot be resumed under the new protocol. At
the end of training, Hugging Face-compatible weights are exported to
`final-model/` and `final-ema/`.

To checkpoint and exit a running job cleanly after its next complete optimizer
update, create a marker in that run directory:

```bash
touch results/langflow-tinystories/SAVE_AND_STOP
```

Rank zero writes the checkpoint atomically and all distributed ranks exit
together. Remove that run's marker before resuming with
`--resume /path/to/step-XXXXXXX.pt`.

### ROCStories adaptation

From the MSEFlow root, run `bash experiments/langflow-rocstories-130m.sh`.
It reads the original CSV, writes a separate `paper-packed-gpt2-l64` cache and
uses CUDA devices `4,5` by default (override `CUDA_VISIBLE_DEVICES` if needed).
The experiment uses length 64, global batch 128, micro-batch 32 per GPU, 20
epochs, 400-update LR warmup to a constant `1e-4`, 500-update bias warmup and
EMA 0.999. The shorter bias warmup is an adaptation to this small dataset, not
the paper's original 5,000-update setting. Learned Gumbel noise scheduling is
unchanged. This matches the paper's method, not its LM1B/OWT training scale.

`bash eval/eval-langflow-rocstories.sh` evaluates the latest numeric-step
checkpoint with EMA, packed references, full-block Gen-PPL and auxiliary MAUVE.
Old story-aligned/length-aware scores are not directly comparable.
Validation CE uses a fixed initial Gumbel distribution and fixed noise, with
and without self-conditioning; it is a diagnostic, not a likelihood estimate.

### Code and regression tests

`train.py` is a thin CLI; `training/data.py`, `objective.py`, `runtime.py` and
`trainer.py` own packing, the loss, distributed/checkpoint state and the loop.
Tests create isolated temporary fixtures and do not touch production runs:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_training.py --gpus 4,5
```

The smoke test checks bitwise-equal uninterrupted/resumed model and EMA weights,
checkpoint loading, packed references, fixed-length generation and the Gen-PPL
pipeline with a small local scorer. It is not a generation-quality benchmark.

## Citation

```
@article{chen2026langflow,
  title={LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling},
  author={Chen, Yuxin and Liang, Chumeng and Sui, Hangke and Guo, Ruihan and Cheng, Chaoran and You, Jiaxuan and Liu, Ge},
  journal={arXiv preprint arXiv:2604.11748},
  year={2026}
}
```
