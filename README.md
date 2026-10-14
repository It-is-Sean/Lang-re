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
warmup, and EMA. The first launch converts the raw text files to reusable token
memmaps beside the dataset.

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
model, EMA, optimizer, LR scheduler, scaler, data position, and RNG state. At
the end of training, Hugging Face-compatible weights are exported to
`final-model/` and `final-ema/`.

To checkpoint and exit a running job cleanly after its next complete optimizer
update, create a marker in that run directory:

```bash
touch results/langflow-tinystories-*/SAVE_AND_STOP
```

Rank zero writes the checkpoint atomically and all distributed ranks exit
together. Resume it with `--resume /path/to/step-XXXXXXX.pt`.

Useful development smoke test:

```bash
python train.py \
    --sequence_length 32 \
    --global_batch_size 2 \
    --micro_batch_size 1 \
    --max_steps 2 \
    --num_workers 0 \
    --precision fp32
```

## Citation

```
@article{chen2026langflow,
  title={LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling},
  author={Chen, Yuxin and Liang, Chumeng and Sui, Hangke and Guo, Ruihan and Cheng, Chaoran and You, Jiaxuan and Liu, Ge},
  journal={arXiv preprint arXiv:2604.11748},
  year={2026}
}
```
