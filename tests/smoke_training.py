"""End-to-end CPU/GPU or torchrun smoke test, including exact checkpoint resume.

Run from the LangFlow root: python tests/smoke_training.py [--gpus 4,5]
All fixtures, logs and checkpoints are created in a fresh temporary directory.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="")
    parser.add_argument("--keep", action="store_true", help="Keep temporary fixtures for inspecting failures")
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="langflow-paper-smoke-")
    if args.keep:
        temporary._finalizer.detach()
    work = Path(temporary.name)
    print(f"SMOKE_DIR={work}", flush=True)
    tokenizer_dir = work / "tokenizer"
    vocabulary = {word: i for i, word in enumerate(["[UNK]", "[BOS]", "[EOS]", "[PAD]", *[f"w{i}" for i in range(28)]])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.save_pretrained(tokenizer_dir)
    source = work / "stories.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f"sentence{i}" for i in range(1, 6)])
        writer.writeheader()
        for row in range(80):
            writer.writerow({f"sentence{i}": f"w{row % 28} w{i} w{(row + i) % 28}" for i in range(1, 6)})
    config = work / "config.json"
    config.write_text(json.dumps(dict(vocab_size=32, hidden_size=16, cond_dim=8, n_blocks=2,
                                     n_heads=2, model_length=8, dropout=0.1)))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus, "OMP_NUM_THREADS": "2",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
    world = len(args.gpus.split(",")) if args.gpus else 1
    launcher = [sys.executable]
    if world > 1:
        launcher += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"]
    common = [str(ROOT / "train.py"), "--dataset", "rocstories", "--data_dir", str(source),
              "--cache_dir", str(work / "cache"), "--tokenizer", str(tokenizer_dir), "--config", str(config),
              "--epochs", "2", "--global_batch_size", "8", "--micro_batch_size", "2",
              "--num_workers", "0", "--preprocess_workers", "2", "--precision", "bf16" if args.gpus else "fp32",
              "--validation_fraction", "0.2", "--warmup_steps", "1", "--bias_warmup_steps", "2",
              "--log_every", "1", "--validate_every", "2", "--validation_batches", "2", "--checkpoint_every", "2"]

    def train(name, steps, *extra):
        subprocess.run(launcher + common + ["--output_dir", str(work / name), "--log_dir", str(work / "logs" / name),
                                           "--max_steps", str(steps), *extra], env=env, cwd=ROOT, check=True)

    train("continuous", 4)
    train("resumed", 2)
    train("resumed", 4, "--resume_latest")
    load = lambda name: torch.load(work / name / "checkpoints/step-0000004-final.pt", map_location="cpu", weights_only=False)
    expected, actual = load("continuous"), load("resumed")
    for key in ("model", "ema"):
        differences = {name: (value - actual[key][name]).abs().max().item()
                       for name, value in expected[key].items() if not torch.equal(value, actual[key][name])}
        if differences:
            raise AssertionError(f"Resume changed {key}: {differences}")
    assert len(actual["rng_by_rank"]) == world
    print("PASS: uninterrupted and resumed model/EMA are bitwise equal", flush=True)

    # Exercise the same checkpoint loader, references, generation, decoding and
    # scorer used by eval, without downloading GPT-2 Large or running MAUVE.
    sys.path.insert(0, str(ROOT))
    from eval_langflow import load_generator, select_references, generate_samples, compute_generated_ppl
    from transformers import GPT2Config, GPT2LMHeadModel
    device = torch.device("cpu")
    model, _ = load_generator(work / "resumed/checkpoints", "ema", device)
    indices, references = select_references(work / "cache", "validation", 4, 42)
    samples = generate_samples(model, 4, 8, 4, 2, 42, device, "fp32")
    assert len(set(indices)) == 4 and all(len(ids) == 8 for ids in samples + references)
    text = tokenizer.batch_decode(samples, skip_special_tokens=False)
    scorer = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1,
                                       n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=3)).eval()
    metrics = compute_generated_ppl(scorer, tokenizer, text, 2, 32, device)
    assert metrics["gen_ppl"] > 0 and metrics["num_scored_tokens"] > 0
    print("PASS: packed-reference generation and Gen-PPL pipeline", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
