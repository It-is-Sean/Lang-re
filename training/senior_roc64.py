"""Read the fixed Spring-2016 + Winter-2017 split used by Project/ccflow/data.py.

This is intentionally NOT paper-packed-v1: one story per row, no BOS, append
EOS then truncate/right-pad to 64, with every position participating in CE.
The source cache is read-only. Hashes bind a run to the exact ordered tensors.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

PROTOCOL = "senior-roc64-full-v1"
EXPECTED = {"rows": 98167, "unique": 98161, "train": 93253, "val": 4908,
            "seq_len": 64, "eos_id": 50256, "split_seed": 42, "val_fraction": 0.05,
            "csv_sha256": "58bb77a008fcb282fd64b5c97a9e68ad65aab2cec56272c215e9f9024e9aea3e"}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_metadata(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    for key, expected in EXPECTED.items():
        if meta.get(key) != expected:
            raise ValueError(f"Senior ROC64 metadata mismatch: {key}={meta.get(key)!r}, expected {expected!r}")
    return {
        "fingerprint": {
            "protocol": PROTOCOL, "dataset": "rocstories-spring2016+winter2017",
            "tokenizer": "gpt2", "vocab_size": 50257, "sequence_length": 64,
            "eos_token_id": 50256, "split_seed": 42, "validation_fraction": 0.05,
            "source_csv_sha256": meta["csv_sha256"],
            "files": {name: sha256(directory / name) for name in ("train.pt", "val.pt", "meta.json")},
            "encoding": "gpt2-no-bos-append-eos-truncate-64-right-pad-eos-all-position-loss",
        },
        "splits": {split: {"documents": meta[key], "blocks": meta[key]}
                   for split, key in (("train", "train"), ("validation", "val"))},
        "source_metadata": meta,
    }


class StoryDataset(Dataset):
    def __init__(self, directory, split):
        filename = {"train": "train.pt", "validation": "val.pt"}[split]
        self.tokens = torch.load(Path(directory) / filename, map_location="cpu", weights_only=True)
        expected = EXPECTED["train" if split == "train" else "val"]
        if self.tokens.dtype != torch.int64 or self.tokens.shape != (expected, 64):
            raise ValueError(f"Invalid full ROC64 tensor: {filename}, {self.tokens.shape}, {self.tokens.dtype}")
        if self.tokens.min() < 0 or self.tokens.max() >= 50257:
            raise ValueError("Invalid GPT-2 token IDs")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]


def quantiles(global_batch_size, ctx):
    # Project/baselines/roc64/train_langflow.py: train_q; no random permutation.
    q = ((torch.rand(global_batch_size, device=ctx.device) / global_batch_size
          + torch.arange(global_batch_size, device=ctx.device) / global_batch_size) % 1).clamp(1e-5, 1-1e-5)
    return q.chunk(ctx.world_size)[ctx.rank]


def ema_decay(max_decay, completed_updates):
    return min(max_decay, (1 + completed_updates) / (10 + completed_updates))


@torch.no_grad()
def validate(model, dataset, ctx, batch_size, probability, bias_scale):
    """Same learned-proposal CE+scheduler objective and fixed seeds as Project.

    Contiguous validation shards, no duplicates; all validation stories used.
    Unlike the historical packed diagnostic, this follows the learned proposal.
    """
    from .objective import LangFlowTrainingObjective
    was_training = model.training
    model.eval()
    objective = LangFlowTrainingObjective(model, probability)
    total = torch.zeros(3, dtype=torch.float64, device=ctx.device)
    indices = torch.arange(len(dataset)).tensor_split(ctx.world_size)[ctx.rank]
    devices = [ctx.device.index] if ctx.device.type == "cuda" else []
    for chunk in indices.split(batch_size):
        if not len(chunk):
            continue
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(4321 + int(chunk[0]))
            x = dataset.tokens[chunk].to(ctx.device)
            n = len(chunk)
            q = ((torch.rand(n, device=ctx.device) / n
                  + torch.arange(n, device=ctx.device) / n) % 1).clamp(1e-5, 1-1e-5)
            _, ce, sched = objective(x, q, bias_scale)
            total += torch.stack((ce.double() * n, sched.double() * n, total.new_tensor(n)))
    if ctx.world_size > 1:
        dist.all_reduce(total)
    model.train(was_training)
    return {"ce": (total[0] / total[2]).item(), "sched": (total[1] / total[2]).item(),
            "loss": ((total[0] + total[1]) / total[2]).item()}
