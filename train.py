"""Train LangFlow with the paper objective.

The implementation follows Algorithm 1 of arXiv:2604.11748:

* sample log-NSR ``gamma`` from a learnable Gumbel proposal;
* perturb normalized token embeddings along the VP path;
* train the denoiser with token-level cross entropy;
* fit the proposal to the detached per-example CE curve; and
* use one detached model prediction for self-conditioning with probability .25.

TinyStories uses fixed-length packed blocks. ROCStories directly reuses the
story-aligned cache produced by ``train_ar.py``: each example contains all five
sentences, is truncated to reserve a terminal EOS, and is padded to the fixed
sequence length. Padding is excluded from attention and loss. Regular Python
and ``torchrun`` launches are supported.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from langflow import LangFlow, LangFlowConfig


LOGGER = logging.getLogger("langflow.train")
PROJECT_DIR = Path(__file__).resolve().parent
_PREPROCESS_TOKENIZER = None
_PREPROCESS_DTYPE: np.dtype | None = None


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1:
        dist.init_process_group(backend=backend)
    return DistributedContext(rank, local_rank, world_size, device)


def barrier(ctx: DistributedContext) -> None:
    if ctx.world_size > 1:
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def save_and_stop_requested(output_dir: Path, ctx: DistributedContext) -> bool:
    """Synchronize a file-based graceful-stop request across every rank."""
    requested = torch.tensor(
        int(ctx.is_main and (output_dir / "SAVE_AND_STOP").is_file()),
        dtype=torch.int32,
        device=ctx.device,
    )
    if ctx.world_size > 1:
        dist.broadcast(requested, src=0)
    return bool(requested.item())


def configure_logging(output_dir: Path, ctx: DistributedContext) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "train.log"))
    logging.basicConfig(
        level=logging.INFO if ctx.is_main else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int, rank: int) -> None:
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def _iter_text_chunks(path: Path, target_chars: int = 1 << 20) -> Iterator[str]:
    """Yield newline-aligned chunks without loading a multi-GB file in RAM."""
    lines: list[str] = []
    size = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            lines.append(line)
            size += len(line)
            if size >= target_chars:
                yield "".join(lines)
                lines.clear()
                size = 0
    if lines:
        yield "".join(lines)


def _init_preprocess_worker(tokenizer_path: str, dtype_name: str) -> None:
    """Load one tokenizer per worker and avoid nested tokenizer thread pools."""
    global _PREPROCESS_TOKENIZER, _PREPROCESS_DTYPE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _PREPROCESS_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True,
    )
    _PREPROCESS_TOKENIZER.model_max_length = int(1e30)
    _PREPROCESS_DTYPE = np.dtype(dtype_name)


def _tokenize_chunk(text: str) -> np.ndarray:
    if _PREPROCESS_TOKENIZER is None or _PREPROCESS_DTYPE is None:
        raise RuntimeError("Preprocessing worker was not initialized")
    token_ids = _PREPROCESS_TOKENIZER(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return np.asarray(token_ids, dtype=_PREPROCESS_DTYPE)


def _tokenize_file(
    raw_path: Path,
    output_path: Path,
    tokenizer,
    dtype: np.dtype,
    preprocess_workers: int,
    worker_tokenizer_path: Path,
) -> int:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    token_count = 0
    pool = None
    try:
        with temp_path.open("wb") as output:
            if preprocess_workers == 1:
                tokenized_chunks = (
                    np.asarray(
                        tokenizer(
                            text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                        )["input_ids"],
                        dtype=dtype,
                    )
                    for text in _iter_text_chunks(raw_path)
                )
            else:
                LOGGER.info(
                    "Tokenizing %s with %d worker processes",
                    raw_path.name,
                    preprocess_workers,
                )
                pool = mp.get_context("spawn").Pool(
                    processes=preprocess_workers,
                    initializer=_init_preprocess_worker,
                    initargs=(str(worker_tokenizer_path), dtype.name),
                )
                tokenized_chunks = pool.imap(_tokenize_chunk, _iter_text_chunks(raw_path), chunksize=1)

            for chunk_index, values in enumerate(tokenized_chunks, start=1):
                values.tofile(output)
                token_count += int(values.size)
                if chunk_index % 25 == 0:
                    LOGGER.info("Tokenized %s: %s tokens", raw_path.name, f"{token_count:,}")
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        os.replace(temp_path, output_path)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return token_count


def prepare_tinystories(
    data_dir: Path,
    cache_dir: Path,
    tokenizer_name: str,
    preprocess_workers: int,
    ctx: DistributedContext,
) -> dict:
    """Build reusable train/validation token memmaps on rank zero."""
    metadata_path = cache_dir / "metadata.json"
    raw_paths = {
        "train": data_dir / "TinyStoriesV2-GPT4-train.txt",
        "validation": data_dir / "TinyStoriesV2-GPT4-valid.txt",
    }

    if ctx.is_main:
        missing = [str(path) for path in raw_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing TinyStoriesV2 files: " + ", ".join(missing))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # These chunks are only being tokenized into a flat cache; they are
        # never passed to GPT-2 as one sequence.  Disable the tokenizer's
        # model-context warning and split into training blocks below instead.
        tokenizer.model_max_length = int(1e30)
        dtype = np.dtype("uint16" if len(tokenizer) <= np.iinfo(np.uint16).max else "uint32")
        fingerprint = {
            "tokenizer": tokenizer_name,
            "vocab_size": len(tokenizer),
            "dtype": dtype.name,
            "sources": {
                split: {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for split, path in raw_paths.items()
            },
        }

        cached = None
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
        bins_exist = all((cache_dir / f"{split}.bin").is_file() for split in raw_paths)

        if cached is None or cached.get("fingerprint") != fingerprint or not bins_exist:
            cache_dir.mkdir(parents=True, exist_ok=True)
            worker_tokenizer_path = cache_dir / "tokenizer"
            tokenizer.save_pretrained(worker_tokenizer_path)
            LOGGER.info("Preparing TinyStoriesV2 token cache in %s", cache_dir)
            counts = {
                split: _tokenize_file(
                    path,
                    cache_dir / f"{split}.bin",
                    tokenizer,
                    dtype,
                    preprocess_workers,
                    worker_tokenizer_path,
                )
                for split, path in raw_paths.items()
            }
            metadata = {"fingerprint": fingerprint, "token_counts": counts}
            temp_metadata = metadata_path.with_suffix(".json.tmp")
            with temp_metadata.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            os.replace(temp_metadata, metadata_path)
        else:
            LOGGER.info("Using existing token cache in %s", cache_dir)

    barrier(ctx)
    if not metadata_path.is_file():
        raise RuntimeError(f"Token-cache preparation failed: {metadata_path} was not created")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_rocstories(
    sequence_cache_dir: Path,
    cache_dir: Path,
    tokenizer_name: str,
    ctx: DistributedContext,
) -> dict:
    """Pack the exact train/validation split made by ``train_ar.py``.

    The source arrays contain one padded 64-token row per story plus its true
    length.  Only true tokens (including the terminal EOS) are flattened, so
    bidirectional LangFlow attention never sees padding.  Story boundaries are
    retained by EOS tokens.
    """
    metadata_path = cache_dir / "metadata.json"
    source_metadata_path = sequence_cache_dir / "metadata.json"
    source_paths = {
        split: (
            sequence_cache_dir / f"{split}-tokens.npy",
            sequence_cache_dir / f"{split}-lengths.npy",
        )
        for split in ("train", "validation")
    }

    if ctx.is_main:
        required = [source_metadata_path]
        for token_path, length_path in source_paths.values():
            required.extend((token_path, length_path))
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing ROCStories token-cache files: " + ", ".join(missing))

        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        source_fingerprint = source_metadata.get("fingerprint", {})
        if source_fingerprint.get("tokenizer") != tokenizer_name:
            raise ValueError(
                f"ROCStories cache tokenizer {source_fingerprint.get('tokenizer')!r} "
                f"does not match requested tokenizer {tokenizer_name!r}"
            )
        train_tokens = np.load(source_paths["train"][0], mmap_mode="r")
        dtype = train_tokens.dtype
        fingerprint = {
            "dataset": "rocstories",
            "tokenizer": tokenizer_name,
            "vocab_size": int(source_fingerprint.get("vocab_size", 50_257)),
            "dtype": dtype.name,
            "source_fingerprint": source_fingerprint,
            "sources": {
                split: {
                    "tokens": {
                        "path": str(token_path.resolve()),
                        "size": token_path.stat().st_size,
                        "mtime_ns": token_path.stat().st_mtime_ns,
                    },
                    "lengths": {
                        "path": str(length_path.resolve()),
                        "size": length_path.stat().st_size,
                        "mtime_ns": length_path.stat().st_mtime_ns,
                    },
                }
                for split, (token_path, length_path) in source_paths.items()
            },
        }
        cached = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
        bins_exist = all((cache_dir / f"{split}.bin").is_file() for split in source_paths)
        if cached is None or cached.get("fingerprint") != fingerprint or not bins_exist:
            cache_dir.mkdir(parents=True, exist_ok=True)
            counts: dict[str, int] = {}
            story_counts: dict[str, int] = {}
            LOGGER.info("Packing ROCStories token cache from %s into %s", sequence_cache_dir, cache_dir)
            for split, (token_path, length_path) in source_paths.items():
                tokens = np.load(token_path, mmap_mode="r")
                lengths = np.load(length_path, mmap_mode="r")
                temporary = (cache_dir / f"{split}.bin").with_suffix(".bin.tmp")
                token_count = 0
                with temporary.open("wb") as output:
                    for start in range(0, len(lengths), 4096):
                        stop = min(start + 4096, len(lengths))
                        packed = np.concatenate([
                            np.asarray(tokens[index, : int(lengths[index])], dtype=dtype)
                            for index in range(start, stop)
                        ])
                        packed.tofile(output)
                        token_count += int(packed.size)
                os.replace(temporary, cache_dir / f"{split}.bin")
                counts[split] = token_count
                story_counts[split] = len(lengths)
                LOGGER.info(
                    "Packed %s ROCStories: %s stories, %s tokens",
                    split, f"{len(lengths):,}", f"{token_count:,}",
                )
            metadata = {
                "fingerprint": fingerprint,
                "token_counts": counts,
                "story_counts": story_counts,
            }
            temporary_metadata = metadata_path.with_suffix(".json.tmp")
            temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            os.replace(temporary_metadata, metadata_path)
        else:
            LOGGER.info("Using existing packed ROCStories cache in %s", cache_dir)

    barrier(ctx)
    if not metadata_path.is_file():
        raise RuntimeError(f"ROCStories cache preparation failed: {metadata_path} was not created")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class TokenBlockDataset(Dataset[torch.Tensor]):
    """Non-overlapping fixed-length views over a flat token memmap."""

    def __init__(self, path: Path, dtype: str, sequence_length: int):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.sequence_length = sequence_length
        self._tokens: np.memmap | None = None
        token_count = path.stat().st_size // self.dtype.itemsize
        self.block_count = token_count // sequence_length
        if self.block_count == 0:
            raise ValueError(f"{path} has fewer than {sequence_length} tokens")

    def __len__(self) -> int:
        return self.block_count

    def _memmap(self) -> np.memmap:
        # Each DataLoader worker opens its own read-only mapping.
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._tokens

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.sequence_length
        values = np.asarray(self._memmap()[start : start + self.sequence_length], dtype=np.int64)
        return torch.from_numpy(values)


class ROCStoriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """One five-sentence ROCStories example per row with a valid-token mask."""

    def __init__(self, cache_dir: Path, split: str, sequence_length: int):
        token_path = cache_dir / f"{split}-tokens.npy"
        length_path = cache_dir / f"{split}-lengths.npy"
        if not token_path.is_file() or not length_path.is_file():
            raise FileNotFoundError(
                f"Missing story-aligned ROCStories cache for {split}: "
                f"{token_path}, {length_path}"
            )
        self.tokens = np.load(token_path, mmap_mode="r")
        self.lengths = np.load(length_path, mmap_mode="r")
        self.sequence_length = sequence_length
        if self.tokens.ndim != 2 or self.tokens.shape[0] != len(self.lengths):
            raise ValueError(
                f"Invalid ROCStories cache shapes: tokens={self.tokens.shape}, "
                f"lengths={self.lengths.shape}"
            )
        if self.tokens.shape[1] != sequence_length:
            raise ValueError(
                f"ROCStories cache width {self.tokens.shape[1]} does not match "
                f"sequence_length={sequence_length}; rebuild it with train_ar.py"
            )

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(self.lengths[index])
        if not 1 <= length <= self.sequence_length:
            raise ValueError(f"Invalid ROCStories length {length} at index {index}")
        input_ids = torch.from_numpy(
            np.array(self.tokens[index], dtype=np.int64, copy=True)
        )
        loss_mask = torch.arange(self.sequence_length) < length
        return input_ids, loss_mask


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token IDs and a valid-token mask for packed or padded datasets."""
    if isinstance(batch, (tuple, list)):
        input_ids, loss_mask = batch
        return input_ids, loss_mask.to(torch.bool)
    return batch, torch.ones_like(batch, dtype=torch.bool)


class LangFlowTrainingObjective(nn.Module):
    def __init__(self, model: LangFlow, self_condition_probability: float):
        super().__init__()
        self.model = model
        self.self_condition_probability = self_condition_probability

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        quantiles: torch.Tensor,
        bias_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Algorithm 1 explicitly stops gradients through sampled gamma.  This
        # lets CE train only the denoiser while the fitting loss trains only the
        # proposal parameters.
        gamma = self.model.proposal(quantiles).detach()
        clean_embeddings = self.model._embed_tokens(input_ids)
        noisy_embeddings = self.model._forward_diffusion(clean_embeddings, gamma)

        self_condition = None
        if self.model.config.self_conditioning and self.self_condition_probability > 0:
            active = torch.rand(input_ids.shape[0], device=input_ids.device) < self.self_condition_probability
            active_indices = active.nonzero(as_tuple=True)[0]
            self_condition = torch.zeros_like(clean_embeddings)
            if active_indices.numel() > 0:
                with torch.no_grad():
                    first_logits = self.model(
                        noisy_embeds=noisy_embeddings.index_select(0, active_indices),
                        timesteps=gamma.index_select(0, active_indices),
                        x_self_cond=None,
                        attention_mask=loss_mask.index_select(0, active_indices),
                        bias_scale=bias_scale,
                        return_dict=False,
                    )
                    first_probabilities = F.softmax(first_logits.float(), dim=-1)
                    prediction = self.model._embed_tokens(first_probabilities)
                self_condition.index_copy_(0, active_indices, prediction.to(self_condition.dtype))

        logits = self.model(
            noisy_embeds=noisy_embeddings,
            timesteps=gamma,
            x_self_cond=self_condition,
            attention_mask=loss_mask,
            bias_scale=bias_scale,
            return_dict=False,
        )
        token_loss = F.cross_entropy(
            logits.float().flatten(0, 1),
            input_ids.flatten(),
            reduction="none",
        ).view_as(input_ids)
        float_mask = loss_mask.to(token_loss.dtype)
        per_example_ce = (
            (token_loss * float_mask).sum(dim=1)
            / float_mask.sum(dim=1).clamp_min(1)
        )
        ce_loss = per_example_ce.mean()

        proposal = self.model.proposal
        proposal_scale = proposal.scale.clamp_min(1e-4)
        entropy_curve = proposal.entropy * torch.exp(
            -torch.exp(-(gamma - proposal.loc) / proposal_scale)
        )
        scheduler_loss = F.mse_loss(entropy_curve, per_example_ce.detach())
        return ce_loss + scheduler_loss, ce_loss.detach(), scheduler_loss.detach()


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def sample_low_discrepancy_quantiles(
    global_batch_size: int,
    ctx: DistributedContext,
) -> torch.Tensor:
    """Return this rank's stratified, shuffled Gumbel quantiles."""
    if ctx.is_main:
        jitter = torch.rand(global_batch_size, device=ctx.device)
        quantiles = (torch.arange(global_batch_size, device=ctx.device) + jitter) / global_batch_size
        quantiles = quantiles[torch.randperm(global_batch_size, device=ctx.device)]
        quantiles.clamp_(1e-5, 1.0 - 1e-5)
    else:
        quantiles = torch.empty(global_batch_size, device=ctx.device)
    if ctx.world_size > 1:
        dist.broadcast(quantiles, src=0)
    per_rank = global_batch_size // ctx.world_size
    start = ctx.rank * per_rank
    return quantiles[start : start + per_rank]


def precision_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    # The objective performs a no-grad self-conditioning forward before its
    # grad-enabled denoising forward.  With autocast's weight cache enabled,
    # low-precision Linear weights first cast under no_grad are cached without
    # autograd history and then reused by the denoising pass.  That silently
    # leaves the Linear parameters without gradients and breaks DDP reduction.
    return torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False)


def reduce_metrics(values: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    values = values.detach().float()
    if ctx.world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= ctx.world_size
    return values


def unwrap_model(training_model: nn.Module) -> LangFlow:
    objective = training_model.module if isinstance(training_model, DDP) else training_model
    return objective.model


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    training_model: nn.Module,
    ema: LangFlow,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    batch_in_epoch: int,
) -> None:
    model = unwrap_model(training_model)
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "args": vars(args),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    model: LangFlow,
    ema: LangFlow,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler,
    restore_rng: bool,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint.get("rng")
    if rng and restore_rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(checkpoint["step"]), int(checkpoint["epoch"]), int(checkpoint["batch_in_epoch"])


@torch.no_grad()
def validate(
    model: LangFlow,
    loader: DataLoader,
    ctx: DistributedContext,
    precision: str,
    max_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=ctx.device)
    total_examples = torch.zeros((), device=ctx.device)
    num_batches = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    fork_devices = [ctx.device.index] if ctx.device.type == "cuda" else []
    # Validation must not advance the training RNG stream.  A fixed noise draw
    # and a grid over proposal quantiles make the metric comparable over time.
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(17_029 + ctx.rank)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed(17_029 + ctx.rank)
        for batch_index, batch in enumerate(loader):
            if batch_index >= num_batches:
                break
            input_ids, loss_mask = unpack_batch(batch)
            input_ids = input_ids.to(ctx.device, non_blocking=True)
            loss_mask = loss_mask.to(ctx.device, non_blocking=True)
            sample_indices = (
                (batch_index * loader.batch_size + torch.arange(input_ids.shape[0], device=ctx.device))
                * ctx.world_size
                + ctx.rank
            )
            denominator = max(1, num_batches * loader.batch_size * ctx.world_size)
            quantiles = ((sample_indices + 0.5) / denominator).clamp(1e-5, 1.0 - 1e-5)
            gamma = model.proposal(quantiles).detach()
            with precision_context(precision, ctx.device):
                logits = model(
                    input_ids=input_ids,
                    timesteps=gamma,
                    attention_mask=loss_mask,
                    bias_scale=1.0,
                    return_dict=False,
                )
                token_loss = F.cross_entropy(
                    logits.float().flatten(0, 1),
                    input_ids.flatten(),
                    reduction="none",
                ).view_as(input_ids)
            total_loss += (token_loss * loss_mask).sum()
            total_examples += loss_mask.sum()
    if ctx.world_size > 1:
        dist.all_reduce(total_loss)
        dist.all_reduce(total_examples)
    model.train(was_training)
    return (total_loss / total_examples.clamp_min(1)).item()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR.parent.parent / "datasets/TinyStoriesV2")
    parser.add_argument("--dataset", choices=("tinystories", "rocstories"), default="tinystories")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "results/langflow-tinystories")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Log directory; defaults to output_dir when omitted",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "langflow/config.json")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--init_from", default=None, help="Optional Hugging Face model name or local model directory")
    parser.add_argument("--resume", type=Path, default=None, help="A .pt checkpoint, or 'latest' via --resume_latest")
    parser.add_argument("--resume_latest", action="store_true")

    parser.add_argument("--sequence_length", type=int, default=None, help="Defaults to model_length in the config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optimizer-step limit; negative means epoch-based")
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=8,
        help="Processes used by rank zero to build the token cache; one disables multiprocessing",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=2500)
    parser.add_argument("--bias_warmup_steps", type=int, default=5000)
    parser.add_argument("--self_condition_probability", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--validate_every", type=int, default=5000)
    parser.add_argument("--validation_batches", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument(
        "--save_every_fraction",
        type=float,
        default=0.0,
        help="Save and validate this many epochs apart; zero disables fractional checkpoints",
    )
    parser.add_argument("--keep_checkpoints", type=int, default=0, help="Keep newest N; zero keeps every checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ctx = setup_distributed()
    configure_logging(args.log_dir or args.output_dir, ctx)
    seed_everything(args.seed, ctx.rank)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.global_batch_size % (ctx.world_size * args.micro_batch_size) != 0:
        raise ValueError("global_batch_size must be divisible by world_size * micro_batch_size")
    accumulation_steps = args.global_batch_size // (ctx.world_size * args.micro_batch_size)
    if not 0 <= args.self_condition_probability <= 1:
        raise ValueError("self_condition_probability must be between zero and one")
    if args.preprocess_workers < 1:
        raise ValueError("preprocess_workers must be at least one")
    if args.save_every_fraction < 0 or args.save_every_fraction > 1:
        raise ValueError("save_every_fraction must be in [0, 1]")

    if args.dataset == "rocstories":
        cache_dir = args.data_dir
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing story-aligned ROCStories metadata: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fingerprint = metadata.get("fingerprint", {})
        if fingerprint.get("tokenizer") != args.tokenizer:
            raise ValueError(
                f"ROCStories cache tokenizer {fingerprint.get('tokenizer')!r} "
                f"does not match requested tokenizer {args.tokenizer!r}"
            )
    else:
        cache_dir = args.cache_dir or args.data_dir / f"tokenized-{args.tokenizer.replace('/', '--')}"
        metadata = prepare_tinystories(args.data_dir, cache_dir, args.tokenizer, args.preprocess_workers, ctx)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if args.init_from:
        model = LangFlow.from_pretrained(args.init_from)
        config = model.config
    else:
        config = LangFlowConfig.from_pretrained(args.config)
        config.vocab_size = len(tokenizer)
        model = LangFlow(config)
    sequence_length = args.sequence_length or config.model_length
    config.model_length = sequence_length
    if config.vocab_size != len(tokenizer):
        raise ValueError(f"Model vocabulary ({config.vocab_size}) != tokenizer vocabulary ({len(tokenizer)})")

    if args.dataset == "rocstories":
        cache_length = int(metadata.get("fingerprint", {}).get("max_length", -1))
        if cache_length != sequence_length:
            raise ValueError(
                f"ROCStories cache max_length={cache_length} does not match "
                f"sequence_length={sequence_length}"
            )
        train_dataset = ROCStoriesDataset(cache_dir, "train", sequence_length)
        valid_dataset = ROCStoriesDataset(cache_dir, "validation", sequence_length)
    else:
        dtype = metadata["fingerprint"]["dtype"]
        train_dataset = TokenBlockDataset(cache_dir / "train.bin", dtype, sequence_length)
        valid_dataset = TokenBlockDataset(cache_dir / "validation.bin", dtype, sequence_length)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=args.seed, drop_last=True
    )
    if ctx.world_size > 1:
        valid_sampler = DistributedSampler(
            valid_dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=False, drop_last=False
        )
    else:
        valid_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=ctx.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + ctx.rank),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.micro_batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=ctx.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed + 10_000 + ctx.rank),
    )
    complete_updates_per_epoch = len(train_loader) // accumulation_steps
    if complete_updates_per_epoch < 1:
        raise ValueError("Not enough training batches for one optimizer update")
    fractional_save_updates: list[int] = []
    if args.save_every_fraction > 0:
        fractional_save_updates = sorted({
            min(complete_updates_per_epoch, max(1, math.ceil(complete_updates_per_epoch * fraction)))
            for fraction in np.arange(
                args.save_every_fraction,
                1.0 + args.save_every_fraction / 2,
                args.save_every_fraction,
            )
        })

    model.to(ctx.device)
    ema = copy.deepcopy(model).to(ctx.device).eval()
    ema.requires_grad_(False)
    objective = LangFlowTrainingObjective(model, args.self_condition_probability).to(ctx.device)
    if ctx.world_size > 1:
        objective = DDP(
            objective,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            # Every trainable parameter participates in the main denoising
            # pass.  Unused-parameter discovery is both unnecessary and, in
            # PyTorch 2.6, can corrupt reducer state when combined with the
            # accumulated no_sync passes below.
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and ctx.device.type == "cuda")

    start_step = 0
    start_epoch = 0
    start_batch = 0
    resume_path = args.resume
    checkpoint_dir = args.output_dir / "checkpoints"
    if args.resume_latest:
        candidates = sorted(checkpoint_dir.glob("step-*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        resume_path = candidates[-1]
    if resume_path is not None:
        start_step, start_epoch, start_batch = restore_checkpoint(
            resume_path, model, ema, optimizer, lr_scheduler, scaler,
            restore_rng=ctx.world_size == 1,
        )
        if ctx.world_size > 1:
            # A rank-zero checkpoint contains only rank zero's RNG state.  Do
            # not clone it onto every worker, which would correlate diffusion
            # noise and self-conditioning masks across GPUs.
            seed_everything(args.seed + start_step * ctx.world_size, ctx.rank)
        LOGGER.info(
            "Resumed %s at step %s, epoch %d, batch %d",
            resume_path,
            f"{start_step:,}",
            start_epoch,
            start_batch,
        )
    else:
        update_ema(ema, model, decay=0.0)

    if ctx.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)
        LOGGER.info(
            "Model parameters: %s | train examples: %s | validation examples: %s",
            f"{sum(parameter.numel() for parameter in model.parameters()):,}",
            f"{len(train_dataset):,}",
            f"{len(valid_dataset):,}",
        )
        LOGGER.info(
            "world_size=%d micro_batch=%d accumulation=%d global_batch=%d sequence_length=%d",
            ctx.world_size,
            args.micro_batch_size,
            accumulation_steps,
            args.global_batch_size,
            sequence_length,
        )
        LOGGER.info(
            "epochs=%d updates_per_epoch=%d total_updates=%d fractional_save_updates=%s",
            args.epochs,
            complete_updates_per_epoch,
            complete_updates_per_epoch * args.epochs,
            fractional_save_updates,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    running = torch.zeros(3, device=ctx.device)
    running_updates = 0
    log_started = time.monotonic()
    quantiles_for_rank: torch.Tensor | None = None
    accumulation_index = 0
    stop_training = False
    save_and_exit = False
    last_epoch = start_epoch
    last_batch = start_batch
    first_logged_step = start_step + 1

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        train_sampler.set_epoch(epoch)
        update_in_epoch = start_batch // accumulation_steps if epoch == start_epoch else 0
        for batch_index, batch in enumerate(train_loader):
            if update_in_epoch >= complete_updates_per_epoch:
                break
            if args.max_steps > 0 and step >= args.max_steps:
                stop_training = True
                break
            if epoch == start_epoch and batch_index < start_batch:
                continue
            last_batch = batch_index + 1
            input_ids, loss_mask = unpack_batch(batch)
            if quantiles_for_rank is None:
                quantiles_for_rank = sample_low_discrepancy_quantiles(args.global_batch_size, ctx)
            q_start = accumulation_index * args.micro_batch_size
            quantiles = quantiles_for_rank[q_start : q_start + args.micro_batch_size]
            input_ids = input_ids.to(ctx.device, non_blocking=True)
            loss_mask = loss_mask.to(ctx.device, non_blocking=True)
            bias_scale = min(1.0, step / max(1, args.bias_warmup_steps))
            final_micro_step = accumulation_index + 1 == accumulation_steps
            sync_context = contextlib.nullcontext()
            if isinstance(objective, DDP) and not final_micro_step:
                sync_context = objective.no_sync()

            # DDP requires both forward and backward to stay inside no_sync;
            # leaving the context before backward corrupts reducer state on
            # the next accumulated micro-batch.
            with sync_context:
                with precision_context(args.precision, ctx.device):
                    total_loss, ce_loss, scheduler_loss = objective(
                        input_ids, loss_mask, quantiles, bias_scale
                    )
                    scaled_loss = total_loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
            running += torch.stack((total_loss.detach(), ce_loss, scheduler_loss)) / accumulation_steps
            accumulation_index += 1

            if not final_micro_step:
                continue

            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.proposal.scale.clamp_(min=1e-4)
                model.proposal.entropy.clamp_(min=1e-4)
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            step += 1
            update_in_epoch += 1
            update_ema(ema, model, args.ema_decay)
            accumulation_index = 0
            quantiles_for_rank = None
            running_updates += 1

            if step == first_logged_step or step % args.log_every == 0:
                metrics = reduce_metrics(running / max(1, running_updates), ctx)
                elapsed = time.monotonic() - log_started
                if ctx.is_main:
                    LOGGER.info(
                        "step=%07d epoch=%d loss=%.4f ce=%.4f scheduler=%.4f lr=%.3e "
                        "steps/s=%.3f peak_memory_gib=%.2f",
                        step,
                        epoch,
                        metrics[0].item(),
                        metrics[1].item(),
                        metrics[2].item(),
                        optimizer.param_groups[0]["lr"],
                        running_updates / max(elapsed, 1e-9),
                        torch.cuda.max_memory_allocated(ctx.device) / 2**30
                        if ctx.device.type == "cuda" else 0.0,
                    )
                    LOGGER.info(
                        "proposal loc=%.4f scale=%.4f entropy=%.4f bias_scale=%.4f",
                        model.proposal.loc.item(),
                        model.proposal.scale.item(),
                        model.proposal.entropy.item(),
                        bias_scale,
                    )
                running.zero_()
                running_updates = 0
                log_started = time.monotonic()

            if save_and_stop_requested(args.output_dir, ctx):
                if ctx.is_main:
                    checkpoint_path = checkpoint_dir / f"step-{step:07d}.pt"
                    save_checkpoint(
                        checkpoint_path, objective, ema, optimizer, lr_scheduler, scaler,
                        args, step, epoch, batch_index + 1,
                    )
                    LOGGER.info("Saved %s", checkpoint_path)
                    LOGGER.info("Saved current progress and stopped by SAVE_AND_STOP request")
                barrier(ctx)
                save_and_exit = True
                stop_training = True
                break

            if args.validate_every > 0 and step % args.validate_every == 0:
                validation_ce = validate(ema, valid_loader, ctx, args.precision, args.validation_batches)
                if ctx.is_main:
                    LOGGER.info("step=%07d validation_ce=%.4f", step, validation_ce)

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                if ctx.is_main:
                    checkpoint_path = checkpoint_dir / f"step-{step:07d}.pt"
                    save_checkpoint(
                        checkpoint_path, objective, ema, optimizer, lr_scheduler, scaler,
                        args, step, epoch, batch_index + 1,
                    )
                    LOGGER.info("Saved %s", checkpoint_path)
                    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
                    if args.keep_checkpoints > 0:
                        for old_checkpoint in checkpoints[:-args.keep_checkpoints]:
                            old_checkpoint.unlink()
                barrier(ctx)

            if update_in_epoch in fractional_save_updates:
                validation_ce = validate(ema, valid_loader, ctx, args.precision, args.validation_batches)
                epoch_progress = epoch + update_in_epoch / complete_updates_per_epoch
                if ctx.is_main:
                    LOGGER.info(
                        "epoch=%.3f step=%07d validation_ce=%.4f",
                        epoch_progress, step, validation_ce,
                    )
                    label = f"{epoch_progress:06.1f}"
                    checkpoint_path = checkpoint_dir / f"epoch-{label}-step-{step:07d}.pt"
                    save_checkpoint(
                        checkpoint_path, objective, ema, optimizer, lr_scheduler, scaler,
                        args, step, epoch, batch_index + 1,
                    )
                    LOGGER.info("Saved %s", checkpoint_path)
                barrier(ctx)

            if args.max_steps > 0 and step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break
        start_batch = 0
        # Do not carry a partial global batch over an epoch boundary.
        if accumulation_index:
            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0
            quantiles_for_rank = None
        last_batch = 0
        last_epoch = epoch + 1

    barrier(ctx)
    if ctx.is_main and not save_and_exit:
        final_checkpoint = checkpoint_dir / f"step-{step:07d}-final.pt"
        save_checkpoint(
            final_checkpoint, objective, ema, optimizer, lr_scheduler, scaler,
            args, step, last_epoch, last_batch,
        )
        model_dir = args.output_dir / "final-model"
        ema_dir = args.output_dir / "final-ema"
        model.save_pretrained(model_dir, safe_serialization=True)
        ema.save_pretrained(ema_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        tokenizer.save_pretrained(ema_dir)
        LOGGER.info("Training complete at step %s; exported model and EMA", f"{step:,}")

    barrier(ctx)
    if ctx.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
