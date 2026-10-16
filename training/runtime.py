"""Distributed execution, mixed precision, EMA and resumable checkpoints."""

import contextlib
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data import PROTOCOL


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


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
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



def latest_checkpoint(directory):
    candidates = []
    for path in directory.glob("step-*.pt"):
        match = re.fullmatch(r"step-(\d+)(-final)?\.pt", path.name)
        if match:
            candidates.append((int(match[1]), bool(match[2]), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {directory}")
    return max(candidates)[2]


def rng_state(ctx):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(ctx.device) if ctx.device.type == "cuda" else None,
    }


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, args,
                    ctx, fingerprint, step, epoch, batch_in_epoch):
    """All ranks participate so resume restores independent RNG streams."""
    states = [None] * ctx.world_size
    state = rng_state(ctx)
    if ctx.world_size > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if ctx.is_main:
        payload = {
            "data_protocol": PROTOCOL, "data_fingerprint": fingerprint,
            "model_config": model.config.to_dict(), "world_size": ctx.world_size,
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "args": vars(args),
            "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
            "rng_by_rank": states,
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        logging.getLogger("langflow.train").info("Saved %s", path)
    barrier(ctx)


def restore_checkpoint(path, model, ema, optimizer, scheduler, scaler, args, ctx, fingerprint):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("data_protocol") != PROTOCOL:
        raise ValueError("Cannot resume a legacy/story-aligned run under paper packing; start a new run")
    if checkpoint["data_fingerprint"] != fingerprint:
        raise ValueError("Checkpoint and current packed data do not match")
    if checkpoint["model_config"] != model.config.to_dict():
        raise ValueError("Exact resume requires the original model configuration")
    if checkpoint["world_size"] != ctx.world_size:
        raise ValueError("Exact resume requires the original world_size")
    for key in ("global_batch_size", "micro_batch_size", "precision", "seed",
                "learning_rate", "warmup_steps", "bias_warmup_steps",
                "self_condition_probability", "ema_decay", "gradient_clip",
                "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon"):
        if checkpoint["args"][key] != getattr(args, key):
            raise ValueError(f"Exact resume requires unchanged {key}")
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    state = checkpoint["rng_by_rank"][ctx.rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if ctx.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], ctx.device)
    return int(checkpoint["step"]), int(checkpoint["epoch"]), int(checkpoint["batch_in_epoch"])
