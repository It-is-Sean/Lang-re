"""Train the LangFlow/Duo autoregressive baseline on ROCStories."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from langflow.ar_model import LangFlowAR
from training.data import PROTOCOL, PackedDataset, load_metadata
import experiment_tracking as tracking


LOGGER = logging.getLogger("langflow.train_ar")
PROJECT_DIR = Path(__file__).resolve().parent
_TOKENIZER = None
_MAX_LENGTH = 64


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
        dist.barrier(device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None)


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
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_tokenizer(tokenizer_dir: str, max_length: int) -> None:
    global _TOKENIZER, _MAX_LENGTH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    _MAX_LENGTH = max_length


def _tokenize_story_batch(stories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    # Reserve the final position for EOS, including for truncated stories.
    encoded = _TOKENIZER(
        stories,
        add_special_tokens=False,
        truncation=True,
        max_length=_MAX_LENGTH - 1,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    eos = _TOKENIZER.eos_token_id
    tokens = np.full((len(stories), _MAX_LENGTH), eos, dtype=np.uint16)
    lengths = np.empty(len(stories), dtype=np.uint8)
    for index, ids in enumerate(encoded):
        ids = ids + [eos]
        length = len(ids)
        tokens[index, :length] = ids
        lengths[index] = length
    return tokens, lengths


def prepare_rocstories(
    csv_path: Path,
    cache_dir: Path,
    tokenizer_name: str,
    max_length: int,
    validation_fraction: float,
    seed: int,
    preprocess_workers: int,
    ctx: DistributedContext,
) -> dict:
    metadata_path = cache_dir / "metadata.json"
    source_stat = csv_path.stat()
    fingerprint = {
        "source": str(csv_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "tokenizer": tokenizer_name,
        "max_length": max_length,
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    if ctx.is_main:
        valid_cache = False
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            valid_cache = metadata.get("fingerprint") == fingerprint and all(
                (cache_dir / f"{split}-tokens.npy").is_file()
                and (cache_dir / f"{split}-lengths.npy").is_file()
                for split in ("train", "validation")
            )
        if not valid_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must provide an EOS token")
            tokenizer_dir = cache_dir / "tokenizer"
            tokenizer.save_pretrained(tokenizer_dir)

            stories: list[str] = []
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    stories.append(" ".join(row[f"sentence{i}"].strip() for i in range(1, 6)))
            indices = list(range(len(stories)))
            random.Random(seed).shuffle(indices)
            validation_count = max(1, round(len(stories) * validation_fraction))
            split_indices = {
                "validation": indices[:validation_count],
                "train": indices[validation_count:],
            }

            LOGGER.info(
                "Tokenizing %s ROCStories with %d processes; max_length=%d",
                f"{len(stories):,}", preprocess_workers, max_length,
            )
            batch_size = 512
            batches = [stories[start : start + batch_size] for start in range(0, len(stories), batch_size)]
            if preprocess_workers == 1:
                _init_tokenizer(str(tokenizer_dir), max_length)
                results = map(_tokenize_story_batch, batches)
                encoded_batches = list(results)
            else:
                with mp.get_context("spawn").Pool(
                    preprocess_workers,
                    initializer=_init_tokenizer,
                    initargs=(str(tokenizer_dir), max_length),
                ) as pool:
                    encoded_batches = list(pool.imap(_tokenize_story_batch, batches, chunksize=1))
            all_tokens = np.concatenate([item[0] for item in encoded_batches])
            all_lengths = np.concatenate([item[1] for item in encoded_batches])
            for split, selected in split_indices.items():
                selected_array = np.asarray(selected, dtype=np.int64)
                np.save(cache_dir / f"{split}-tokens.npy", all_tokens[selected_array])
                np.save(cache_dir / f"{split}-lengths.npy", all_lengths[selected_array])
            metadata = {
                "fingerprint": fingerprint,
                "total_stories": len(stories),
                "train_stories": len(split_indices["train"]),
                "validation_stories": len(split_indices["validation"]),
                "truncated_stories": int((all_lengths == max_length).sum()),
                "mean_sequence_length": float(all_lengths.mean()),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            LOGGER.info("Reusing ROCStories token cache at %s", cache_dir)
    barrier(ctx)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class ROCStoriesDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str):
        self.tokens = np.load(cache_dir / f"{split}-tokens.npy", mmap_mode="r")
        self.lengths = np.load(cache_dir / f"{split}-lengths.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(np.array(self.tokens[index], dtype=np.int64)), int(self.lengths[index])


class PackedARDataset(PackedDataset):
    """Reuse the exact diffusion cache; predict positions 1..L-1 from BOS."""

    def __getitem__(self, index):
        return super().__getitem__(index), self.length


def precision_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def causal_loss(logits: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, int]:
    targets = input_ids[:, 1:].clone()
    positions = torch.arange(targets.shape[1], device=targets.device)[None, :]
    targets.masked_fill_(positions >= (lengths[:, None] - 1), -100)
    valid_tokens = int((targets != -100).sum().item())
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss, valid_tokens


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def unwrap(model: nn.Module) -> LangFlowAR:
    return model.module if isinstance(model, DDP) else model


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    training_model: nn.Module,
    ema: LangFlowAR,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    args: argparse.Namespace,
    global_step: int,
    next_epoch: int,
    next_batch: int,
    epoch_progress: float,
) -> None:
    model = unwrap(training_model)
    atomic_save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "epoch_progress": epoch_progress,
            "args": vars(args),
            "data_protocol": args.data_protocol,
            "data_fingerprint": args.data_fingerprint,
        },
        path,
    )


@torch.no_grad()
def validate(
    model: LangFlowAR,
    loader: DataLoader,
    ctx: DistributedContext,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=ctx.device)
    token_count = torch.zeros((), device=ctx.device)
    for input_ids, lengths in loader:
        input_ids = input_ids.to(ctx.device, non_blocking=True)
        lengths = lengths.to(ctx.device, non_blocking=True)
        with precision_context(precision, ctx.device):
            logits = model(input_ids)
        targets = input_ids[:, 1:].clone()
        positions = torch.arange(targets.shape[1], device=targets.device)[None, :]
        targets.masked_fill_(positions >= (lengths[:, None] - 1), -100)
        loss_sum += F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1), ignore_index=-100, reduction="sum",
        )
        token_count += (targets != -100).sum()
    if ctx.world_size > 1:
        dist.all_reduce(loss_sum)
        dist.all_reduce(token_count)
    mean_loss = (loss_sum / token_count.clamp_min(1)).item()
    model.train()
    return mean_loss, math.exp(min(mean_loss, 20.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--data_protocol", choices=("story-aligned", PROTOCOL), default="story-aligned")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--preprocess_workers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_fraction", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=-1, help="Smoke-test/debug limit")
    parser.add_argument("--no_checkpoint", action="store_true", help="Disable checkpoints for smoke tests")
    return parser


@tracking.tracked
def main() -> None:
    args = build_parser().parse_args()
    ctx = setup_distributed()
    if (args.output_dir / "checkpoints").exists() and any((args.output_dir / "checkpoints").iterdir()):
        raise ValueError("Training output already contains checkpoints; choose a new run directory")
    configure_logging(args.log_dir or args.output_dir, ctx)
    seed_everything(args.seed, ctx.rank)
    torch.set_float32_matmul_precision("high")
    if ctx.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.max_length < 2 or args.max_length > 255:
        raise ValueError("max_length must be in [2, 255]")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if args.preprocess_workers < 1:
        raise ValueError("preprocess_workers must be positive")
    if not 0 < args.save_every_fraction <= 1:
        raise ValueError("save_every_fraction must be in (0, 1]")
    denominator = ctx.world_size * args.micro_batch_size
    if args.global_batch_size % denominator:
        raise ValueError("global_batch_size must be divisible by world_size * micro_batch_size")
    accumulation_steps = args.global_batch_size // denominator

    cache_dir = args.cache_dir or args.data.parent / f"tokenized-{args.tokenizer.replace('/', '--')}-l{args.max_length}"
    if args.data_protocol == PROTOCOL:
        if args.cache_dir is None:
            raise ValueError("Packed AR requires an existing --cache_dir")
        metadata = load_metadata(cache_dir)
    else:
        metadata = prepare_rocstories(
            args.data, cache_dir, args.tokenizer, args.max_length,
            args.validation_fraction, args.seed, args.preprocess_workers, ctx,
        )
    tokenizer = AutoTokenizer.from_pretrained(cache_dir / "tokenizer", local_files_only=True)
    args.data_fingerprint = metadata['fingerprint']
    if args.data_protocol == PROTOCOL:
        fp = args.data_fingerprint
        if (fp['sequence_length'] != args.max_length or fp['tokenizer'] != args.tokenizer
                or fp['vocab_size'] != len(tokenizer)
                or fp['bos_token_id'] != tokenizer.bos_token_id
                or fp['eos_token_id'] != tokenizer.eos_token_id):
            raise ValueError("Packed cache length/tokenizer does not match AR configuration")
    dataset_class = PackedARDataset if args.data_protocol == PROTOCOL else ROCStoriesDataset
    train_dataset = dataset_class(cache_dir, "train")
    validation_dataset = dataset_class(cache_dir, "validation")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=ctx.world_size, rank=ctx.rank,
        shuffle=True, seed=args.seed, drop_last=True,
    )
    validation_sampler = range(ctx.rank, len(validation_dataset), ctx.world_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=ctx.device.type == "cuda",
        persistent_workers=args.num_workers > 0, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + ctx.rank),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.micro_batch_size, sampler=validation_sampler,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=ctx.device.type == "cuda", persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed + 10_000 + ctx.rank),
    )

    model = LangFlowAR(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size,
        n_blocks=args.n_blocks, n_heads=args.n_heads, dropout=args.dropout,
    ).to(ctx.device)
    if ctx.world_size > 1:
        training_model: nn.Module = DDP(
            model, device_ids=[ctx.local_rank], find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        training_model = model
    # DDP first synchronizes rank-zero initialization; EMA must start from it.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and ctx.device.type == "cuda")

    complete_updates_per_epoch = len(train_loader) // accumulation_steps
    if complete_updates_per_epoch < 1:
        raise ValueError("Not enough batches for one optimizer update")
    save_updates = sorted({
        min(complete_updates_per_epoch, max(1, math.ceil(complete_updates_per_epoch * fraction)))
        for fraction in np.arange(args.save_every_fraction, 1.0 + args.save_every_fraction / 2, args.save_every_fraction)
    })
    checkpoint_dir = args.output_dir / "checkpoints"
    if ctx.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            **vars(args),
            "data": str(args.data), "cache_dir": str(cache_dir), "output_dir": str(args.output_dir),
            "world_size": ctx.world_size, "gradient_accumulation_steps": accumulation_steps,
            "optimizer_steps_per_epoch": complete_updates_per_epoch,
            "total_optimizer_steps": complete_updates_per_epoch * args.epochs,
            "checkpoint_updates_within_epoch": save_updates,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model_config": model.config_dict(), "dataset": metadata,
        }
        (args.output_dir / "training_config.json").write_text(
            json.dumps(run_config, indent=2, default=str), encoding="utf-8"
        )
        tracking.start(run_config, args.output_dir, model="ar", job_type="train",
                       log_dir=args.log_dir, rank=ctx.rank)
        LOGGER.info(
            "parameters=%s train=%s validation=%s world_size=%d micro_batch=%d accumulation=%d global_batch=%d",
            f"{run_config['model_parameters']:,}", f"{len(train_dataset):,}",
            f"{len(validation_dataset):,}", ctx.world_size,
            args.micro_batch_size, accumulation_steps, args.global_batch_size,
        )
        LOGGER.info(
            "epochs=%d updates_per_epoch=%d total_updates=%d save_updates=%s max_length=%d tokenizer=%s",
            args.epochs, complete_updates_per_epoch,
            complete_updates_per_epoch * args.epochs, save_updates,
            args.max_length, args.tokenizer,
        )

    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    started = time.monotonic()
    stop = False
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        update_in_epoch = 0
        accumulation_index = 0
        running_loss = 0.0
        running_updates = 0
        for batch_index, (input_ids, lengths) in enumerate(train_loader):
            if update_in_epoch >= complete_updates_per_epoch:
                break
            input_ids = input_ids.to(ctx.device, non_blocking=True)
            lengths = lengths.to(ctx.device, non_blocking=True)
            final_micro_step = accumulation_index + 1 == accumulation_steps
            sync_context = contextlib.nullcontext()
            if isinstance(training_model, DDP) and not final_micro_step:
                sync_context = training_model.no_sync()
            with sync_context:
                with precision_context(args.precision, ctx.device):
                    logits = training_model(input_ids)
                    loss, _ = causal_loss(logits, input_ids, lengths)
                    scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
            running_loss += loss.detach().item() / accumulation_steps
            accumulation_index += 1
            if not final_micro_step:
                continue

            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_ema(ema, model, args.ema_decay)
            global_step += 1
            update_in_epoch += 1
            accumulation_index = 0
            running_updates += 1

            if global_step == 1 or global_step % args.log_every == 0:
                metric = torch.tensor(running_loss / max(1, running_updates), device=ctx.device)
                if ctx.world_size > 1:
                    dist.all_reduce(metric)
                    metric /= ctx.world_size
                if ctx.is_main:
                    elapsed = time.monotonic() - started
                    completed = epoch * complete_updates_per_epoch + update_in_epoch
                    total = args.epochs * complete_updates_per_epoch
                    eta = elapsed / max(1, completed) * (total - completed)
                    tracking.log({
                        "train/loss": metric.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + update_in_epoch / complete_updates_per_epoch,
                        "perf/steps_per_second": completed / max(elapsed, 1e-9),
                        "perf/eta_seconds": eta,
                    }, step=global_step)
                    LOGGER.info(
                        "step=%05d/%05d epoch=%.3f loss=%.4f lr=%.3e steps/s=%.3f eta=%s peak_memory_gib=%.2f",
                        global_step, total, epoch + update_in_epoch / complete_updates_per_epoch,
                        metric.item(), optimizer.param_groups[0]["lr"],
                        completed / max(elapsed, 1e-9), time.strftime("%H:%M:%S", time.gmtime(eta)),
                        torch.cuda.max_memory_allocated(ctx.device) / 2**30 if ctx.device.type == "cuda" else 0.0,
                    )
                running_loss = 0.0
                running_updates = 0

            if update_in_epoch in save_updates:
                validation_loss, perplexity = validate(ema, validation_loader, ctx, args.precision)
                epoch_progress = epoch + update_in_epoch / complete_updates_per_epoch
                if ctx.is_main:
                    tracking.log({"val/nll": validation_loss, "val/ppl": perplexity},
                                 step=global_step)
                    LOGGER.info(
                        "validation epoch=%.3f loss=%.4f perplexity=%.4f",
                        epoch_progress, validation_loss, perplexity,
                    )
                    if not args.no_checkpoint:
                        label = f"{epoch_progress:06.1f}"
                        checkpoint_path = checkpoint_dir / f"epoch-{label}-step-{global_step:07d}.pt"
                        save_checkpoint(
                            checkpoint_path, training_model, ema, optimizer, scheduler,
                            scaler, args, global_step,
                            epoch + (1 if update_in_epoch == complete_updates_per_epoch else 0),
                            0 if update_in_epoch == complete_updates_per_epoch else batch_index + 1,
                            epoch_progress,
                        )
                        LOGGER.info("Saved checkpoint %s", checkpoint_path)
                barrier(ctx)
                training_model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if ctx.is_main:
        LOGGER.info("Training finished at step=%d elapsed=%s", global_step, time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - started)))
        tracking.summary({"complete": True, "final_optimizer_step": global_step})
    if ctx.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
