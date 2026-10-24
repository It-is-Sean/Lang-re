"""Training orchestration for the fixed-length, paper-style protocol."""

import argparse
import contextlib
import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from langflow import LangFlow, LangFlowConfig
import experiment_tracking as tracking
from .data import PROTOCOL, PackedDataset, load_metadata, prepare_data
from .objective import LangFlowTrainingObjective, validate
from . import senior_roc64
from .runtime import (
    barrier, configure_logging, latest_checkpoint, precision_context, reduce_metrics,
    restore_checkpoint, sample_low_discrepancy_quantiles, save_and_stop_requested,
    save_checkpoint, seed_everything, setup_distributed, update_ema,
)

LOGGER = logging.getLogger("langflow.train")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR.parent.parent / "datasets/TinyStoriesV2")
    parser.add_argument("--dataset", choices=("tinystories", "rocstories"), default="tinystories")
    parser.add_argument("--training_profile", choices=("paper-packed", "senior-roc64"), default="paper-packed")
    parser.add_argument("--data_seed", type=int, default=None, help="Sampler seed; profile defaults: training seed / 0")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "results/langflow-tinystories")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Separate log directory; defaults to project logs/<run-name>",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "langflow/config.json")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--init_from", default=None, help="Optional Hugging Face model name or local model directory")
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="Senior ROC64 epoch-boundary EMA initialization; fresh optimizer, cumulative epoch/step")
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
    return parser


def run(args, ctx):
    senior = args.training_profile == "senior-roc64"
    if args.data_seed is None:
        args.data_seed = 0 if senior else args.seed
    if senior and (args.dataset != "rocstories" or args.tokenizer != "gpt2" or args.precision != "bf16"):
        raise ValueError("Senior ROC64 requires dataset=rocstories, tokenizer=gpt2 and precision=bf16")
    if min(args.global_batch_size, args.micro_batch_size, args.epochs, args.log_every) <= 0:
        raise ValueError("Batch sizes, epochs and log_every must be positive")
    if not 0 <= args.ema_decay < 1 or not 0 < args.validation_fraction < 1:
        raise ValueError("Invalid EMA decay or validation fraction")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if args.warmup_steps < 0 or args.bias_warmup_steps < 0:
        raise ValueError("Warmup steps must be nonnegative; zero disables warmup")
    if args.resume and args.resume_latest:
        raise ValueError("Use only one of --resume and --resume_latest")
    if sum(bool(value) for value in (args.init_from, args.init_checkpoint, args.resume, args.resume_latest)) > 1:
        raise ValueError("Use only one initialization or resume option")
    if args.init_checkpoint and not senior:
        raise ValueError("Checkpoint EMA initialization requires senior-roc64")
    checkpoint_dir = args.output_dir / "checkpoints"
    resume_path = latest_checkpoint(checkpoint_dir) if args.resume_latest else args.resume
    if resume_path is None and checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
        raise ValueError("Output contains checkpoints; resume or choose a new output directory")
    log_dir = args.log_dir or PROJECT_DIR.parent.parent / "logs" / args.output_dir.name
    configure_logging(log_dir, ctx)
    seed_everything(args.seed, 0 if senior else ctx.rank)
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    initial = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        model = LangFlow(LangFlowConfig.from_dict(initial["model_config"]))
        model.load_state_dict(initial["ema"], strict=True)
    elif args.init_from:
        model = LangFlow.from_pretrained(args.init_from)
    else:
        config = LangFlowConfig.from_pretrained(args.config)
        config.vocab_size = len(tokenizer)
        model = LangFlow(config)
    sequence_length = args.sequence_length or model.config.model_length
    if sequence_length < 3 or model.config.vocab_size != len(tokenizer):
        raise ValueError("Invalid sequence length or tokenizer vocabulary mismatch")
    model.config.model_length = sequence_length
    if senior:
        if sequence_length != 64:
            raise ValueError("Senior ROC64 uses length 64")
        model.config.precision_scope = "backbone-bf16"
    args.sequence_length = sequence_length
    source_dir = args.data_dir.parent if args.data_dir.is_file() else args.data_dir
    args.cache_dir = args.cache_dir or source_dir / (
        f"paper-packed-{args.tokenizer.replace('/', '--')}-l{sequence_length}"
    )
    if senior:
        args.cache_dir = args.data_dir
    preparation_error = [None]
    if ctx.is_main:
        try:
            if senior:
                senior_roc64.load_metadata(args.cache_dir)
                senior_roc64.StoryDataset(args.cache_dir, "train")
                senior_roc64.StoryDataset(args.cache_dir, "validation")
            else:
                prepare_data(args, tokenizer, sequence_length)
        except Exception as error:
            preparation_error[0] = f"{type(error).__name__}: {error}"
    if ctx.world_size > 1:
        dist.broadcast_object_list(preparation_error, src=0)
    if preparation_error[0]:
        raise RuntimeError(preparation_error[0])
    metadata = (senior_roc64.load_metadata if senior else load_metadata)(args.cache_dir)
    fingerprint = metadata["fingerprint"]
    dataset_class = senior_roc64.StoryDataset if senior else PackedDataset
    train_dataset = dataset_class(args.cache_dir, "train")
    valid_dataset = dataset_class(args.cache_dir, "validation")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=args.data_seed, drop_last=True
    )
    # Shard validation without padding/duplicating its last examples.
    valid_sampler = range(ctx.rank, len(valid_dataset), ctx.world_size)
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
    initial_epoch = 0
    if initial is not None:
        if (initial.get("data_protocol") != senior_roc64.PROTOCOL
                or initial.get("data_fingerprint") != fingerprint
                or initial["args"]["global_batch_size"] != args.global_batch_size):
            raise ValueError("EMA initialization requires identical data and global batch")
        initial_epoch, remainder = divmod(initial["step"], complete_updates_per_epoch)
        source_microbatches = (len(train_dataset) // initial["world_size"]
                               // initial["args"]["micro_batch_size"])
        source_accumulation = args.global_batch_size // (initial["world_size"] * initial["args"]["micro_batch_size"])
        completed_boundary = (initial["epoch"] == initial_epoch - 1
                              and initial["batch_in_epoch"] == source_microbatches // source_accumulation * source_accumulation)
        next_boundary = initial["epoch"] == initial_epoch and initial["batch_in_epoch"] == 0
        if remainder or not (completed_boundary or next_boundary) or args.epochs <= initial_epoch:
            raise ValueError("Require an epoch-boundary source and a larger cumulative --epochs target")
        LOGGER.info("Initialize EMA from %s at epoch=%d step=%d; fresh optimizer, cumulative EMA/bias schedule",
                    args.init_checkpoint, initial_epoch, initial["step"])
        del initial
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

    # Initialize EMA only after DDP has synchronized initial model parameters.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step if senior else step + 1) / max(1, args.warmup_steps))
        if args.warmup_steps else 1.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and ctx.device.type == "cuda")

    start_step = initial_epoch * complete_updates_per_epoch
    start_epoch = initial_epoch
    start_batch = 0
    if resume_path is not None:
        start_step, start_epoch, start_batch = restore_checkpoint(
            resume_path, model, ema, optimizer, lr_scheduler, scaler,
            args, ctx, fingerprint,
        )
        if start_batch % accumulation_steps or start_batch > complete_updates_per_epoch * accumulation_steps:
            raise ValueError("Checkpoint is not at a complete global-batch boundary")
        LOGGER.info("Resumed %s at step=%d epoch=%d batch=%d",
                    resume_path, start_step, start_epoch, start_batch)

    if ctx.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)
        (args.output_dir / "data.json").write_text(json.dumps(metadata, indent=2))
        LOGGER.info("data_protocol=%s cache=%s", fingerprint["protocol"], args.cache_dir)
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

    barrier(ctx)

    tracking.start(
        {**vars(args), "data_protocol": fingerprint["protocol"], "data_fingerprint": fingerprint,
         "world_size": ctx.world_size, "accumulation_steps": accumulation_steps,
         "updates_per_epoch": complete_updates_per_epoch, "start_step": start_step,
         "model_config": model.config.to_dict()},
        args.output_dir, model="langflow", job_type="train", log_dir=log_dir, rank=ctx.rank,
    )

    def save(epoch, batch, final=False):
        suffix = "-final" if final else ""
        save_checkpoint(
            checkpoint_dir / f"step-{step:07d}{suffix}.pt", model, ema, optimizer,
            lr_scheduler, scaler, args, ctx, fingerprint, step, epoch, batch,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    running = torch.zeros(3, device=ctx.device)
    running_updates = 0
    log_started = time.monotonic()
    quantiles_for_rank: torch.Tensor | None = None
    accumulation_index = 0
    stop_training = args.max_steps > 0 and step >= args.max_steps
    last_epoch = start_epoch
    last_batch = start_batch
    first_logged_step = start_step + 1

    for epoch in range(start_epoch, args.epochs):
        if stop_training:
            break
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
            input_ids = batch
            if quantiles_for_rank is None:
                sampler = senior_roc64.quantiles if senior else sample_low_discrepancy_quantiles
                quantiles_for_rank = sampler(args.global_batch_size, ctx)
            q_start = accumulation_index * args.micro_batch_size
            quantiles = quantiles_for_rank[q_start : q_start + args.micro_batch_size]
            input_ids = input_ids.to(ctx.device, non_blocking=True)
            bias_scale = min(1.0, step / args.bias_warmup_steps) if args.bias_warmup_steps else 1.0
            final_micro_step = accumulation_index + 1 == accumulation_steps
            sync_context = contextlib.nullcontext()
            if isinstance(objective, DDP) and not final_micro_step:
                sync_context = objective.no_sync()

            # DDP requires both forward and backward to stay inside no_sync;
            # leaving the context before backward corrupts reducer state on
            # the next accumulated micro-batch.
            with sync_context:
                with precision_context("fp32" if senior else args.precision, ctx.device):
                    total_loss, ce_loss, scheduler_loss = objective(
                        input_ids, quantiles, bias_scale
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
            if not senior:
                with torch.no_grad():
                    model.proposal.scale.clamp_(min=1e-4)
                    model.proposal.entropy.clamp_(min=1e-4)
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            step += 1
            update_in_epoch += 1
            decay = senior_roc64.ema_decay(args.ema_decay, step) if senior else args.ema_decay
            update_ema(ema, model, decay)
            accumulation_index = 0
            quantiles_for_rank = None
            running_updates += 1

            if step == first_logged_step or step % args.log_every == 0:
                metrics = reduce_metrics(running / max(1, running_updates), ctx)
                elapsed = time.monotonic() - log_started
                if ctx.is_main:
                    tracking.log({
                        "train/loss": metrics[0].item(), "train/ce": metrics[1].item(),
                        "train/scheduler_loss": metrics[2].item(),
                        "train/epoch": epoch + update_in_epoch / complete_updates_per_epoch,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "perf/steps_per_second": running_updates / max(elapsed, 1e-9),
                        "schedule/loc": model.proposal.loc.item(),
                        "schedule/scale": model.proposal.scale.item(),
                        "schedule/entropy": model.proposal.entropy.item(),
                        "schedule/bias_scale": bias_scale,
                    }, step=step)
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

            fractional_due = update_in_epoch in fractional_save_updates
            stop_requested = save_and_stop_requested(args.output_dir, ctx)
            if fractional_due or (args.validate_every > 0 and step % args.validate_every == 0):
                if senior:
                    scores = senior_roc64.validate(ema, valid_dataset, ctx, args.micro_batch_size,
                                                  args.self_condition_probability, bias_scale)
                    raw = senior_roc64.validate(model, valid_dataset, ctx, args.micro_batch_size,
                                               args.self_condition_probability, bias_scale)
                    scores.update({f"{k}_raw": v for k, v in raw.items()})
                    LOGGER.info("step=%07d validation_ce=%.4f validation_loss=%.4f (EMA)",
                                step, scores["ce"], scores["loss"])
                else:
                    scores = validate(ema, valid_loader, ctx, args.precision, args.validation_batches)
                    LOGGER.info("step=%07d validation_ce_no_sc=%.4f validation_ce_sc=%.4f",
                                step, scores["ce_no_sc"], scores["ce_sc"])
                tracking.log({f"val/{key}": value for key, value in scores.items()}, step=step)
            # One checkpoint per step, even when multiple triggers coincide.
            if stop_requested or fractional_due or (args.checkpoint_every > 0 and step % args.checkpoint_every == 0):
                # Senior run keeps exactly one full checkpoint per epoch,
                # including the final epoch (no second final.pt copy).
                is_final = senior and epoch + 1 == args.epochs and update_in_epoch == complete_updates_per_epoch
                save(epoch, batch_index + 1, final=is_final)
            if stop_requested:
                tracking.summary({"stopped_by_request": True, "final_optimizer_step": step})
                LOGGER.info("Saved current progress and stopped by SAVE_AND_STOP request")
                return

            if args.max_steps > 0 and step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break
        start_batch = 0
        last_batch = 0
        last_epoch = epoch + 1

    if not (senior and (checkpoint_dir / f"step-{step:07d}-final.pt").exists()):
        save(last_epoch, last_batch, final=True)
    if ctx.is_main:
        for name, exported in (("final-model", model), ("final-ema", ema)):
            exported.save_pretrained(args.output_dir / name, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir / name)
        LOGGER.info("Training complete at step %d; exported model and EMA", step)
        tracking.summary({"complete": True, "final_optimizer_step": step})
    barrier(ctx)


@tracking.tracked
def main():
    args = build_parser().parse_args()
    ctx = setup_distributed()
    try:
        run(args, ctx)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
