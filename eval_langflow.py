"""Evaluate fixed-length LangFlow samples against packed train/validation blocks.

The entire generated block is decoded with special tokens retained, matching
the released Gen-PPL evaluator. References are packed blocks, never prompts.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langflow import LangFlow, LangFlowConfig
from training.data import PROTOCOL, PackedDataset, load_metadata
from training.runtime import latest_checkpoint


LOGGER = logging.getLogger("langflow.eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--num-samples-per-split", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--ppl-model", default="gpt2-large")
    parser.add_argument("--ppl-batch-size", type=int, default=32)
    parser.add_argument("--max-ppl-length", type=int, default=64)
    parser.add_argument("--mauve-model", default="gpt2-large")
    parser.add_argument("--mauve-batch-size", type=int, default=32)
    parser.add_argument("--mauve-max-text-length", type=int, default=64)
    parser.add_argument("--skip-mauve", action="store_true", help="Gen-PPL-only checkpoint screening")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def load_generator(checkpoint_path, weights, device):
    if checkpoint_path.is_dir():
        checkpoint_path = latest_checkpoint(checkpoint_path)
    LOGGER.info("Loading %s weights from %s", weights, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("data_protocol") != PROTOCOL:
        raise ValueError("This evaluator requires a paper-packed checkpoint; legacy runs use their original evaluator")
    model = LangFlow(LangFlowConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint[weights])
    fingerprint = checkpoint["data_fingerprint"]
    del checkpoint
    gc.collect()
    return model.to(device).eval(), fingerprint


def select_references(cache_dir, split, num_samples, seed):
    dataset = PackedDataset(cache_dir, split)
    if not 0 < num_samples <= len(dataset):
        raise ValueError(f"Requested {num_samples} {split} blocks, only {len(dataset)} available")
    selected = np.random.default_rng(seed).choice(len(dataset), size=num_samples, replace=False)
    return selected.tolist(), [dataset[int(index)].tolist() for index in selected]


@torch.inference_mode()
def generate_samples(
    model: LangFlow,
    num_samples: int,
    sequence_length: int,
    num_steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> list[list[int]]:
    if num_steps < 2:
        raise ValueError("num_steps must be at least two")
    if batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generated: list[list[int]] = []
    for start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - start)
        with autocast_context(device, precision):
            token_ids = model.generate_samples(
                num_samples=current_batch,
                seq_length=sequence_length,
                num_steps=num_steps,
                device=device,
            )
        generated.extend(token_ids.cpu().tolist())
        LOGGER.info("Generated %d/%d samples", len(generated), num_samples)
    return generated


def mean_unigram_entropy(token_sequences: list[list[int]]) -> float:
    entropies = []
    for sequence in token_sequences:
        _, counts = np.unique(np.asarray(sequence, dtype=np.int64), return_counts=True)
        probabilities = counts / counts.sum()
        entropies.append(float(-(probabilities * np.log(probabilities)).sum()))
    return float(np.mean(entropies))


@torch.inference_mode()
def compute_generated_ppl(
    scorer,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float | int]:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        outputs = scorer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        valid_tokens = int((attention_mask.sum(dim=1) - 1).clamp(min=0).sum().item())
        if valid_tokens:
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_tokens += valid_tokens
    if total_tokens == 0:
        raise ValueError("No generated tokens could be scored")
    nll = total_nll / total_tokens
    return {
        "gen_ppl": math.exp(nll),
        "nll_per_token": nll,
        "num_scored_tokens": total_tokens,
    }


def compute_mauve(
    reference_texts: list[str],
    generated_texts: list[str],
    model_name: str,
    batch_size: int,
    max_text_length: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    try:
        import mauve
    except ImportError as error:
        raise RuntimeError("MAUVE is required: pip install mauve-text") from error
    device_id = device.index if device.type == "cuda" else -1
    if device.type == "cuda" and device_id is None:
        device_id = torch.cuda.current_device()
    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        featurize_model_name=model_name,
        device_id=device_id,
        max_text_length=max_text_length,
        batch_size=batch_size,
        seed=seed,
        verbose=True,
    )
    return {
        "mauve": float(result.mauve),
        "frontier_integral": float(result.frontier_integral),
        "mauve_star": float(result.mauve_star),
        "frontier_integral_star": float(result.frontier_integral_star),
    }


def write_samples(
    path: Path,
    indices: list[int],
    references: list[str],
    generations: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, reference, generation in zip(
            indices, references, generations, strict=True
        ):
            handle.write(json.dumps({
                "block_index": index,
                "reference": reference,
                "generation": generation,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint.is_dir():
        args.checkpoint = latest_checkpoint(args.checkpoint)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    generator, fingerprint = load_generator(args.checkpoint, args.weights, device)
    metadata = load_metadata(args.cache_dir)
    if metadata["fingerprint"] != fingerprint:
        raise ValueError("Evaluation cache must match the checkpoint training cache")
    if (args.sequence_length != generator.config.model_length
            or args.sequence_length != fingerprint["sequence_length"]
            or args.tokenizer != fingerprint["tokenizer"]
            or len(tokenizer) != generator.config.vocab_size):
        raise ValueError("Tokenizer/sequence length must match the checkpoint")
    if args.num_samples_per_split < 1 or args.ppl_batch_size < 1 or args.mauve_batch_size < 1:
        raise ValueError("Sample count and scoring batch sizes must be positive")

    split_outputs: dict[str, dict] = {}
    for split_number, split in enumerate(args.splits):
        LOGGER.info("Selecting references and generating for %s", split)
        indices, reference_ids = select_references(
            args.cache_dir, split, args.num_samples_per_split,
            args.seed + split_number,
        )
        generated_ids = generate_samples(
            generator, args.num_samples_per_split, args.sequence_length,
            args.num_steps, args.generation_batch_size,
            args.seed + 10_000 + split_number, device, args.precision,
        )
        reference_texts = tokenizer.batch_decode(reference_ids, skip_special_tokens=False)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        write_samples(
            args.output_dir / f"{split}-samples.jsonl",
            indices, reference_texts, generated_texts,
        )
        split_outputs[split] = {
            "block_indices": indices,
            "reference_texts": reference_texts,
            "generated_texts": generated_texts,
            "sample_entropy": mean_unigram_entropy(generated_ids),
            "mean_generated_tokens": float(np.mean([len(ids) for ids in generated_ids])),
            "mean_reference_tokens": float(np.mean([len(ids) for ids in reference_ids])),
        }

    del generator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    LOGGER.info("Loading Gen PPL scorer %s", args.ppl_model)
    scorer_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model, use_fast=True)
    if scorer_tokenizer.pad_token is None:
        scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_dtype = torch.float16 if device.type == "cuda" else torch.float32
    scorer = AutoModelForCausalLM.from_pretrained(
        args.ppl_model, dtype=scorer_dtype
    ).to(device).eval()
    for split in args.splits:
        split_outputs[split].update(compute_generated_ppl(
            scorer, scorer_tokenizer, split_outputs[split]["generated_texts"],
            args.ppl_batch_size, args.max_ppl_length, device,
        ))
        LOGGER.info(
            "%s Gen PPL=%.4f entropy=%.4f", split,
            split_outputs[split]["gen_ppl"], split_outputs[split]["sample_entropy"],
        )
    del scorer, scorer_tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for split_number, split in enumerate(args.splits if not args.skip_mauve else []):
        LOGGER.info("Computing MAUVE for %s", split)
        split_outputs[split].update(compute_mauve(
            split_outputs[split]["reference_texts"],
            split_outputs[split]["generated_texts"],
            args.mauve_model, args.mauve_batch_size,
            args.mauve_max_text_length, args.seed + split_number, device,
        ))

    results = {
        "complete": True,
        "mauve_computed": not args.skip_mauve,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_protocol": PROTOCOL,
        "data_fingerprint": fingerprint,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "splits": split_outputs,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        split: {
            key: value for key, value in values.items()
            if key not in {"block_indices", "reference_texts", "generated_texts"}
        }
        for split, values in split_outputs.items()
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Evaluation complete: %s", metrics_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
