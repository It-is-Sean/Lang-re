"""Evaluate packed AR unconditionally, or legacy story checkpoints on prompts.

For each split, this script samples distinct reference stories, keeps a fixed
prefix as the prompt, generates an autoregressive continuation, and reports:

* generative perplexity under a separate causal LM (GPT-2 Large by default),
* mean unigram entropy of generated token sequences, and
* MAUVE between generated stories and references from the same split.

Samples and selected dataset indices are saved so every metric is auditable.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langflow.ar_model import LangFlowAR
from training.data import PROTOCOL, load_metadata
from eval_langflow import select_references
import experiment_tracking as tracking


LOGGER = logging.getLogger("langflow.eval_ar")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=("train", "validation"))
    parser.add_argument("--num-samples-per-split", type=int, default=2000)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--ppl-model", default="gpt2-large")
    parser.add_argument("--ppl-batch-size", type=int, default=32)
    parser.add_argument("--max-ppl-length", type=int, default=64)
    parser.add_argument("--mauve-model", default="gpt2-large")
    parser.add_argument("--mauve-batch-size", type=int, default=32)
    parser.add_argument("--mauve-max-text-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generator(checkpoint_path: Path, weights: str, device: torch.device) -> LangFlowAR:
    LOGGER.info("Loading %s weights from %s", weights, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    config = checkpoint["model_config"]
    model = LangFlowAR(
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        n_blocks=int(config["n_blocks"]),
        n_heads=int(config["n_heads"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint[weights])
    model.data_protocol = checkpoint.get('data_protocol', 'story-aligned')
    model.data_fingerprint = checkpoint.get('data_fingerprint')
    del checkpoint
    gc.collect()
    return model.to(device).eval()


def select_split_samples(
    cache_dir: Path,
    split: str,
    num_samples: int,
    prompt_length: int,
    seed: int,
) -> tuple[list[int], list[list[int]], torch.Tensor]:
    tokens = np.load(cache_dir / f"{split}-tokens.npy", mmap_mode="r")
    lengths = np.load(cache_dir / f"{split}-lengths.npy", mmap_mode="r")
    eligible = np.flatnonzero(lengths > prompt_length)
    if num_samples > len(eligible):
        raise ValueError(f"Requested {num_samples} {split} samples, only {len(eligible)} are eligible")
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=num_samples, replace=False)
    references = [tokens[index, : int(lengths[index])].astype(np.int64).tolist() for index in selected]
    prompts = torch.from_numpy(
        np.stack([tokens[index, :prompt_length].astype(np.int64) for index in selected])
    )
    return selected.astype(np.int64).tolist(), references, prompts


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    logits = logits.float() / temperature
    if top_k > 0:
        cutoff = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits.masked_fill_(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(1, sorted_indices, sorted_logits)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(1)


@torch.inference_mode()
def generate_continuations(
    model: LangFlowAR,
    prompts: torch.Tensor,
    eos_token_id: int,
    max_length: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    device: torch.device,
    stop_at_eos: bool = True,
) -> list[list[int]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if prompts.shape[1] >= max_length:
        raise ValueError("prompt_length must be smaller than max_length")
    rng = torch.Generator(device=device).manual_seed(seed)
    generated_samples: list[list[int]] = []
    for start in range(0, len(prompts), batch_size):
        token_ids = prompts[start : start + batch_size].to(device, non_blocking=True)
        finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=device)
        while token_ids.shape[1] < max_length and (not stop_at_eos or not bool(finished.all())):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(token_ids)[:, -1]
            next_token = sample_next_token(logits, temperature, top_p, top_k, rng)
            if stop_at_eos:
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
            finished |= next_token.eq(eos_token_id)

        for row in token_ids.cpu().tolist():
            if stop_at_eos:
                try:
                    eos_position = row.index(eos_token_id, prompts.shape[1])
                    row = row[: eos_position + 1]
                except ValueError:
                    pass
            generated_samples.append(row)
        LOGGER.info("Generated %d/%d samples", len(generated_samples), len(prompts))
    return generated_samples


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
            texts[start : start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
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
    return {"gen_ppl": math.exp(nll), "nll_per_token": nll, "num_scored_tokens": total_tokens}


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
    prompts: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, reference, generation, prompt in zip(indices, references, generations, prompts, strict=True):
            handle.write(json.dumps({
                "dataset_index": index,
                "prompt": prompt,
                "reference": reference,
                "generation": generation,
            }, ensure_ascii=False) + "\n")


@tracking.tracked
def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint.is_dir():
        candidates = [(int(match[1]), path) for path in args.checkpoint.glob('epoch-*-step-*.pt')
                      if (match := re.search(r'-step-(\d+)\.pt$', path.name))]
        if not candidates:
            raise FileNotFoundError(f'No AR checkpoints in {args.checkpoint}')
        args.checkpoint = max(candidates)[1]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Generator tokenizer must define eos_token_id")

    generator_model = load_generator(args.checkpoint, args.weights, device)
    packed = generator_model.data_protocol == PROTOCOL
    metadata = json.loads((args.cache_dir / 'metadata.json').read_text())
    if packed != (metadata['fingerprint'].get('protocol') == PROTOCOL):
        raise ValueError('Checkpoint and evaluation cache protocols differ')
    if packed:
        fingerprint = load_metadata(args.cache_dir)['fingerprint']
        if generator_model.data_fingerprint != fingerprint:
            raise ValueError('Packed cache fingerprint differs from training')
        if (args.max_length != fingerprint['sequence_length']
                or args.tokenizer != fingerprint['tokenizer']
                or len(tokenizer) != fingerprint['vocab_size']
                or tokenizer.bos_token_id != fingerprint['bos_token_id']
                or tokenizer.eos_token_id != fingerprint['eos_token_id']
                or args.max_ppl_length < args.max_length
                or args.mauve_max_text_length < args.max_length):
            raise ValueError('Packed evaluation requires matching tokenizer and full-length scoring')
        args.prompt_length = 1
    args.data_protocol = generator_model.data_protocol
    args.generation_mode = 'unconditional-bos-fixed-length' if packed else 'story-prefix-continuation'
    if args.num_samples_per_split < 1 or args.generation_batch_size < 1:
        raise ValueError('Sample count and batch size must be positive')
    tracking.start({**vars(args), "data_fingerprint": metadata['fingerprint']},
                   args.output_dir, model="ar", job_type="eval",
                   group_dir=args.checkpoint.parent.parent)
    split_outputs: dict[str, dict] = {}
    for split_number, split in enumerate(args.splits):
        LOGGER.info("Selecting and generating %s split", split)
        if packed:
            indices, reference_ids = select_references(
                args.cache_dir, split, args.num_samples_per_split, args.seed + split_number)
            # References are never used as prompts, not even their first token.
            prompts = torch.full((args.num_samples_per_split, 1), tokenizer.bos_token_id, dtype=torch.long)
        else:
            indices, reference_ids, prompts = select_split_samples(
                args.cache_dir, split, args.num_samples_per_split,
                args.prompt_length, args.seed + split_number,
            )
        generated_ids = generate_continuations(
            generator_model, prompts, tokenizer.eos_token_id, args.max_length,
            args.generation_batch_size, args.temperature, args.top_p, args.top_k,
            args.seed + 10_000 + split_number, device, stop_at_eos=not packed,
        )
        reference_texts = tokenizer.batch_decode(reference_ids, skip_special_tokens=not packed)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=not packed)
        prompt_texts = tokenizer.batch_decode(prompts.tolist(), skip_special_tokens=not packed)
        write_samples(
            args.output_dir / f"{split}-samples.jsonl", indices,
            reference_texts, generated_texts, prompt_texts,
        )
        split_outputs[split] = {
            "dataset_indices": indices,
            "reference_texts": reference_texts,
            "generated_texts": generated_texts,
            "sample_entropy": mean_unigram_entropy(generated_ids),
            "mean_generated_tokens": float(np.mean([len(ids) for ids in generated_ids])),
            "mean_reference_tokens": float(np.mean([len(ids) for ids in reference_ids])),
        }

    del generator_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    LOGGER.info("Loading Gen PPL scorer %s", args.ppl_model)
    scorer_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model, use_fast=True)
    if scorer_tokenizer.pad_token is None:
        scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_dtype = torch.float16 if device.type == "cuda" else torch.float32
    scorer = AutoModelForCausalLM.from_pretrained(args.ppl_model, dtype=scorer_dtype).to(device).eval()
    for split in args.splits:
        split_outputs[split].update(compute_generated_ppl(
            scorer, scorer_tokenizer, split_outputs[split]["generated_texts"],
            args.ppl_batch_size, args.max_ppl_length, device,
        ))
        tracking.log_evaluation(split_outputs, complete=False)
        LOGGER.info(
            "%s Gen PPL=%.4f entropy=%.4f", split,
            split_outputs[split]["gen_ppl"], split_outputs[split]["sample_entropy"],
        )
    del scorer, scorer_tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for split_number, split in enumerate(args.splits):
        LOGGER.info("Computing MAUVE for %s", split)
        split_outputs[split].update(compute_mauve(
            split_outputs[split]["reference_texts"],
            split_outputs[split]["generated_texts"],
            args.mauve_model, args.mauve_batch_size,
            args.mauve_max_text_length, args.seed + split_number, device,
        ))

    results = {
        "complete": True,
        "data_fingerprint": metadata['fingerprint'],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "splits": split_outputs,
    }
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        split: {key: value for key, value in values.items() if key not in {
            "dataset_indices", "reference_texts", "generated_texts"
        }}
        for split, values in split_outputs.items()
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Evaluation complete: %s", output_path)
    tracking.log_evaluation(summary, complete=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
