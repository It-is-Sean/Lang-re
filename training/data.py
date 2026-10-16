"""Document tokenization and [BOS] + packed payload + [EOS] datasets.

Split documents before packing. Never build these caches from the truncated
story-aligned arrays used by the AR baseline. As in the released dataloader,
each batch of 1000 documents is concatenated and its incomplete tail dropped.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


PROTOCOL = "paper-packed-v1"
PACK_BATCH_SIZE = 1000
LOGGER = logging.getLogger("langflow.data")
_TOKENIZER = None
_LENGTH = None


def pack_documents(documents, sequence_length, bos, eos):
    """Append document EOS, concatenate, chunk L-2 tokens, wrap BOS/EOS."""
    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")
    payload = list(itertools.chain.from_iterable([*ids, eos] for ids in documents))
    width = sequence_length - 2
    count, dropped = divmod(len(payload), width)
    blocks = np.empty((count, sequence_length), dtype=np.int64)
    blocks[:, 0] = bos
    blocks[:, -1] = eos
    blocks[:, 1:-1] = np.asarray(payload[:count * width], dtype=np.int64).reshape(count, width)
    return blocks, dropped


def _init_tokenizer(path, sequence_length):
    global _TOKENIZER, _LENGTH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    _TOKENIZER.model_max_length = int(1e30)
    _LENGTH = sequence_length


def _encode_batch(documents):
    tokens = _TOKENIZER(
        documents, add_special_tokens=False, truncation=False,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]
    blocks, dropped = pack_documents(
        tokens, _LENGTH, _TOKENIZER.bos_token_id, _TOKENIZER.eos_token_id,
    )
    return blocks, dropped, len(documents)


def _batches(documents):
    iterator = iter(documents)
    while batch := list(itertools.islice(iterator, PACK_BATCH_SIZE)):
        yield batch


def read_tinystories(path):
    """Stream complete stories separated by the dataset's end-of-text marker."""
    parts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            chunks = line.split("<|endoftext|>")
            for index, chunk in enumerate(chunks):
                parts.append(chunk)
                if index < len(chunks) - 1:
                    text = "".join(parts).strip()
                    if text:
                        yield text
                    parts.clear()
    text = "".join(parts).strip()
    if text:
        yield text


def read_rocstories(path, seed, validation_fraction):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        stories = [
            " ".join(row[f"sentence{i}"].strip() for i in range(1, 6))
            for row in csv.DictReader(handle)
        ]
    indices = list(range(len(stories)))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, round(len(stories) * validation_fraction))
    if n_valid >= len(stories):
        raise ValueError("ROCStories needs nonempty train and validation document splits")
    splits = {"validation": indices[:n_valid], "train": indices[n_valid:]}
    return {split: [stories[i] for i in selected] for split, selected in splits.items()}, splits


def load_metadata(cache_dir):
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if metadata.get("fingerprint", {}).get("protocol") != PROTOCOL:
        raise ValueError(f"{cache_dir} is not a {PROTOCOL} cache; prepare a new cache from raw documents")
    return metadata


def prepare_data(args, tokenizer, sequence_length):
    """Called on rank zero only; metadata is committed after both split files."""
    if args.dataset == "rocstories":
        source = args.data_dir if args.data_dir.is_file() else args.data_dir / "ROCStories_winter2017.csv"
        sources = {"csv": source}
    else:
        sources = {
            split: args.data_dir / f"TinyStoriesV2-GPT4-{suffix}.txt"
            for split, suffix in (("train", "train"), ("validation", "valid"))
        }
    fingerprint = {
        "protocol": PROTOCOL, "dataset": args.dataset, "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer), "sequence_length": sequence_length,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16" if len(tokenizer) <= 65536 else "uint32",
        "pack_batch_size": PACK_BATCH_SIZE,
        "seed": args.seed, "validation_fraction": args.validation_fraction,
        "sources": {
            split: {"path": str(path.resolve()), "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns}
            for split, path in sources.items()
        },
    }
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Paper packing requires a tokenizer with BOS and EOS tokens")
    cache_dir = args.cache_dir
    if (cache_dir / "metadata.json").exists():
        metadata = load_metadata(cache_dir)
        if metadata["fingerprint"] != fingerprint:
            raise ValueError(f"Cache configuration changed; choose a new --cache_dir: {cache_dir}")
        for split in ("train", "validation"):
            PackedDataset(cache_dir, split)  # Verify file sizes before reuse.
        return metadata
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = cache_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    if args.dataset == "rocstories":
        documents, indices = read_rocstories(source, args.seed, args.validation_fraction)
        for split, selected in indices.items():
            np.save(cache_dir / f"{split}-document-indices.npy", np.asarray(selected))
    else:
        documents = {split: read_tinystories(path) for split, path in sources.items()}
    dtype = np.dtype(fingerprint["dtype"])
    metadata = {"fingerprint": fingerprint, "splits": {}}
    pool = None
    try:
        if args.preprocess_workers > 1:
            pool = mp.get_context("spawn").Pool(
                args.preprocess_workers, initializer=_init_tokenizer,
                initargs=(str(tokenizer_dir), sequence_length),
            )
        else:
            _init_tokenizer(str(tokenizer_dir), sequence_length)
        for split, texts in documents.items():
            batches = _batches(texts)
            encoded = pool.imap(_encode_batch, batches) if pool else map(_encode_batch, batches)
            stats = {"blocks": 0, "documents": 0, "dropped_tail_tokens": 0}
            temporary = cache_dir / f"{split}.bin.tmp"
            with temporary.open("wb") as handle:
                for blocks, dropped, count in encoded:
                    blocks.astype(dtype).tofile(handle)
                    stats["blocks"] += len(blocks)
                    stats["documents"] += count
                    stats["dropped_tail_tokens"] += dropped
            if stats["blocks"] == 0:
                raise ValueError(f"No complete {sequence_length}-token blocks in {split}")
            os.replace(temporary, cache_dir / f"{split}.bin")
            metadata["splits"][split] = stats
            LOGGER.info("Packed %s: %s", split, stats)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    temporary = cache_dir / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, cache_dir / "metadata.json")
    return metadata


class PackedDataset(Dataset):
    """All positions, including BOS and every EOS, participate in attention/loss."""

    def __init__(self, cache_dir, split):
        metadata = load_metadata(cache_dir)
        self.path = cache_dir / f"{split}.bin"
        self.length = metadata["fingerprint"]["sequence_length"]
        self.dtype = np.dtype(metadata["fingerprint"]["dtype"])
        self.count = metadata["splits"][split]["blocks"]
        expected_bytes = self.count * self.length * self.dtype.itemsize
        if self.path.stat().st_size != expected_bytes:
            raise ValueError(f"Incomplete or incompatible packed cache: {self.path}")
        self._tokens = None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r", shape=(self.count, self.length))
        return torch.from_numpy(np.array(self._tokens[index], dtype=np.int64, copy=True))

    def __getstate__(self):
        return {**self.__dict__, "_tokens": None}
