"""Small W&B adapter shared by training and evaluation (no torch dependency).

Each invocation is a separate run, including checkpoint resumes. Related runs
share a group; optimizer_step is an explicit axis, not W&B's history counter.
Offline is the safe default until WANDB_MODE=online is explicitly requested.
"""

import functools
import json
import logging
import os
from pathlib import Path
import sys

_run = None


def clean_config(value):
    if isinstance(value, dict):
        return {str(k): clean_config(v) for k, v in value.items()
                if str(k).lower() not in {
                    "api_key", "wandb_api_key", "password", "secret", "access_token", "auth_token"}}
    if isinstance(value, (tuple, list)):
        return [clean_config(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def log_directory(output_dir):
    """Keep tracking files out of results/, including evaluation subdirectories."""
    output = Path(output_dir).resolve()
    for parent in output.parents:
        if parent.name == "results":
            return parent.parent / "logs" / output.relative_to(parent)
    return output.parent / "logs" / output.name


def start(config, output_dir, *, model, job_type, log_dir=None, rank=0, group_dir=None):
    global _run
    if rank != 0:
        return
    mode = os.environ.get("WANDB_MODE", "offline")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("WANDB_MODE must be online, offline or disabled")
    if mode == "disabled":
        return
    if _run is not None:
        raise RuntimeError("A tracking run is already active")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb in this Python environment, or set WANDB_MODE=disabled") from exc
    directory = Path(os.environ.get("WANDB_DIR") or log_dir or log_directory(output_dir))
    directory.mkdir(parents=True, exist_ok=True)
    group = Path(group_dir or output_dir).resolve()
    default_group = f"{group.parent.name}/{group.name}"
    # Console output can contain dataset examples in other trainers. Do not
    # upload it, source code, weights or sample text through this adapter.
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    # Do not inherit an ID from a parent process (e.g. a checkpoint sweep).
    _run = wandb.init(
        project=os.environ.get("WANDB_PROJECT") or "MSEFlow",
        entity=os.environ.get("WANDB_ENTITY") or "yunchenjiang-university-of-illinois-urbana-champaign-org",
        mode=mode, dir=str(directory), id=wandb.util.generate_id(), resume="never",
        name=os.environ.get("WANDB_NAME") or f"{model}-{job_type}-{Path(output_dir).name}",
        group=os.environ.get("WANDB_RUN_GROUP") or default_group,
        job_type=job_type, tags=[model, job_type], save_code=False,
        config=clean_config({**config, "model_family": model, "job_type": job_type}),
    )
    _run.define_metric("optimizer_step")
    for namespace in ("train/*", "val/*", "schedule/*", "perf/*"):
        _run.define_metric(namespace, step_metric="optimizer_step")
    _run.define_metric("checkpoint_step")
    _run.define_metric("sweep/*", step_metric="checkpoint_step")
    _run.summary["complete"] = False
    # A local pointer, not a credential. Unique filename also preserves resumed
    # offline sessions instead of pretending W&B supports offline resume.
    (directory / f"tracking-{_run.id}.json").write_text(json.dumps({
        "id": _run.id, "project": _run.project, "mode": mode,
        "group": os.environ.get("WANDB_RUN_GROUP") or default_group,
    }, indent=2) + "\n")


def log(values, *, step=None):
    if _run is not None:
        values = dict(values)
        if step is not None:
            values["optimizer_step"] = step
        # No explicit W&B step: train and validation can share an optimizer
        # step without one log being dropped as an out-of-order history row.
        _run.log(values)


def log_evaluation(splits, *, complete):
    metrics = {f"eval/{split}/{key}": value
               for split, values in splits.items() for key, value in values.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    log(metrics)
    if _run is not None:
        _run.summary.update({**metrics, "complete": complete})


def summary(values):
    if _run is not None:
        _run.summary.update(clean_config(values))


def tracked(function):
    """Finish on success/failure, but never mask the original exception."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        global _run
        try:
            return function(*args, **kwargs)
        finally:
            failed = sys.exc_info()[0] is not None
            if _run is not None:
                run, _run = _run, None
                try:
                    run.summary["process_succeeded"] = not failed
                    run.finish(exit_code=1 if failed else 0)
                except Exception:
                    logging.getLogger(__name__).exception("Could not finish W&B logging")
    return wrapper
