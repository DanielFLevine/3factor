import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
import torch
from rng_helpers import capture_rng_state

from dataclasses import dataclass
from typing import Any, Iterator, List, Tuple

import constants as constants_module
from params import params

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRICS_FIELDNAMES = [
    "episode",
    "timestamp",
    "loss",
    "mean_reward",
    "mean_test_reward",
    "mean_loss_last_pe",
    "mean_reward_last_pe",
    "mean_test_reward_last_pe",
    "time_spent_last_pe",
    "test_perf_overall",
    "test_perf_adjacent",
    "test_perf_nonadjacent",
    "test_perf_old_cues",
]

@dataclass
class RunMetadataBundle:
    run_metadata: dict
    run_metadata_path: Path
    metrics_csv_path: Path
    checkpoint_path: Path
    metrics_rows: List[Any]
    run_dir: Path
    params_path: Path
    constants_path: Path
    seed: Any

    def as_tuple(self) -> Tuple[Any, ...]:
        return (
            self.run_metadata,
            self.run_metadata_path,
            self.metrics_csv_path,
            self.checkpoint_path,
            self.metrics_rows,
            self.run_dir,
            self.params_path,
            self.constants_path,
            self.seed,
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(self.as_tuple())


def get_run_metadata(args):
    resume_dir = Path(args.resume_dir).resolve() if args.resume_dir else None
    is_resuming = resume_dir is not None
    if is_resuming:
        run_dir = resume_dir
        if not run_dir.is_dir():
            raise ValueError(f"Resume directory {run_dir} does not exist or is not a directory")
        logger.info("Resuming run from %s", run_dir)
        run_metadata_path = run_dir / "run_metadata.json"
        run_metadata = load_json(run_metadata_path, default={}) or {}
        RUN_TIMESTAMP = run_metadata.get("run_timestamp", "unknown")
        params_path = run_dir / "params.json"
        constants_path = run_dir / "constants.json"
        metrics_csv_path = run_dir / "metrics.csv"
        metrics_rows = load_metrics_csv(metrics_csv_path)
        checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else run_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_params = checkpoint_state.get('params')
        if saved_params:
            params.update(saved_params)
        seed = params.get('rngseed', args.seed)
        resume_history = run_metadata.get("resume_history", [])
        resume_history.append(datetime.now().isoformat())
        run_metadata["resume_history"] = resume_history
        run_metadata["resume_count"] = len(resume_history)
        run_metadata["resumed_from_checkpoint"] = str(checkpoint_path)
        run_metadata.setdefault("run_directory", str(run_dir))
        dump_json(run_metadata, run_metadata_path)
    else:
        run_dir, RUN_TIMESTAMP = create_unique_run_dir(RUNS_DIR)
        logger.info("Run outputs will be saved to %s", run_dir)
        params_path = run_dir / "params.json"
        constants_path = run_dir / "constants.json"
        run_metadata_path = run_dir / "run_metadata.json"
        run_metadata = {
            "run_timestamp": RUN_TIMESTAMP,
            "created_at": datetime.now().isoformat(),
            "run_directory": str(run_dir),
        }
        dump_json(run_metadata, run_metadata_path)
        metrics_csv_path = run_dir / "metrics.csv"
        metrics_rows = []
        checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else run_dir / "checkpoint.pt"
        seed = args.seed

    return RunMetadataBundle(
        run_metadata=run_metadata,
        run_metadata_path=run_metadata_path,
        metrics_csv_path=metrics_csv_path,
        checkpoint_path=checkpoint_path,
        metrics_rows=metrics_rows,
        run_dir=run_dir,
        params_path=params_path,
        constants_path=constants_path,
        seed=seed,
    )

def save_checkpoint(
    run_metadata_bundle: RunMetadataBundle,
    next_episode,
    net,
    optimizer,
    all_losses_objective,
    all_mean_rewards_ep,
    all_mean_testrewards_ep,
    all_grad_norms,
    lossbetweensaves,
    old_cue_data,
    totalnbtrials,
    nbtrialswithcc,
    nowtime,
):
    run_metadata = run_metadata_bundle.run_metadata
    metrics_csv_path = run_metadata_bundle.metrics_csv_path
    metrics_rows = run_metadata_bundle.metrics_rows
    checkpoint_path = run_metadata_bundle.checkpoint_path
    run_metadata_path = run_metadata_bundle.run_metadata_path
    checkpoint_payload = {
        "version": 1,
        "next_episode": next_episode,
        "params": params.copy(),
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "all_losses_objective": list(all_losses_objective),
        "all_mean_rewards_ep": list(all_mean_rewards_ep),
        "all_mean_testrewards_ep": list(all_mean_testrewards_ep),
        "all_grad_norms": [float(x) for x in all_grad_norms],
        "lossbetweensaves": float(lossbetweensaves),
        "old_cue_data": old_cue_data,
        "totalnbtrials": int(totalnbtrials),
        "nbtrialswithcc": int(nbtrialswithcc),
        "nowtime": float(nowtime),
        "rng_state": capture_rng_state(),
        "metrics_rows": len(metrics_rows),
        "checkpoint_created_at": datetime.now().isoformat(),
    }
    try:
        torch.save(checkpoint_payload, checkpoint_path)
        run_metadata["last_checkpoint_write"] = checkpoint_payload["checkpoint_created_at"]
        run_metadata["last_completed_episode"] = next_episode - 1
        run_metadata["next_episode"] = next_episode
        run_metadata["metrics_rows"] = len(metrics_rows)
        dump_json(run_metadata, run_metadata_path)
        logger.info("Checkpoint saved to %s (episode %s)", checkpoint_path, next_episode)
    except OSError as exc:
        logger.warning("Failed to write checkpoint at %s: %s", checkpoint_path, exc)

def serialize_for_json(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, range):
        return list(value)
    if isinstance(value, dict):
        return {k: serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def create_unique_run_dir(parent: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = parent / timestamp
    counter = 1
    while run_path.exists():
        run_path = parent / f"{timestamp}_{counter}"
        counter += 1
    run_path.mkdir()
    return run_path, timestamp


def dump_json(data, path: Path):
    try:
        with path.open('w') as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write JSON file %s: %s", path, exc)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        with path.open('r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read JSON file %s: %s", path, exc)
        return default


def get_constants_dict():
    return {
        name: serialize_for_json(getattr(constants_module, name))
        for name in dir(constants_module)
        if name.isupper()
    }


def snapshot_configuration(run_metadata_dict, params_path_obj, constants_path_obj, metadata_path_obj):
    params_snapshot = {k: serialize_for_json(v) for k, v in params.items()}
    dump_json(params_snapshot, params_path_obj)
    constants_snapshot = get_constants_dict()
    dump_json(constants_snapshot, constants_path_obj)
    run_metadata_dict.update({
        "params_file": str(params_path_obj),
        "constants_file": str(constants_path_obj),
    })
    dump_json(run_metadata_dict, metadata_path_obj)


def load_metrics_csv(path: Path):
    if not path.exists():
        return []
    try:
        with path.open('r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return [dict(row) for row in reader]
    except OSError as exc:
        logger.warning("Failed to read metrics CSV at %s: %s", path, exc)
    return []

def write_metrics_csv(rows, run_metadata_bundle: RunMetadataBundle):
    run_metadata = run_metadata_bundle.run_metadata
    run_metadata_path = run_metadata_bundle.run_metadata_path
    metrics_csv_path = run_metadata_bundle.metrics_csv_path
    if not rows:
        return
    try:
        with metrics_csv_path.open('w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=METRICS_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        run_metadata["last_metrics_write"] = datetime.now().isoformat()
        run_metadata["metrics_rows"] = len(rows)
        dump_json(run_metadata, run_metadata_path)
    except OSError as exc:
        logger.warning("Failed to write metrics CSV at %s: %s", metrics_csv_path, exc)