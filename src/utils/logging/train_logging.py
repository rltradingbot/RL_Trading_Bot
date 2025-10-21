import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_level(level_str: str) -> int:
    try:
        return getattr(logging, level_str.upper())
    except Exception:
        return logging.INFO


def setup_training_logger(log_dir: Path, level: str = "INFO", run_name: Optional[str] = None) -> Tuple[logging.Logger, Path]:
    """Create a file-based logger for training and return the logger and run directory.

    The logger writes to `<log_dir>/<run_name>/train.log` with real-time flushing.
    If `run_name` is not provided, a timestamp-based name is used.
    """
    if not isinstance(log_dir, Path):
        log_dir = Path(str(log_dir))

    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = log_dir / run_name
    _ensure_dir(run_dir)

    logger = logging.getLogger("training")
    logger.setLevel(_parse_level(level))
    logger.propagate = False

    # Remove existing handlers (useful in notebooks/re-runs)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    log_file = run_dir / "train.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(_parse_level(level))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. run_dir={run_dir}")
    return logger, run_dir


@contextmanager
def log_timing(logger: logging.Logger, label: str, level: int = logging.INFO):
    """Context manager that logs start and end times with elapsed seconds."""
    start = time.perf_counter()
    logger.log(level, f"[START] {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, f"[END] {label} | {elapsed:.3f}s")


@dataclass
class EpisodeMetrics:
    episode_index: int
    steps_in_episode: int
    total_env_steps: int
    episode_reward: float
    # reward component means over the episode
    ret_core_mean: Optional[float] = None
    sigma_down_mean: Optional[float] = None
    Dret_mean: Optional[float] = None
    Treynor_mean: Optional[float] = None
    loss_v_mean: Optional[float] = None
    loss_q_mean: Optional[float] = None
    loss_diff_mean: Optional[float] = None
    q_mean: Optional[float] = None
    v_mean: Optional[float] = None
    updates: int = 0
    replay_size: int = 0
    replay_is_ready: bool = False
    episode_sec: Optional[float] = None
    steps_per_sec: Optional[float] = None


@dataclass
class ValidationMetrics:
    after_episode: int
    algo_mean_final_return: float
    equal_mean_final_return: float
    btc_mean_final_return: float
    duration_sec: Optional[float] = None


@dataclass
class MetricsRecorder:
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    validations: List[ValidationMetrics] = field(default_factory=list)

    def add_episode(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    def add_validation(self, metrics: ValidationMetrics) -> None:
        self.validations.append(metrics)

    def write_csvs(self, out_dir: Path) -> Tuple[Path, Optional[Path]]:
        """Write episode and validation metrics as CSV files in `out_dir`.

        Returns the paths of the written CSV files (validation path may be None).
        """
        _ensure_dir(out_dir)
        ep_path = out_dir / "training_metrics.csv"
        with ep_path.open("w", encoding="utf-8") as f:
            header = [
                "episode_index",
                "steps_in_episode",
                "total_env_steps",
                "episode_reward",
                "ret_core_mean",
                "sigma_down_mean",
                "Dret_mean",
                "Treynor_mean",
                "loss_v_mean",
                "loss_q_mean",
                "loss_diff_mean",
                "q_mean",
                "v_mean",
                "updates",
                "replay_size",
                "replay_is_ready",
                "episode_sec",
                "steps_per_sec",
            ]
            f.write(",".join(header) + "\n")
            for m in self.episodes:
                row = [
                    m.episode_index,
                    m.steps_in_episode,
                    m.total_env_steps,
                    f"{m.episode_reward:.6f}",
                    (f"{m.ret_core_mean:.6f}" if m.ret_core_mean is not None else ""),
                    (f"{m.sigma_down_mean:.6f}" if m.sigma_down_mean is not None else ""),
                    (f"{m.Dret_mean:.6f}" if m.Dret_mean is not None else ""),
                    (f"{m.Treynor_mean:.6f}" if m.Treynor_mean is not None else ""),
                    (f"{m.loss_v_mean:.6f}" if m.loss_v_mean is not None else ""),
                    (f"{m.loss_q_mean:.6f}" if m.loss_q_mean is not None else ""),
                    (f"{m.loss_diff_mean:.6f}" if m.loss_diff_mean is not None else ""),
                    (f"{m.q_mean:.6f}" if m.q_mean is not None else ""),
                    (f"{m.v_mean:.6f}" if m.v_mean is not None else ""),
                    m.updates,
                    m.replay_size,
                    int(bool(m.replay_is_ready)),
                    (f"{m.episode_sec:.6f}" if m.episode_sec is not None else ""),
                    (f"{m.steps_per_sec:.6f}" if m.steps_per_sec is not None else ""),
                ]
                f.write(",".join(map(str, row)) + "\n")

        val_path: Optional[Path] = None
        if len(self.validations) > 0:
            val_path = out_dir / "validation_metrics.csv"
            with val_path.open("w", encoding="utf-8") as f:
                header = [
                    "after_episode",
                    "algo_mean_final_return",
                    "equal_mean_final_return",
                    "btc_mean_final_return",
                    "duration_sec",
                ]
                f.write(",".join(header) + "\n")
                for v in self.validations:
                    row = [
                        v.after_episode,
                        f"{v.algo_mean_final_return:.6f}",
                        f"{v.equal_mean_final_return:.6f}",
                        f"{v.btc_mean_final_return:.6f}",
                        (f"{v.duration_sec:.6f}" if v.duration_sec is not None else ""),
                    ]
                    f.write(",".join(map(str, row)) + "\n")

        return ep_path, val_path


