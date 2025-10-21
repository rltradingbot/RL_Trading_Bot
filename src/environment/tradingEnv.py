# tradingEnv.py
"""
Reinforcement learning trading environment with risk-aware reward.

Overview
--------
- Backend: `BinanceVisionDataset` provides feature windows anchored at timestamps.
- State dict contains CPU float32 tensors:
  - features (L, C): engineered features
  - masks (L, C): 1.0 if present, 0.0 otherwise
  - elapsed (L, C): log1p minutes since last observation per channel
  - portfolio_weights_list (L, N+1): history of portfolio weights, last column is cash/USDT
- Action: long-only allocation of length N+1 that sums to 1.0; last slot is cash (USDT).
- Reward: computed by `RiskAwareReward` using portfolio gross return, per-step costs
  sampled from a Gaussian model (mu, sigma), and benchmark/market returns.
- All inputs/outputs are CPU tensors; move to GPU externally if needed.

Costs and slippage
------------------
- Per-step cost is sampled as `max(N(mu, sigma), 0)` and applied with the symmetric
  half-turnover convention: cost_t = 0.5 * cost * L1(action_t - portfolio_t).
- Units match simple returns (e.g., 0.002 = 0.2%).

Usage
-----
```python
from environment.tradingEnv import RLTradingEnv, EnvConfig
from data.binance.dataset import DatasetConfig

dataset_cfg = DatasetConfig(window_size=128, base_interval="1m", symbol_order=symbols)
env_cfg = EnvConfig(
    symbols=symbols,
    min_steps=512,
    max_steps=4096,
    transaction_cost_mu=0.0024,
    transaction_cost_sigma=0.0005,
    reward_window=128,
    reward_weights=(0.25, 0.25, 0.25, 0.25),
    benchmark_mode="btc",
    reward_annualize=False,
)
env = RLTradingEnv(data=data, dataset_cfg=dataset_cfg, env_cfg=env_cfg)

state, info = env.reset()
done = False
while not done:
    action = env.sample_random_action()
    next_state, reward, done, step_info = env.step(action)
    state = next_state
```
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Union
import random
import math
import pandas as pd
import torch
import numpy as np
from collections import deque

from data.binance.dataset import BinanceVisionDataset, DatasetConfig
from data.binance.constants import interval_to_minutes
from environment.reward import RewardConfig, RiskAwareReward

ArrayLike = Iterable[float]

@dataclass
class EnvConfig:
    """Configuration for `RLTradingEnv`.

    Attributes
    ----------
    symbols : List[str]
        Trading symbols, e.g., ["BTCUSDT", "ETHUSDT"].
    min_steps, max_steps : int
        Episode length bounds in steps. The environment samples a random length in this range.
    device : str
        Always enforced to CPU internally.
    seed : Optional[int]
        Optional RNG seed for reproducibility.
    transaction_cost_mu, transaction_cost_sigma : Optional[float]
        Gaussian cost parameters; units are simple returns. If either is None, cost=0.
    reward_window : int
        Rolling window length for the reward aggregator.
    reward_weights : Optional[Tuple[float, float, float, float]]
        Weights (w1, w2, w3, w4) for reward components.
    benchmark_mode : str
        "btc" to use BTCUSDT as benchmark/market if present; otherwise "mean".
    reward_annualize : bool
        If True, reward uses annualized returns; otherwise mean returns.
    """
    symbols: List[str]
    min_steps: int
    max_steps: int
    device: str = "cpu"
    seed: Optional[int] = None
    transaction_cost_mu: Optional[float] = None
    transaction_cost_sigma: Optional[float] = None
    reward_window: int = 128
    reward_weights: Optional[Tuple[float, float, float, float]] = None
    benchmark_mode: str = "btc"
    reward_annualize: bool = False

class RLTradingEnv:
    def __init__(self,
                 data: Dict[str, Dict[str, pd.DataFrame]],
                 dataset_cfg: DatasetConfig,
                 env_cfg: EnvConfig):
        if env_cfg.seed is not None:
            random.seed(env_cfg.seed)
            np.random.seed(env_cfg.seed)
            torch.manual_seed(env_cfg.seed)

        self.cfg = env_cfg
        self.dataset = BinanceVisionDataset(data=data, cfg=dataset_cfg)

        self.device = torch.device("cpu")
        self.symbols = env_cfg.symbols[:]
        self.n_assets = len(self.symbols)
        self.n_weights = self.n_assets + 1
        self.base_interval: str = self.dataset.base_interval

        # reward config
        iv_minutes = interval_to_minutes(self.base_interval)
        per_year = max(1, int(round((365 * 24 * 60) / max(1, iv_minutes))))  # assume 24/7
        w = env_cfg.reward_weights if env_cfg.reward_weights is not None else (1.0, 1.0, 1.0, 1.0)
        r_cfg = RewardConfig(
            w1=float(w[0]), w2=float(w[1]), w3=float(w[2]), w4=float(w[3]),
            window=int(env_cfg.reward_window),
            benchmark_mode=str(env_cfg.benchmark_mode).lower(),
            periods_per_year=per_year,
            beta_epsilon=1e-6,
            annualize=bool(env_cfg.reward_annualize),
        )
        self.rewarder = RiskAwareReward(cfg=r_cfg, symbols=self.symbols)

        # close price series for reward calculation
        self._close_series: Dict[str, pd.Series] = {}
        for s in self.symbols:
            if s in data and self.base_interval in data[s] and "close" in data[s][self.base_interval].columns:
                self._close_series[s] = data[s][self.base_interval]["close"].astype(float).copy()
            else:
                self._close_series[s] = pd.Series(dtype=float)

        # episode state
        self._episode_anchor_idxes: List[int] = []
        self._steps: int = 0
        self._t_ptr: int = 0

        # portfolio state
        self._portfolio_now: torch.Tensor = self._cash_only()
        self._pw_hist: deque = deque(maxlen=self.dataset.cfg.window_size)

        self.window_size = self.dataset.cfg.window_size
        self._reset_pw_hist_to_cash()

    # ---------- Public ----------
    def reset(self) -> Tuple[dict, dict]:
        self._steps = random.randint(self.cfg.min_steps, self.cfg.max_steps)

        total = len(self.dataset)
        max_start = max(0, total - (self._steps + 1))
        if max_start <= 0:
            self._steps = max(1, min(self._steps, max(1, total - 1)))
            max_start = max(0, total - (self._steps + 1))
        start = random.randint(0, max_start) if max_start > 0 else 0
        self._episode_anchor_idxes = list(range(start, start + self._steps + 1))
        self._t_ptr = 0

        self._portfolio_now = self._cash_only()
        self._reset_pw_hist_to_cash()
        self.rewarder.reset()

        state = self._build_state(self._current_anchor_ts())
        info = {"t": self._current_anchor_ts(),
                "episode_steps": self._steps,
                "start_anchor_index": start}
        return state, info

    def step(self, action: Union[ArrayLike, np.ndarray, torch.Tensor]) -> Tuple[dict, float, bool, dict]:
        with torch.no_grad():
            action_w = self._sanitize_action(action)  # (N+1,)

            t_now = self._current_anchor_ts()
            t_next = self._next_anchor_ts()
            asset_returns = self._asset_simple_returns(t_now, t_next)  # (N,)

            # total portfolio return (assuming cash 0%)
            port_ret_gross = torch.dot(action_w[:-1], asset_returns).item()

            # transaction cost/slippage (half-turnover)
            step_cost = 0.0
            eff_cost = self._effective_transaction_cost()
            if eff_cost > 0.0:
                turnover = torch.sum(torch.abs(action_w - self._portfolio_now)).item()
                step_cost = 0.5 * float(eff_cost) * turnover

            # update reward (cumulative cost is handled internally by the reward module)
            reward_value, comps = self.rewarder.update(
                portfolio_return_gross=float(port_ret_gross),
                step_cost=float(step_cost),
                asset_returns=asset_returns
            )

            # portfolio roll
            self._portfolio_now = action_w.detach().clone()
            self._advance_pw_hist(self._portfolio_now)

            self._t_ptr += 1
            done = (self._t_ptr >= self._steps)

            state = self._build_state(self._current_anchor_ts())
            info = {
                "t": t_now,
                "t_next": t_next,
                "portfolio_return_gross": float(port_ret_gross),
                "step_cost": float(step_cost),
                "done": bool(done),
                "reward_components": comps,
                "benchmark_mode": self.rewarder.cfg.benchmark_mode,
                "annualize": self.rewarder.cfg.annualize,
            }
        return state, float(reward_value), bool(done), info

    # ---------- Helpers ----------
    def _cash_only(self) -> torch.Tensor:
        w = torch.zeros(self.n_weights, dtype=torch.float32, device=self.device)
        w[-1] = 1.0  # USDT
        return w

    def _reset_pw_hist_to_cash(self):
        self._pw_hist.clear()
        base = self._cash_only()
        for _ in range(self.window_size):
            self._pw_hist.append(base.detach().clone())

    def _advance_pw_hist(self, new_w: torch.Tensor):
        self._pw_hist.append(new_w.detach().clone())

    def _ensure_cpu_f32(self, x: Union[np.ndarray, torch.Tensor, ArrayLike]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=self.device, dtype=torch.float32).contiguous()
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).contiguous()

    def _sanitize_action(self, action: Union[ArrayLike, np.ndarray, torch.Tensor]) -> torch.Tensor:
        a = self._ensure_cpu_f32(action).view(-1)
        if a.numel() != self.n_weights:
            raise ValueError(f"action length {a.numel()} does not match expected {self.n_weights}")
        if not torch.isfinite(a).all():
            raise ValueError("action contains NaN or Inf")
        if (a < 0.0).any():
            raise ValueError("action contains negative values")
        s = float(a.sum().item())
        if not math.isfinite(s) or abs(s - 1.0) > 1e-5:
            raise ValueError("action sum must be 1.0")
        return a

    def _current_anchor_ts(self) -> pd.Timestamp:
        idx = self._episode_anchor_idxes[self._t_ptr]
        return self.dataset.anchors[idx]

    def _next_anchor_ts(self) -> pd.Timestamp:
        idx = self._episode_anchor_idxes[self._t_ptr + 1]
        return self.dataset.anchors[idx]

    def _asset_simple_returns(self, t0: pd.Timestamp, t1: pd.Timestamp) -> torch.Tensor:
        rets = [0.0] * self.n_assets
        for i, s in enumerate(self.symbols):
            ser = self._close_series.get(s)
            if ser is None or ser.empty:
                rets[i] = 0.0
                continue
            c0 = ser.get(t0, np.nan)
            c1 = ser.get(t1, np.nan)
            if math.isfinite(c0) and math.isfinite(c1) and c0 > 0:
                rets[i] = float(c1 / c0 - 1.0)
            else:
                rets[i] = 0.0
        return torch.tensor(rets, dtype=torch.float32, device=self.device)

    def _effective_transaction_cost(self) -> float:
        mu = getattr(self.cfg, "transaction_cost_mu", None)
        sigma = getattr(self.cfg, "transaction_cost_sigma", None)
        if mu is None or sigma is None:
            return 0.0
        mean_t = torch.tensor(float(mu), dtype=torch.float32)
        std_t = torch.tensor(float(sigma), dtype=torch.float32)
        sampled_t = torch.normal(mean=mean_t, std=std_t)
        sampled = float(sampled_t.item())
        if not math.isfinite(sampled):
            sampled = float(mu)
        if sampled < 0.0:
            sampled = 0.0
        return sampled

    def _build_state(self, t_anchor: pd.Timestamp) -> dict:
        X, M, T, meta = self.dataset[t_anchor]
        X = self._ensure_cpu_f32(X)
        M = self._ensure_cpu_f32(M)
        T = self._ensure_cpu_f32(T)
        PW = torch.stack(list(self._pw_hist), dim=0)
        return {
            "features": X,
            "masks": M,
            "elapsed": T,
            "portfolio_weights_list": PW,
            "meta": {"anchor_time": t_anchor, "combos": meta["combos"], "feature_names_6": meta["feature_names_6"]}
        }

    @property
    def action_size(self) -> int:
        return self.n_weights

    @property
    def window_C(self) -> int:
        return self.dataset.total_feature_channels

    def sample_random_action(self) -> torch.Tensor:
        dist = torch.distributions.Dirichlet(torch.ones(self.n_weights, dtype=torch.float32, device=self.device))
        a = dist.sample()
        return a.to(self.device)


__all__ = ["RLTradingEnv"]
