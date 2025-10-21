# reward.py
"""
Risk-aware reward module for portfolio allocation.

Overview
--------
- Units are simple returns per step (e.g., 0.01 = 1%).
- The environment supplies per-step transaction cost (including slippage).
  This module aggregates costs over a rolling window and uses the average
  cost when computing the core reward terms.
- The benchmark/market return can be defined as BTCUSDT (if present) or
  as the equal-weight mean over all symbols.
- Annualization is optional and controlled by `RewardConfig.annualize` and
  `RewardConfig.periods_per_year`.

Composite reward formula
------------------------
Let:
- rp_gross[t] = portfolio gross return at step t (before costs)
- cost[t] = per-step transaction cost at t (provided by the environment)
- rb[t] = benchmark return at t
- rm[t] = market return at t

We maintain rolling windows of these series with length `window`.
- mu_p_net = mean(rp_gross) - mean(cost)
- mu_b = mean(rb)
- mu_m = mean(rm)
- ret_core = annualize ? (periods_per_year * mu_p_net) : mu_p_net
- sigma_down = sqrt( mean( clip(- (rp_gross - cost), 0, inf)^2 ) )
- beta is estimated via sample covariance/variance with ddof=1 against rm.
  denom = sign(beta) * max(|beta|, beta_epsilon)
- Differential return: Dret = (mu_p_net - mu_b) / denom
- Treynor ratio (Rf=0): Treynor = ret_core / denom

Final reward:
  R = w1 * ret_core - w2 * sigma_down + w3 * Dret + w4 * Treynor

Typical usage
-------------
```python
from environment.reward import RewardConfig, RiskAwareReward
import torch

cfg = RewardConfig(window=256, benchmark_mode="btc", periods_per_year=525600,
                   w1=1.0, w2=1.0, w3=1.0, w4=1.0, annualize=True)
rewarder = RiskAwareReward(cfg, symbols=["BTCUSDT", "ETHUSDT"]) 

rewarder.reset()
R, comps = rewarder.update(
    portfolio_return_gross=0.0012,
    step_cost=0.0003,
    asset_returns=torch.tensor([0.0015, 0.0008], dtype=torch.float32)
)
```
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Dict, Optional
import torch

@dataclass
class RewardConfig:
    """Configuration for `RiskAwareReward`.

    Attributes
    ----------
    w1 : float
        Weight for the core return term (annualized or mean).
    w2 : float
        Weight for the downside risk penalty, computed from net returns.
    w3 : float
        Weight for the differential return term (excess over benchmark scaled by beta).
    w4 : float
        Weight for the Treynor term (core return scaled by beta).
    window : int
        Rolling window length in steps.
    benchmark_mode : str
        "btc" to use BTCUSDT as benchmark/market if present; otherwise "mean" (equal-weight mean).
    periods_per_year : int
        Number of periods per year, used when `annualize=True`.
    beta_epsilon : float
        Small positive value for numerical stability in beta scaling.
    annualize : bool
        If True, annualize the core return; otherwise use mean return.
    """
    # weights
    w1: float = 0.25  # (annualized or mean) return
    w2: float = 0.25   # downside risk penalty
    w3: float = 0.25  # differential return
    w4: float = 0.25  # treynor
    # window length
    window: int = 128
    # benchmark definition: "btc" or "mean"
    benchmark_mode: str = "btc"
    # number of periods per year based on interval (e.g., 1-minute bars = approx 525600)
    periods_per_year: int = 525600
    # numerical stability
    beta_epsilon: float = 1e-6
    # annualize option: True for annualized return/treynor, False for mean return/treynor
    annualize: bool = False

class RiskAwareReward:
    """Risk-aware composite reward.

    This class maintains rolling windows of portfolio gross returns, per-step costs,
    and benchmark/market returns. On each `update`, it computes a composite reward:

        R = w1 * ret_core - w2 * sigma_down + w3 * Dret + w4 * Treynor

    where `ret_core` is annualized or mean net return, `sigma_down` is downside
    deviation of net returns, `Dret` is differential return vs. benchmark scaled by
    beta, and `Treynor` is the Treynor ratio with Rf=0.

    Parameters
    ----------
    cfg : RewardConfig
        Reward configuration, including weights, window length, benchmark mode, and
        annualization options.
    symbols : List[str]
        The trading symbols, used to locate the BTCUSDT index if `benchmark_mode="btc"`.
    """
    def __init__(self, cfg: RewardConfig, symbols: List[str]):
        self.cfg = cfg
        self.symbols = symbols

        self._btc_index: Optional[int] = None
        if cfg.benchmark_mode.lower() == "btc":
            sym_upper = [s.upper() for s in symbols]
            self._btc_index = sym_upper.index("BTCUSDT") if "BTCUSDT" in sym_upper else None

        # record: total return (rp_gross), cost, benchmark/market return (rb, rm)
        self._rp_gross = deque(maxlen=self.cfg.window)
        self._costs    = deque(maxlen=self.cfg.window)
        self._rb       = deque(maxlen=self.cfg.window)
        self._rm       = deque(maxlen=self.cfg.window)

    def reset(self):
        """Clear all rolling buffers to start a fresh episode or evaluation run."""
        self._rp_gross.clear()
        self._costs.clear()
        self._rb.clear()
        self._rm.clear()

    def _bench_and_market_return(self, asset_returns: torch.Tensor) -> Tuple[float, float]:
        """Compute benchmark and market returns for the current step.

        If `benchmark_mode` is "btc" and BTCUSDT exists in `symbols`, both the
        benchmark and market returns are set to BTCUSDT's return. Otherwise, both
        are the equal-weight mean over all assets.

        Parameters
        ----------
        asset_returns : torch.Tensor
            Simple returns for each asset at the current step, shape (N,), CPU float32.

        Returns
        -------
        Tuple[float, float]
            (rb, rm) benchmark and market returns.
        """
        if self.cfg.benchmark_mode.lower() == "btc" and self._btc_index is not None:
            rb = float(asset_returns[self._btc_index].item())
            rm = rb
        else:
            mean_ret = float(asset_returns.mean().item()) if asset_returns.numel() > 0 else 0.0
            rb, rm = mean_ret, mean_ret
        return rb, rm

    def update(
        self,
        portfolio_return_gross: float,   # total return before transaction cost
        step_cost: float,                # current step transaction cost (including slippage)
        asset_returns: torch.Tensor      # simple returns per asset (N,)
    ) -> Tuple[float, Dict[str, float]]:
        """Ingest one step and compute the composite reward.

        Parameters
        ----------
        portfolio_return_gross : float
            Portfolio gross return before subtracting costs, for this step.
        step_cost : float
            Transaction cost (including slippage) for this step.
        asset_returns : np.ndarray
            Simple returns per asset at this step, shape (N,). Used to compute
            benchmark/market returns for beta and differential terms.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            A pair of (reward_value, components_dict), where components include:
            - ret_core, sigma_down, Dret, Treynor, beta, denom,
              mu_p_net, mu_b, cost_mean, cum_cost_window.
        """

        rb_t, rm_t = self._bench_and_market_return(asset_returns)

        self._rp_gross.append(float(portfolio_return_gross))
        self._costs.append(float(step_cost))
        self._rb.append(float(rb_t))
        self._rm.append(float(rm_t))

        n = len(self._rp_gross)
        if n == 0:
            return 0.0, {
                "ret_core": 0.0, "sigma_down": 0.0, "Dret": 0.0, "Treynor": 0.0,
                "beta": 0.0, "denom": self.cfg.beta_epsilon,
                "mu_p_net": 0.0, "mu_b": 0.0, "cost_mean": 0.0, "cum_cost_window": 0.0
            }

        # Convert buffers to CPU torch tensors
        rp_g = torch.tensor(list(self._rp_gross), dtype=torch.float32)
        cs   = torch.tensor(list(self._costs),    dtype=torch.float32)
        rb   = torch.tensor(list(self._rb),       dtype=torch.float32)
        rm   = torch.tensor(list(self._rm),       dtype=torch.float32)

        # cumulative/mean cost in the window
        cum_cost = cs.sum()
        cost_mean = cum_cost / float(n)

        # net return series (for downside)
        rp_net_series = rp_g - cs

        # means
        mu_p_net = rp_g.mean() - cost_mean
        mu_b     = rb.mean()
        mu_m     = rm.mean()

        # ret_core: annualize option
        if self.cfg.annualize:
            ret_core = float(self.cfg.periods_per_year) * mu_p_net
        else:
            ret_core = mu_p_net

        # downside deviation (based on net return)
        sigma_down = torch.sqrt(torch.mean(torch.clamp(-rp_net_series, min=0.0) ** 2))

        # β estimation (sample covariance/variance, ddof=1). If sample size < 2, β=0
        if n < 2:
            beta_hat_t = torch.tensor(0.0, dtype=torch.float32)
        else:
            var_rm_t = torch.var(rm, unbiased=True)
            if (var_rm_t <= 0.0) or (not torch.isfinite(var_rm_t)):
                beta_hat_t = torch.tensor(0.0, dtype=torch.float32)
            else:
                cov_pr_t = ((rp_g - rp_g.mean()) * (rm - mu_m)).sum() / float(n - 1)
                beta_hat_t = cov_pr_t / var_rm_t

        # numerical stability: denom = sign(β)*max(|β|, ε)
        beta_hat = float(beta_hat_t.item())
        sign_beta = 1.0 if beta_hat >= 0.0 else -1.0
        denom = sign_beta * max(abs(beta_hat), float(self.cfg.beta_epsilon))

        # Differential Return / Treynor
        Dret = (mu_p_net - mu_b) / denom
        Treynor = ret_core / denom  # assume R_f = 0

        # final reward
        R = (float(self.cfg.w1) * float(ret_core)
             - float(self.cfg.w2) * float(sigma_down)
             + float(self.cfg.w3) * float(Dret)
             + float(self.cfg.w4) * float(Treynor))

        comps = {
            "ret_core": float(ret_core),
            "sigma_down": float(sigma_down),
            "Dret": float(Dret),
            "Treynor": float(Treynor),
            "beta": float(beta_hat),
            "denom": float(denom),
            "mu_p_net": float(mu_p_net.item() if isinstance(mu_p_net, torch.Tensor) else mu_p_net),
            "mu_b": float(mu_b.item() if isinstance(mu_b, torch.Tensor) else mu_b),
            "cost_mean": float(cost_mean.item() if isinstance(cost_mean, torch.Tensor) else cost_mean),
            "cum_cost_window": float(cum_cost.item() if isinstance(cum_cost, torch.Tensor) else cum_cost),
        }
        return float(R), comps
