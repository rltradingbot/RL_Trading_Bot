"""Evaluation utilities for RL trading agents.

This module evaluates a provided agent against two simple baselines on the
`RLTradingEnv` environment:
- Agent policy using `agent.act` with exploration disabled
- Equal-weight allocation across all assets (cash weight is zero)
- BTC-only allocation (falls back to the first asset when BTC is absent)

Transaction costs and slippage are applied by the environment at each step.
The main entrypoint `evaluate_agent` returns summary statistics, per-episode
final returns, and equity curves that can be plotted for analysis.

Usage example:
    summary = evaluate_agent(
        agent=agent,
        data_pre_dict=data_pre_dict,
        dataset_cfg=dataset_cfg,
        env_cfg=env_cfg,
        n_episodes=16,
    )
    print(summary["algo"]["mean_final_return"])  # aggregated metric for agent
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
import numpy as np

from environment.tradingEnv import RLTradingEnv, EnvConfig
from data.binance.dataset import DatasetConfig


def _pack_state_for_buffer(state: Dict[str, Any]) -> torch.Tensor:
    """Pack environment state dict into a (L, D_packed) tensor.

    Concatenates, along the last dimension: features, masks, elapsed time
    features, and portfolio_weights_list (including cash). This matches the
    training-time packing convention expected by the agent.

    Args:
        state: Dict with keys "features", "masks", "elapsed", and
            "portfolio_weights_list".

    Returns:
        A contiguous tensor of shape (L, 3*C + N + 1).
    """
    x = state["features"]
    m = state["masks"]
    t = state["elapsed"]
    pw = state["portfolio_weights_list"]
    return torch.cat([x, m, t, pw], dim=-1).contiguous()


@torch.no_grad()
def evaluate_agent(
    agent: torch.nn.Module,
    data_pre_dict: Dict[str, Dict[str, Any]],
    dataset_cfg: DatasetConfig,
    env_cfg: EnvConfig,
    n_episodes: int = 32,
) -> Dict[str, Any]:
    """Evaluate an agent and baseline policies on `RLTradingEnv`.

    For each episode, a seed environment defines an anchor schedule and three
    runs are executed using identical schedules: agent, equal-weight, and
    BTC-only. At each step, equity compounds gross portfolio returns minus
    step costs, which are produced by the environment.

    Args:
        agent: Model exposing `.act(packed_state)` returning portfolio weights
            of shape (N+1,) on the simplex. If `agent.cfg.epsilon_explore`
            exists, it will be temporarily set to 0.0 during evaluation.
        data_pre_dict: Preprocessed data dictionary consumed by `RLTradingEnv`.
        dataset_cfg: Dataset configuration (symbols, features, windows).
        env_cfg: Environment configuration (costs, slippage, reward settings).
        n_episodes: Number of episodes to evaluate.

    Returns:
        Dict containing summary statistics, per-episode final returns, and
        equity curves for each policy.
    """
    old_eps = None
    if hasattr(agent, "cfg") and hasattr(agent.cfg, "epsilon_explore"):
        old_eps = float(agent.cfg.epsilon_explore)
        agent.cfg.epsilon_explore = 0.0
    agent.eval()

    def _force_same_episode_like(env_ref: RLTradingEnv) -> RLTradingEnv:
        """Create a new env that reproduces `env_ref`'s episode anchor schedule."""
        e = RLTradingEnv(data=data_pre_dict, dataset_cfg=dataset_cfg, env_cfg=env_cfg)
        e.reset()
        e._episode_anchor_idxes = env_ref._episode_anchor_idxes[:]
        e._t_ptr = 0
        e._steps = env_ref._steps
        e._portfolio_now = e._cash_only()
        e._reset_pw_hist_to_cash()
        e.rewarder.reset()
        return e

    def _policy_algo(state_dict: Dict[str, Any], env_local: RLTradingEnv) -> torch.Tensor:
        """Agent policy that packs state and calls `agent.act`."""
        s_packed = _pack_state_for_buffer(state_dict)
        a = agent.act(s_packed)
        return a

    def _policy_equal(state_dict: Dict[str, Any], env_local: RLTradingEnv) -> torch.Tensor:
        """Equal-weight policy across assets; cash weight is zero unless no assets."""
        n = env_local.n_assets
        a = torch.zeros(n + 1, dtype=torch.float32)
        if n > 0:
            a[:-1] = 1.0 / n
        else:
            a[-1] = 1.0
        return a

    def _policy_btc(state_dict: Dict[str, Any], env_local: RLTradingEnv) -> torch.Tensor:
        """BTC-only policy if available; otherwise first asset; if none, cash only."""
        n = env_local.n_assets
        a = torch.zeros(n + 1, dtype=torch.float32)
        idx = 0
        if "BTCUSDT" in env_local.symbols:
            idx = env_local.symbols.index("BTCUSDT")
        if n > 0:
            a[idx] = 1.0
        else:
            a[-1] = 1.0
        return a

    def _run_episode(env_seed_env: RLTradingEnv, policy_fn) -> Tuple[float, List[float]]:
        """Run one episode using `policy_fn` on the cloned anchor schedule.

        Compounds equity by applying gross returns minus step costs. Returns
        the final simple return and the equity curve for the episode.
        """
        env = _force_same_episode_like(env_seed_env)
        equity = 1.0
        eq_curve = [equity]

        state, _ = env._build_state(env._current_anchor_ts()), {}
        done = False
        while not done:
            action = policy_fn(state, env)
            next_state, _, done, info_step = env.step(action)
            gross = float(info_step["portfolio_return_gross"])
            cost = float(info_step["step_cost"])
            net = gross - cost
            equity *= (1.0 + net)
            eq_curve.append(equity)
            state = next_state
        final_ret = equity - 1.0
        return final_ret, eq_curve

    results_algo, results_eq, results_btc = [], [], []
    curves_algo, curves_eq, curves_btc = [], [], []

    for _ in range(n_episodes):
        env_seed = RLTradingEnv(data=data_pre_dict, dataset_cfg=dataset_cfg, env_cfg=env_cfg)
        env_seed.reset()

        r_algo, c_algo = _run_episode(env_seed, _policy_algo)
        r_eq,   c_eq   = _run_episode(env_seed, _policy_equal)
        r_btc,  c_btc  = _run_episode(env_seed, _policy_btc)

        results_algo.append(r_algo)
        results_eq.append(r_eq)
        results_btc.append(r_btc)
        curves_algo.append(c_algo)
        curves_eq.append(c_eq)
        curves_btc.append(c_btc)

    def _summary(arr: List[float]) -> Dict[str, float]:
        """Compute mean, median, and population std of final returns."""
        x = np.array(arr, dtype=np.float64)
        return {
            "mean_final_return": float(x.mean()),
            "median_final_return": float(np.median(x)),
            "std_final_return": float(x.std(ddof=0)),
        }

    summary = {
        "algo": _summary(results_algo),
        "equal": _summary(results_eq),
        "btc": _summary(results_btc),
        "per_episode": {
            "algo": results_algo,
            "equal": results_eq,
            "btc": results_btc,
        },
        "equity_curves": {
            "algo": curves_algo,
            "equal": curves_eq,
            "btc": curves_btc,
        }
    }

    if old_eps is not None:
        agent.cfg.epsilon_explore = old_eps
    agent.train()
    return summary
