"""
Training entrypoint for the portfolio reinforcement learning experiment.

Overview
--------
- Load Binance Vision monthly CSVs, preprocess to a regular grid, and construct a Dataset.
- Create an RLTradingEnv with risk-aware reward and Gaussian transaction costs.
- Initialize an IDQL agent (IQL critics + diffusion behavior model) and a replay buffer.
- Iterate episodes: act, step, store transitions, periodically update the agent.

All environment inputs/outputs are CPU tensors. The agent moves batches to the
configured device internally during training.
"""
import torch
from typing import Dict, Any
from pathlib import Path
import yaml
import time
import sys

from utils.logging.train_logging import (
    setup_training_logger,
    log_timing,
    MetricsRecorder,
    EpisodeMetrics,
    ValidationMetrics,
)

from data.binance.load_csv import load_vision_monthlies_csv
from data.binance.preprocess import PreprocessPipeline
from data.binance.dataset import DatasetConfig

from environment.tradingEnv import RLTradingEnv, EnvConfig
from utils.replay_buffer.replay_buffer import ReplayBuffer

# NEW: import IDQL agent
from models.IDQL import IDQLAgent, IDQLConfig

from environment.validation import evaluate_agent


def move_state_to_device(state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor values to `device`, keeping metadata on CPU."""
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k == "meta":
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def pack_state_for_buffer(state: Dict[str, Any]) -> torch.Tensor:
    """Pack state into a single CPU tensor for replay storage.

    Concatenate features, masks, elapsed, and portfolio weights along the last
    dimension to preserve the time axis L.
    """
    x = state["features"]
    m = state["masks"]
    t = state["elapsed"]
    pw = state["portfolio_weights_list"]
    packed = torch.cat([x, m, t, pw], dim=-1).contiguous()
    return packed.detach().to("cpu")


def save_agent_checkpoint(agent: torch.nn.Module, path: Path, episode: int, total_env_steps: int) -> None:
    """Save minimal checkpoint containing agent weights and training counters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": int(episode),
        "total_env_steps": int(total_env_steps),
        "state_dict": agent.state_dict(),
    }
    torch.save(payload, str(path))


if __name__ == "__main__":
    # 0) Load configuration from YAML
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "train_config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # 0.1) Setup logging and metrics recorder
    log_cfg = config["logging"]
    log_root = Path(log_cfg["root_dir"])
    log_level = str(log_cfg["level"])
    run_name = log_cfg["run_name"]
    logger, run_dir = setup_training_logger(log_root, level=log_level, run_name=run_name)
    metrics = MetricsRecorder()

    val_cfg = config["validation"]
    do_validation = bool(val_cfg["enabled"])
    validate_every_episodes = int(val_cfg["every_episodes"])
    n_val_episodes = int(val_cfg["n_episodes"])
    val_start_month = str(val_cfg["start_month"])
    val_end_month = str(val_cfg["end_month"])

    # Checkpoint configuration
    ckpt_cfg = config["checkpoint"]
    save_checkpoints = bool(ckpt_cfg["enabled"])
    ckpt_root = Path(ckpt_cfg["dir"])
    ckpt_run_dir = None
    if save_checkpoints:
        ckpt_run_dir = ckpt_root / run_dir.name
        ckpt_run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and preprocess raw data
    with log_timing(logger, "Data loading (train)"):
        symbols = list(config["symbols"])
        train_start_month = str(config["train_start_month"])
        train_end_month = str(config["train_end_month"])
        binance_kline_dataset_root = str(config["binance_kline_dataset_root"])

        train_data_raw_dict: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            train_data_raw_dict[symbol] = load_vision_monthlies_csv(
                symbol, train_start_month, train_end_month, binance_kline_dataset_root
            )

    preprocessing_pipline = PreprocessPipeline(list(config["preprocessing_pipeline"]))

    with log_timing(logger, "Preprocess (train)"):
        train_data_pre_dict: Dict[str, Dict[str, Any]] = {}
        for sym, iv_map in train_data_raw_dict.items():
            train_data_pre_dict[sym] = {}
            for iv, df in iv_map.items():
                df_prep = preprocessing_pipline(df, iv)
                train_data_pre_dict[sym][iv] = df_prep
    
    with log_timing(logger, "Data loading (validation)"):
        val_data_raw_dict: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            val_data_raw_dict[symbol] = load_vision_monthlies_csv(
                symbol, val_start_month, val_end_month, binance_kline_dataset_root
            )

    with log_timing(logger, "Preprocess (validation)"):
        val_data_pre_dict: Dict[str, Dict[str, Any]] = {}
        for sym, iv_map in val_data_raw_dict.items():
            val_data_pre_dict[sym] = {}
            for iv, df in iv_map.items():
                df_prep = preprocessing_pipline(df, iv)
                val_data_pre_dict[sym][iv] = df_prep

    # 2) Build dataset and environment configuration
    window_size = int(config["window_size"])
    base_interval = str(config["base_interval"])  # e.g., "1m"
    strict_anchor = bool(config["strict_anchor"])  # dataset setting
    dataset_cfg = DatasetConfig(
        window_size=window_size,
        strict_anchor=strict_anchor,
        symbol_order=symbols,
        base_interval=base_interval,
    )

    random_seed = int(config["random_seed"])  # for env and replay buffer

    slippage_mu = float(config["slippage_mu"])
    slippage_sigma = float(config["slippage_sigma"])
    env_cfg = EnvConfig(
        symbols=symbols,
        min_steps=int(config["env"]["min_steps"]),
        max_steps=int(config["env"]["max_steps"]),
        transaction_cost_mu=slippage_mu,
        transaction_cost_sigma=slippage_sigma,
        reward_window=window_size,
        reward_weights=tuple(config["env"]["reward_weights"]),
        reward_annualize=bool(config["env"]["reward_annualize"]),
        seed=random_seed,
    )

    with log_timing(logger, "Env init"):
        env = RLTradingEnv(data=train_data_pre_dict, dataset_cfg=dataset_cfg, env_cfg=env_cfg)

    # 3) Initialize device and replay buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_capacity = int(config["replay_buffer"]["capacity"])   # 65536
    replay_ready_after = int(config["replay_buffer"]["ready_after"])  # 8192
    replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=random_seed, ready_after=replay_ready_after)

    # 4) Training hyperparameters
    batch_size = int(config["training"]["batch_size"])  # 128
    min_steps_for_update_replay = int(config["training"]["min_steps_for_update_replay"])  # 4096
    train_iters_per_update = int(config["training"]["train_iters_per_update"])  # 256
    max_episodes = int(config["training"]["max_episodes"])  # 8192
    max_total_env_steps = int(config["training"]["max_total_env_steps"])  # 16777216

    # 5) Initialize IDQL agent (infer packed dim via a dry reset)
    state0, _ = env.reset()
    L = window_size
    packed0 = pack_state_for_buffer(state0)  # (L, D_packed)
    D_packed = packed0.size(-1)
    action_dim = env.action_size  # N+1

    idql_cfg = IDQLConfig(
        action_dim=action_dim,
        per_step_state_dim=D_packed,
        window_size=L,
        state_emb_dim=int(config["idql"]["state_emb_dim"]),
        hidden=int(config["idql"]["hidden"]),
        gamma=float(config["idql"]["gamma"]),
        tau_expectile=float(config["idql"]["tau_expectile"]),
        lr=float(config["idql"]["lr"]),
        target_ema=float(config["idql"]["target_ema"]),
        diff_T=int(config["idql"]["diff_T"]),
        diff_time_dim=int(config["idql"]["diff_time_dim"]),
        diff_blocks=int(config["idql"]["diff_blocks"]),
        diff_samples_per_state=int(config["idql"]["diff_samples_per_state"]),
        diff_beta_min=float(config["idql"]["diff_beta_min"]),
        diff_beta_max=float(config["idql"]["diff_beta_max"]),
        diff_beta_schedule=str(config["idql"]["diff_beta_schedule"]),
        diff_dropout=float(config["idql"].get("diff_dropout", 0.1)),
        policy_ema=float(config["idql"].get("policy_ema", 0.01)),
        epsilon_explore=float(config["idql"]["epsilon_explore"]),
    )

    with log_timing(logger, "Agent init"):
        agent = IDQLAgent(cfg=idql_cfg, device=device).to(device)

    # Reset again to start training loop cleanly
    state_cpu, info = env.reset()
    total_env_steps = 0
    steps_since_update = 0

    logger.info("Starting training loop")

    try:
        for episode in range(1, max_episodes + 1):
            ep_start = time.perf_counter()
            # Start a new episode
            state_cpu, info = env.reset()
            done = False
            ep_steps = 0
            ep_reward_sum = 0.0
            # reward component running sums
            ret_core_sum = 0.0
            sigma_down_sum = 0.0
            Dret_sum = 0.0
            Treynor_sum = 0.0
            ep_updates = 0
            # running means for losses
            loss_v_acc = 0.0
            loss_q_acc = 0.0
            loss_diff_acc = 0.0
            q_mean_acc = 0.0
            v_mean_acc = 0.0

            while not done:
                s_cpu_packed = pack_state_for_buffer(state_cpu)  # (L, D_packed)
                with torch.no_grad():
                    action_cpu = agent.act(s_cpu_packed)  # (A,) CPU simplex

                # Environment step
                next_state_cpu, reward, done, step_info = env.step(action_cpu)
                comps = step_info.get("reward_components", {})
                ret_core_sum += float(comps.get("ret_core", 0.0))
                sigma_down_sum += float(comps.get("sigma_down", 0.0))
                Dret_sum += float(comps.get("Dret", 0.0))
                Treynor_sum += float(comps.get("Treynor", 0.0))

                # Store transition (CPU)
                ns_cpu_packed = pack_state_for_buffer(next_state_cpu)
                r_cpu = torch.tensor([reward], dtype=torch.float32)
                d_cpu = torch.tensor([done], dtype=torch.bool)

                # action already CPU; ensure contiguous
                a_cpu = action_cpu.detach().to("cpu").contiguous()

                replay_buffer.push(s_cpu_packed, a_cpu, r_cpu, ns_cpu_packed, d_cpu)

                total_env_steps += 1
                steps_since_update += 1
                ep_steps += 1
                ep_reward_sum += float(reward)
                state_cpu = next_state_cpu

                # Periodic training burst
                if replay_buffer.is_ready() and steps_since_update >= min_steps_for_update_replay:
                    t0 = time.perf_counter()
                    burst_loss_v = 0.0
                    burst_loss_q = 0.0
                    burst_loss_diff = 0.0
                    burst_q_mean = 0.0
                    burst_v_mean = 0.0
                    for _ in range(train_iters_per_update):
                        batch = replay_buffer.sample(batch_size, replace=False)  # CPU tensors
                        if device.type == "cuda":
                            batch = batch.pin_memory().to(device, non_blocking=True)
                        else:
                            batch = batch.to(device)
                        states_b, actions_b, rewards_b, next_states_b, dones_b = batch.as_tuple()
                        losses = agent.update((states_b, actions_b, rewards_b, next_states_b, dones_b))
                        # accumulate
                        lv = float(losses["loss_v"])
                        lq = float(losses["loss_q"])
                        ld = float(losses["loss_diff"])
                        qm = float(losses["q_mean"])
                        vm = float(losses["v_mean"])
                        loss_v_acc += lv
                        loss_q_acc += lq
                        loss_diff_acc += ld
                        q_mean_acc += qm
                        v_mean_acc += vm
                        burst_loss_v += lv
                        burst_loss_q += lq
                        burst_loss_diff += ld
                        burst_q_mean += qm
                        burst_v_mean += vm
                        ep_updates += 1
                    t1 = time.perf_counter()
                    denom = float(train_iters_per_update)
                    logger.info(
                        f"Update burst | iters={train_iters_per_update} time={t1 - t0:.3f}s "
                        f"replay_size={replay_buffer.size()} "
                        f"loss_v_mean={burst_loss_v/denom:.6f} loss_q_mean={burst_loss_q/denom:.6f} "
                        f"loss_diff_mean={burst_loss_diff/denom:.6f} q_mean={burst_q_mean/denom:.6f} v_mean={burst_v_mean/denom:.6f}"
                    )
                    steps_since_update = 0

                if total_env_steps >= max_total_env_steps:
                    done = True
                    break

            # Episode summary
            ep_dur = time.perf_counter() - ep_start
            steps_per_sec = (ep_steps / ep_dur) if ep_dur > 0 else 0.0
            loss_v_mean = (loss_v_acc / ep_updates) if ep_updates > 0 else None
            loss_q_mean = (loss_q_acc / ep_updates) if ep_updates > 0 else None
            loss_diff_mean = (loss_diff_acc / ep_updates) if ep_updates > 0 else None
            q_mean = (q_mean_acc / ep_updates) if ep_updates > 0 else None
            v_mean = (v_mean_acc / ep_updates) if ep_updates > 0 else None

            denom_steps = float(ep_steps) if ep_steps > 0 else 1.0
            em = EpisodeMetrics(
                episode_index=episode,
                steps_in_episode=ep_steps,
                total_env_steps=total_env_steps,
                episode_reward=ep_reward_sum,
                ret_core_mean=(ret_core_sum / denom_steps) if ep_steps > 0 else None,
                sigma_down_mean=(sigma_down_sum / denom_steps) if ep_steps > 0 else None,
                Dret_mean=(Dret_sum / denom_steps) if ep_steps > 0 else None,
                Treynor_mean=(Treynor_sum / denom_steps) if ep_steps > 0 else None,
                loss_v_mean=loss_v_mean,
                loss_q_mean=loss_q_mean,
                loss_diff_mean=loss_diff_mean,
                q_mean=q_mean,
                v_mean=v_mean,
                updates=ep_updates,
                replay_size=replay_buffer.size(),
                replay_is_ready=replay_buffer.is_ready(),
                episode_sec=ep_dur,
                steps_per_sec=steps_per_sec,
            )
            metrics.add_episode(em)
            rc = f"{em.ret_core_mean:.6f}" if em.ret_core_mean is not None else "NA"
            sd = f"{em.sigma_down_mean:.6f}" if em.sigma_down_mean is not None else "NA"
            dr = f"{em.Dret_mean:.6f}" if em.Dret_mean is not None else "NA"
            tr = f"{em.Treynor_mean:.6f}" if em.Treynor_mean is not None else "NA"
            logger.info(
                f"Episode {episode} | steps={ep_steps} total_steps={total_env_steps} "
                f"reward_sum={ep_reward_sum:.6f} ret_core={rc} sigma_down={sd} Dret={dr} Treynor={tr} "
                f"updates={ep_updates} dur={ep_dur:.2f}s ({steps_per_sec:.2f} steps/s)"
            )

            if total_env_steps >= max_total_env_steps:
                logger.info("Reached max_total_env_steps. Stopping training.")
                break

            # Validation
            if do_validation and (episode % validate_every_episodes == 0):
                # Save checkpoint alongside validation
                ckpt_path = None
                if save_checkpoints and ckpt_run_dir is not None:
                    ckpt_path = ckpt_run_dir / f"episode_{episode:06d}_steps_{total_env_steps:09d}.pt"
                    save_agent_checkpoint(agent, ckpt_path, episode, total_env_steps)
                    logger.info(f"[CKPT] Saved checkpoint: {ckpt_path}")

                t_val0 = time.perf_counter()
                logger.info(
                    f"[VAL] After episode {episode} period={val_start_month}~{val_end_month} running {n_val_episodes} episodes"
                    + (f" | ckpt={ckpt_path}" if ckpt_path is not None else "")
                )

                # Evaluate the saved model (reload to ensure checkpoint integrity),
                # fall back to in-memory agent if saving is disabled
                agent_for_eval = agent
                if ckpt_path is not None:
                    agent_for_eval = IDQLAgent(cfg=idql_cfg, device=device).to(device)
                    try:
                        ckpt_obj = torch.load(str(ckpt_path), map_location=device)
                        agent_for_eval.load_state_dict(ckpt_obj["state_dict"])
                    except Exception as ex:
                        logger.warning(f"[CKPT] Failed to load checkpoint for eval ({ex}). Using in-memory agent.")
                        agent_for_eval = agent

                val_summary = evaluate_agent(
                    agent=agent_for_eval,
                    data_pre_dict=val_data_pre_dict,
                    dataset_cfg=dataset_cfg,
                    env_cfg=env_cfg,
                    n_episodes=n_val_episodes,
                )
                t_val1 = time.perf_counter()
                a = float(val_summary["algo"]["mean_final_return"])
                e = float(val_summary["equal"]["mean_final_return"])
                b = float(val_summary["btc"]["mean_final_return"])
                logger.info(
                    f"[VAL] mean_final_return | ALGO={a:.6f}  EQUAL={e:.6f}  BTC={b:.6f} | time={t_val1 - t_val0:.3f}s"
                    + (f" | ckpt={ckpt_path}" if ckpt_path is not None else "")
                )
                vm = ValidationMetrics(
                    after_episode=episode,
                    algo_mean_final_return=a,
                    equal_mean_final_return=e,
                    btc_mean_final_return=b,
                    duration_sec=(t_val1 - t_val0),
                )
                metrics.add_validation(vm)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Finalizing and saving metrics...")
    except Exception as ex:
        logger.exception(f"Unhandled exception during training: {ex}")
        raise
    finally:
        # Always write CSVs at the end
        csv_dir = run_dir
        ep_csv, val_csv = metrics.write_csvs(csv_dir)
        logger.info(f"Metrics CSV written: episodes={ep_csv}{' validations=' + str(val_csv) if val_csv else ''}")
