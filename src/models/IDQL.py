"""
Implicit Q-Learning agent with diffusion behavior policy for portfolio allocation.

Overview:
- Value network trained via expectile regression on target Q.
- Q network trained with TD target bootstrapped from V.
- Behavior policy modeled with a variance-preserving diffusion (epsilon prediction) conditioned on state.
- Policy encoder is decoupled from critic encoder via EMA to stabilize learning.

Usage:
    import torch
    from src.models.IDQL import IDQLConfig, IDQLAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = IDQLConfig(
        action_dim=10,
        per_step_state_dim=32,
        window_size=20,
    )
    agent = IDQLAgent(cfg, device).to(device)

    # Example training batch
    B, T, A = 64, cfg.window_size, cfg.action_dim
    states = torch.randn(B, T, cfg.per_step_state_dim, device=device)
    actions = torch.softmax(torch.randn(B, A, device=device), dim=-1)
    rewards = torch.randn(B, 1, device=device)
    next_states = torch.randn(B, T, cfg.per_step_state_dim, device=device)
    dones = torch.zeros(B, 1, device=device, dtype=torch.bool)

    metrics = agent.update((states, actions, rewards, next_states, dones))

    # Acting
    state_packed = torch.randn(T, cfg.per_step_state_dim)
    action = agent.act(state_packed)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def expectile_loss(delta: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Compute element-wise expectile regression loss.

    Args:
        delta: Difference target - prediction, arbitrary shape.
        tau: Expectile level in (0, 1). Lower values emphasize underestimation.

    Returns:
        Tensor of the same shape as delta with weighted squared error.
    """
    weight = torch.where(delta < 0, 1.0 - tau, tau)
    return weight * (delta ** 2)


def weights_to_logits(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert nonnegative weights (e.g., portfolio allocations) to centered logits.

    Args:
        weights: Tensor of shape [..., A] with nonnegative entries typically summing to 1.
        eps: Small floor to avoid log(0).

    Returns:
        Tensor of logits with mean 0 along the last dimension.
    """
    w = torch.clamp(weights, min=eps)
    logits = torch.log(w)
    logits = logits - logits.mean(dim=-1, keepdim=True)
    return logits


def logits_to_weights(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to normalized weights via softmax along the last dimension.

    Args:
        logits: Tensor of shape [..., A].

    Returns:
        Tensor of shape [..., A] with values in [0, 1] summing to 1 along last dim.
    """
    return F.softmax(logits, dim=-1)


class StateEncoder(nn.Module):
    """
    Windowed state encoder.

    Given a tensor of shape [B, T, D] (or [T, D]), concatenates the last step
    features with the temporal mean, then maps them through an MLP with
    LayerNorm and GELU activations to produce a fixed-size embedding.

    Args:
        in_per_step: Number of per-step input features D.
        hidden: Hidden width of the MLP.
        out_dim: Output embedding dimension.
    """
    def __init__(self, in_per_step: int, hidden: int = 512, out_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_per_step * 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of per-step features into a single embedding.

        Args:
            S: State tensor of shape [B, T, D] or [T, D].

        Returns:
            Tensor of shape [B, out_dim].
        """
        if S.dim() == 2:
            S = S.unsqueeze(0)
        x_last = S[:, -1, :]
        x_mean = S.mean(dim=1)
        x = torch.cat([x_last, x_mean], dim=-1)
        h = F.gelu(self.ln1(self.fc1(x)))
        h = F.gelu(self.ln2(self.fc2(h)))
        z = self.fc3(h)
        return z


class QNetwork(nn.Module):
    """
    State-action value network Q(z, a).

    Args:
        state_dim: Dimension of encoded state z.
        action_dim: Dimension of action vector (e.g., allocations).
        hidden: Hidden width.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for batches of (z, a).

        Args:
            z: Encoded state [B, state_dim].
            a: Action weights/logits [B, action_dim].

        Returns:
            Tensor [B] of Q-values.
        """
        x = torch.cat([z, a], dim=-1)
        h = F.gelu(self.ln1(self.fc1(x)))
        h = F.gelu(self.ln2(self.fc2(h)))
        q = self.fc3(h)
        return q.squeeze(-1)


class VNetwork(nn.Module):
    """
    State value network V(z).

    Args:
        state_dim: Dimension of encoded state z.
        hidden: Hidden width.
    """
    def __init__(self, state_dim: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            z: Encoded state [B, state_dim].

        Returns:
            Tensor [B] of state values.
        """
        h = F.gelu(self.ln1(self.fc1(z)))
        h = F.gelu(self.ln2(self.fc2(h)))
        v = self.fc3(h)
        return v.squeeze(-1)


class TimestepEmbed(nn.Module):
    """
    Timestep embedding with sinusoidal features followed by two Linear+SiLU layers.

    Args:
        dim: Output embedding dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Build sinusoidal embedding of integer timesteps compatible with dim size.

        Args:
            t: Integer timesteps tensor [N] or broadcastable shape.
            dim: Target embedding dimension.

        Returns:
            Tensor [..., dim] of sinusoidal features.
        """
        device = t.device
        half = dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device)
        )
        args = t.float().unsqueeze(-1) / freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Produce learned time embedding for diffusion step t.

        Args:
            t: Integer timestep tensor.

        Returns:
            Tensor [..., dim].
        """
        h = self.sinusoidal(t, self.dim)
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))
        return h


class MLPResNetBlock(nn.Module):
    """
    Residual MLP block with pre-norm and dropout.

    Sequence: Dropout -> LayerNorm -> Linear(4h) -> GELU -> Linear(h) -> Residual.
    """
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, 4 * hidden)
        self.fc2 = nn.Linear(4 * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block.

        Args:
            x: Input tensor [..., hidden].

        Returns:
            Tensor of the same shape as x.
        """
        h = self.dropout(x)
        h = self.ln(h)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DiffusionEpsModel(nn.Module):
    """
    Epsilon predictor for variance-preserving diffusion conditioned on state.

    The network receives the current noisy action x_t, the encoded state z, and
    the integer timestep t, and predicts the noise epsilon that was added to x_0.

    Args:
        action_dim: Dimensionality of the action space.
        state_dim: Dimensionality of the state embedding.
        hidden: Hidden width of the LN-ResNet MLP.
        time_dim: Size of the learned time embedding.
        num_blocks: Number of residual MLP blocks.
        dropout: Dropout rate inside residual blocks.
    """
    def __init__(self, action_dim: int, state_dim: int, hidden: int = 512,
                 time_dim: int = 128, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.time_embed = TimestepEmbed(time_dim)
        self.inp = nn.Linear(action_dim + state_dim + time_dim, hidden)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden, dropout) for _ in range(num_blocks)])
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, x_t: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict epsilon given current x_t, state embedding z, and timestep t.

        Args:
            x_t: Noisy action tensor [B, A].
            z: State embedding [B, state_dim].
            t: Integer timesteps [B].

        Returns:
            Tensor [B, A] of predicted noise.
        """
        e_t = self.time_embed(t)
        h = torch.cat([x_t, z, e_t], dim=-1)
        h = self.inp(h)
        for blk in self.blocks:
            h = blk(h)
        h = F.gelu(h)
        eps_hat = self.out(h)
        return eps_hat


class VPBetas:
    """
    Beta schedule utilities for variance-preserving (VP) diffusion.
    """
    @staticmethod
    def make_betas(
        T: int,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        schedule: str = "cosine",
    ) -> torch.Tensor:
        """
        Construct a length-T beta schedule for VP diffusion.

        Args:
            T: Number of diffusion steps.
            beta_min: Minimum beta.
            beta_max: Maximum beta.
            schedule: One of "linear" or "cosine".

        Returns:
            Tensor [T] of betas.
        """
        schedule = schedule.lower()
        if schedule == "linear":
            return torch.linspace(beta_min, beta_max, steps=T)
        if schedule == "cosine":
            s = 0.008
            steps = T + 1
            t = torch.linspace(0, T, steps=steps, dtype=torch.float32)
            f = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
            alpha_bars = f / f[0]
            betas = torch.zeros(T, dtype=torch.float32)
            for i in range(T):
                beta = 1.0 - (alpha_bars[i + 1] / alpha_bars[i])
                betas[i] = torch.clamp(beta, 1e-8, 0.999)
            return betas
        raise ValueError(f"Unknown schedule: {schedule}")

@dataclass
class IDQLConfig:
    """
    Configuration for `IDQLAgent` and its submodules.

    Groups:
    - Networks: embedding widths and dimensions for state/action.
    - RL: discount `gamma`, expectile `tau_expectile`, optimizer `lr`, target EMA.
    - Diffusion: steps, time embedding size, residual blocks, dropout, beta schedule.
    - Coupling: `policy_ema` controlling EMA from critic to policy encoder.
    - Acting: `epsilon_explore` for epsilon-greedy on top of diffusion sampling.

    `per_step_state_dim` is the feature dimension D per time step, and
    `window_size` is the number of steps T consumed by the encoder.
    """
    action_dim: int
    per_step_state_dim: int
    window_size: int
    state_emb_dim: int = 256
    hidden: int = 512

    gamma: float = 0.99
    tau_expectile: float = 0.7
    lr: float = 3e-4
    target_ema: float = 0.005

    diff_T: int = 5
    diff_time_dim: int = 128
    diff_blocks: int = 3
    diff_samples_per_state: int = 32
    diff_beta_min: float = 1e-4
    diff_beta_max: float = 2e-2
    diff_beta_schedule: str = "cosine"
    diff_dropout: float = 0.1

    policy_ema: float = 0.01

    epsilon_explore: float = 0.05

class IDQLAgent(nn.Module):
    """
    Implicit Q-Learning agent with diffusion behavior policy.

    Components:
    - Critic encoder used by Q and V.
    - Policy encoder maintained by EMA from critic encoder (no policy gradients).
    - Value network V trained by expectile regression on target Q.
    - Q network trained by TD target with V(s').
    - Diffusion epsilon model trained by behavior cloning in logits space.

    Training order per update:
    1) Update encoder + critics (Q, V) jointly.
    2) Update target Q via EMA and policy encoder via EMA from critic encoder.
    3) Train diffusion policy via MSE on epsilon prediction.

    Args:
        cfg: Configuration object.
        device: Torch device to allocate buffers and run compute.
    """
    def __init__(self, cfg: IDQLConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.encoder = StateEncoder(cfg.per_step_state_dim, out_dim=cfg.state_emb_dim, hidden=cfg.hidden)
        self.encoder_policy = StateEncoder(cfg.per_step_state_dim, out_dim=cfg.state_emb_dim, hidden=cfg.hidden)
        self.encoder_policy.load_state_dict(self.encoder.state_dict())
        for p in self.encoder_policy.parameters():
            p.requires_grad = False

        self.q = QNetwork(cfg.state_emb_dim, cfg.action_dim, hidden=cfg.hidden)
        self.q_target = QNetwork(cfg.state_emb_dim, cfg.action_dim, hidden=cfg.hidden)
        self.v = VNetwork(cfg.state_emb_dim, hidden=cfg.hidden)

        self.diff = DiffusionEpsModel(
            action_dim=cfg.action_dim,
            state_dim=cfg.state_emb_dim,
            hidden=cfg.hidden,
            time_dim=cfg.diff_time_dim,
            num_blocks=cfg.diff_blocks,
            dropout=cfg.diff_dropout,
        )

        self.q_target.load_state_dict(self.q.state_dict())

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)
        self.opt_diff = torch.optim.Adam(self.diff.parameters(), lr=cfg.lr)

        betas = VPBetas.make_betas(
            cfg.diff_T, cfg.diff_beta_min, cfg.diff_beta_max, schedule=cfg.diff_beta_schedule
        ).to(device)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode_state(self, S: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of packed states for critic networks.

        Args:
            S: Tensor [B, T, D] or [T, D].

        Returns:
            Tensor [B, state_emb_dim].
        """
        return self.encoder(S)

    def encode_state_policy(self, S: torch.Tensor) -> torch.Tensor:
        """
        Encode states with the decoupled policy encoder (EMA of critic encoder).
        No gradients flow into the policy encoder.

        Args:
            S: Tensor [B, T, D] or [T, D].

        Returns:
            Tensor [B, state_emb_dim].
        """
        with torch.no_grad():
            return self.encoder_policy(S)

    @torch.no_grad()
    def _ema_update_policy_encoder(self):
        """
        Exponential moving average update: critic encoder -> policy encoder.
        """
        tau = self.cfg.policy_ema
        for p_pol, p in zip(self.encoder_policy.parameters(), self.encoder.parameters()):
            p_pol.mul_(1.0 - tau).add_(p, alpha=tau)

    def diffusion_loss_bc(self, S: torch.Tensor, A_weights: torch.Tensor) -> torch.Tensor:
        """
        Behavior cloning loss for diffusion policy in logits space.

        Args:
            S: States [B, T, D] or [T, D].
            A_weights: Ground-truth action weights [B, A] that sum to 1.

        Returns:
            Scalar tensor of MSE between predicted and true noise.
        """
        B = S.size(0)
        z = self.encode_state_policy(S)
        x0 = weights_to_logits(A_weights)

        T = self.cfg.diff_T
        t = torch.randint(1, T + 1, (B,), device=S.device)
        eps = torch.randn_like(x0)
        a_bar_t = self.alpha_bars[t - 1].view(-1, 1)

        x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps
        eps_hat = self.diff(x_t, z, t)
        return F.mse_loss(eps_hat, eps)

    def update(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        One training step updating critics/encoder, targets, and diffusion policy.

        Args:
            batch: Tuple (states, actions, rewards, next_states, dones) where
                - states: [B, T, D]
                - actions: [B, A] (weights that sum to 1)
                - rewards: [B, 1]
                - next_states: [B, T, D]
                - dones: [B, 1] boolean

        Returns:
            Dict of scalar metrics: loss_v, loss_q, loss_diff, q_mean, v_mean.
        """
        states, actions, rewards, next_states, dones = batch
        rewards = rewards.squeeze(-1)
        dones_f = dones.float().squeeze(-1)

        z = self.encode_state(states)
        z_next = self.encode_state(next_states)

        with torch.no_grad():
            q_tgt_vals = self.q_target(z, actions)
        v_pred = self.v(z)
        v_loss = expectile_loss(q_tgt_vals - v_pred, self.cfg.tau_expectile).mean()

        with torch.no_grad():
            v_next = self.v(z_next)
            y = rewards + self.cfg.gamma * (1.0 - dones_f) * v_next
        q_pred = self.q(z, actions)
        q_loss = F.mse_loss(q_pred, y)

        self.opt_enc.zero_grad(set_to_none=True)
        self.opt_v.zero_grad(set_to_none=True)
        self.opt_q.zero_grad(set_to_none=True)
        (v_loss + q_loss).backward()
        self.opt_enc.step(); self.opt_v.step(); self.opt_q.step()

        with torch.no_grad():
            tau = self.cfg.target_ema
            for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_t.mul_(1.0 - tau).add_(p, alpha=tau)

        self._ema_update_policy_encoder()

        diff_loss = self.diffusion_loss_bc(states, actions)
        self.opt_diff.zero_grad(set_to_none=True)
        diff_loss.backward()
        self.opt_diff.step()

        return {
            "loss_v": float(v_loss.item()),
            "loss_q": float(q_loss.item()),
            "loss_diff": float(diff_loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "v_mean": float(v_pred.mean().item()),
        }

    @torch.no_grad()
    def sample_behavior_actions(self, S: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Sample candidate actions from the diffusion behavior model.

        Args:
            S: Packed state(s) [B, T, D] or [T, D].
            num_samples: Number of samples to draw per input state (B must be 1).

        Returns:
            Tensor [num_samples, A] of action weights that sum to 1.
        """
        if S.dim() == 2:
            S = S.unsqueeze(0)
        z_pol = self.encode_state_policy(S)
        z_pol = z_pol.repeat(num_samples, 1)

        A = self.cfg.action_dim
        x = torch.randn(num_samples, A, device=S.device)
        T = self.cfg.diff_T

        for ti in range(T, 0, -1):
            t = torch.full((num_samples,), ti, device=S.device, dtype=torch.long)
            eps_hat = self.diff(x, z_pol, t)
            beta_t = self.betas[ti - 1]
            alpha_t = self.alphas[ti - 1]
            a_bar_t = self.alpha_bars[ti - 1]
            x = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1.0 - a_bar_t) * eps_hat)
            if ti > 1:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)

        return logits_to_weights(x)

    @torch.no_grad()
    def act(self, state_packed: torch.Tensor) -> torch.Tensor:
        """
        Select an action for a single packed state using epsilon-greedy over
        diffusion-sampled candidates scored by the Q-network.

        Args:
            state_packed: Tensor [T, D] for a single environment state window.

        Returns:
            Tensor [A] of action weights that sum to 1 (on CPU).
        """
        device = self.device
        S = state_packed.to(device).unsqueeze(0)
        if torch.rand(1).item() < self.cfg.epsilon_explore:
            a = torch.distributions.Dirichlet(torch.ones(self.cfg.action_dim, device=device)).sample()
            return a.detach().to("cpu")

        N = self.cfg.diff_samples_per_state
        cand = self.sample_behavior_actions(S, N)
        z_critic = self.encode_state(S).repeat(N, 1)
        q_vals = self.q(z_critic, cand)
        best = cand[torch.argmax(q_vals)]
        return best.detach().to("cpu")
