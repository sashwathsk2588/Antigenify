"""Conditional flow matching in ESM-2 latent space.

Implements the rectified-flow / conditional flow matching formulation
(Lipman et al. 2023) that spec §4.2 calls for. Linear conditional paths:

    x_0 ~ N(0, I)
    x_t = (1 - t) * x_0 + t * x_1
    u_t(x_t | x_1) = x_1 - x_0

Training minimises

    E_{x_1, x_0, t} || v_theta(x_t, t, cond) - (x_1 - x_0) ||^2 .

Inference solves the ODE dx/dt = v_theta(x, t, cond) from t=0 to t=1 with a
first-order (Euler) or second-order (Heun predictor-corrector) integrator.

Classifier guidance follows the standard idiom for continuous-time
generative models:

    v_guided(x_t, t) = v_theta(x_t, t, cond)
                     + sum_i lambda_i(t) * grad_{x_t} log p_i(target | x_hat_1)

where x_hat_1 = x_t + (1 - t) * v_theta is the model's current estimate of
the clean endpoint (derived directly from the linear path and the predicted
velocity), and each `guidance_fn(x_hat_1, t) -> log_prob` is a differentiable
head such as the trained MHC-I binder predictor.

The velocity model must accept a leading batch dimension and match the
target sample shape, and follow the contract:

    v = velocity_model(x_t, t, **cond_kwargs)   # same shape as x_t
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn


def _view_t_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a (B,) tensor of times to broadcast against `x` shape (B, ...)."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatching:
    """Rectified flow matching with linear conditional paths."""

    def __init__(
        self,
        time_sampler: Callable[[int, torch.device], torch.Tensor] | None = None,
    ) -> None:
        # Default: uniform on [0, 1). Injectable for schedule variants (logit-normal
        # etc.), which some papers use for stability.
        self._time_sampler = time_sampler or self._default_time_sampler

    @staticmethod
    def _default_time_sampler(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation between the noise sample x_0 and the data x_1."""
        t_b = _view_t_like(t, x_1)
        return (1.0 - t_b) * x_0 + t_b * x_1

    @staticmethod
    def target_velocity(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return x_1 - x_0

    def predict_x1(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover the clean endpoint from a predicted velocity.

        From x_t = (1 - t) * x_0 + t * x_1 and v = x_1 - x_0:
            x_1 = x_t + (1 - t) * v
        """
        t_b = _view_t_like(t, x_t)
        return x_t + (1.0 - t_b) * v

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def training_loss(
        self,
        velocity_model: Callable[..., torch.Tensor],
        x_1: torch.Tensor,
        cond_kwargs: dict | None = None,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One CFM training step. Returns `{loss, x_t, t, v_pred, v_target}`."""
        cond_kwargs = cond_kwargs or {}
        batch_size = x_1.shape[0]
        device = x_1.device
        if t is None:
            t = self._time_sampler(batch_size, device)
        if noise is None:
            noise = torch.randn_like(x_1)
        x_t = self.interpolate(noise, x_1, t)
        v_target = self.target_velocity(noise, x_1)
        v_pred = velocity_model(x_t, t, **cond_kwargs)
        if v_pred.shape != v_target.shape:
            raise ValueError(
                f"velocity_model produced shape {tuple(v_pred.shape)}, "
                f"expected {tuple(v_target.shape)}"
            )
        loss = (v_pred - v_target).pow(2).mean()
        return {
            "loss": loss,
            "x_t": x_t,
            "t": t,
            "v_pred": v_pred,
            "v_target": v_target,
        }

    # ------------------------------------------------------------------ #
    # ODE integrators (unguided)
    # ------------------------------------------------------------------ #

    def euler_step(
        self,
        velocity_model: Callable[..., torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        cond_kwargs: dict | None = None,
    ) -> torch.Tensor:
        cond_kwargs = cond_kwargs or {}
        v = velocity_model(x_t, t, **cond_kwargs)
        return x_t + dt * v

    def heun_step(
        self,
        velocity_model: Callable[..., torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        cond_kwargs: dict | None = None,
    ) -> torch.Tensor:
        cond_kwargs = cond_kwargs or {}
        v_a = velocity_model(x_t, t, **cond_kwargs)
        x_pred = x_t + dt * v_a
        t_next = (t + dt).clamp(max=1.0)
        v_b = velocity_model(x_pred, t_next, **cond_kwargs)
        return x_t + 0.5 * dt * (v_a + v_b)

    def sample(
        self,
        velocity_model: Callable[..., torch.Tensor],
        shape: Sequence[int],
        num_steps: int = 25,
        cond_kwargs: dict | None = None,
        solver: str = "euler",
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Integrate the ODE from t=0 to t=1 starting at x_0 ~ N(0, I)."""
        step = self._resolve_solver(solver)
        device = self._resolve_device(velocity_model, device)
        x = torch.randn(*shape, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            x = step(velocity_model, x, t, dt, cond_kwargs)
        return x

    # ------------------------------------------------------------------ #
    # Guided sampling (classifier guidance on x_hat_1)
    # ------------------------------------------------------------------ #

    def sample_guided(
        self,
        velocity_model: Callable[..., torch.Tensor],
        shape: Sequence[int],
        guidance_fns: Sequence[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        guidance_weights: Sequence[float | Callable[[float], float]] | None = None,
        num_steps: int = 25,
        cond_kwargs: dict | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Euler integration with classifier guidance on x_hat_1.

        Each `guidance_fns[i]` maps `(x_hat_1, t) -> log_prob` of shape (B,).
        We add lambda_i(t) * grad_{x_t} log p_i(x_hat_1) to the model velocity.

        `guidance_weights[i]` is either a constant float or a callable taking
        the current t (float in [0, 1]) and returning a float weight — useful
        for late-only or cosine schedules.
        """
        if guidance_weights is None:
            guidance_weights = [1.0] * len(guidance_fns)
        if len(guidance_weights) != len(guidance_fns):
            raise ValueError(
                f"guidance_weights length {len(guidance_weights)} does not match "
                f"guidance_fns length {len(guidance_fns)}"
            )
        cond_kwargs = cond_kwargs or {}
        device = self._resolve_device(velocity_model, device)
        x = torch.randn(*shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((shape[0],), t_val, device=device)

            x = x.detach().requires_grad_(True)
            v = velocity_model(x, t, **cond_kwargs)
            x_hat_1 = self.predict_x1(x, v, t)

            total_grad = torch.zeros_like(x)
            for fn, weight in zip(guidance_fns, guidance_weights):
                w = weight(t_val) if callable(weight) else weight
                if w == 0.0:
                    continue
                log_p = fn(x_hat_1, t).sum()
                grad = torch.autograd.grad(log_p, x, retain_graph=True)[0]
                total_grad = total_grad + w * grad

            v_guided = v.detach() + total_grad.detach()
            x = (x.detach() + dt * v_guided).detach()
        return x

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve_solver(self, solver: str) -> Callable[..., torch.Tensor]:
        if solver == "euler":
            return self.euler_step
        if solver == "heun":
            return self.heun_step
        raise ValueError(f"Unknown solver {solver!r}; expected 'euler' or 'heun'")

    @staticmethod
    def _resolve_device(
        velocity_model, device: torch.device | str | None
    ) -> torch.device:
        if device is not None:
            return torch.device(device)
        if isinstance(velocity_model, nn.Module):
            params = list(velocity_model.parameters())
            if params:
                return params[0].device
        return torch.device("cpu")
