from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from diffusionGS.models.scheduler.ddim_scheduler import (
    DDIMSchedulerOutput,
    betas_for_alpha_bar,
    rescale_zero_terminal_snr,
)


@dataclass
class DiscreteSchedulerOutput(BaseOutput):
    """Output class for the discrete scheduler's `step` function."""

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    A lightweight scheduler for discrete (categorical or token-based) states.

    The public API mirrors the ``DDIMScheduler`` used elsewhere in the
    codebase, supporting ``set_timesteps`` and ``step`` with different
    ``prediction_type`` settings. The scheduler blends predicted clean logits
    and current noisy logits to simulate discrete state transitions while
    following a diffusion-style noise schedule.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "sample",
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        transition_power: float = 1.0,
        rescale_betas_zero_snr: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps, alpha_transform_type="cosine")
        else:
            raise NotImplementedError(f"beta_schedule {beta_schedule} is not implemented for DiscreteScheduler")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)

        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy()).long()
        self.set_timesteps(num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None):
        """Set timesteps for inference following the configured spacing."""

        if self.config.timestep_spacing == "leading":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
        elif self.config.timestep_spacing == "trailing":
            timesteps = np.linspace(
                self.config.num_train_timesteps - 1, 0, num_inference_steps, endpoint=False
            ) + 1
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.config.timestep_spacing}")

        timesteps = torch.from_numpy(np.round(timesteps)[::-1].copy()).long() + self.config.steps_offset
        self.timesteps = timesteps.to(device) if device is not None else timesteps
        self.num_inference_steps = num_inference_steps

    def _get_variance(self, timestep: torch.Tensor) -> torch.Tensor:
        prev_timestep = timestep - 1
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = torch.where(prev_timestep >= 0, self.alphas_cumprod[prev_timestep], self.final_alpha_cumprod)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, tuple]:
        """
        Progress the sample from ``t`` to ``t-1`` using discrete transitions.

        Args are consistent with ``DDIMScheduler.step``. The method expects
        ``sample`` and ``model_output`` to contain logits over categories
        (e.g., token vocab or categorical states) and mixes them according to
        the diffusion noise level.
        """

        if isinstance(timestep, int):
            timestep = torch.tensor(timestep, device=sample.device)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.final_alpha_cumprod

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(1 - alpha_prod_t) * model_output
        else:
            raise ValueError(f"prediction_type {self.config.prediction_type} is not supported")

        # convert logits to probabilities for discrete blending
        clean_prob = torch.softmax(pred_original_sample, dim=-1)
        noisy_prob = torch.softmax(sample, dim=-1)

        # variance controls how much noise remains; convert to a mixing weight
        variance = self._get_variance(timestep)
        blend = torch.clamp(torch.sqrt(alpha_prod_t_prev / alpha_prod_t), 0.0, 1.0)
        blend = blend * (1.0 - variance).to(sample.device)
        blend = blend * self.config.transition_power

        blend = blend.view(*([1] * (clean_prob.ndim - 1)), 1)

        prev_prob = blend * clean_prob + (1 - blend) * noisy_prob
        prev_prob = prev_prob.clamp(min=1e-12)
        prev_sample = torch.log(prev_prob)

        if not return_dict:
            return prev_sample, pred_original_sample

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
