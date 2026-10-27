"""ComfyUI-compatible DPM-Solver++(2M) sampling for flow models.

The sampler is model agnostic: ``denoise_fn(x, sigma)`` receives the current
latent and sigma and returns the clean-latent estimate (x0). The sigma
convention is the same as ai-toolkit's flow models: ``x_t = x0 + sigma * eps``
with sigma 1 at pure noise.

``normal_sigmas`` reproduces ComfyUI's "normal" scheduler for a
``ModelSamplingDiscreteFlow`` model: a linear grid in normalized timestep
space, mapped through the model's time-shift curve and terminated at sigma 0.
It is not tied to a particular architecture; pass the model's shift and, when
it differs, its training timestep count or terminal sigma.
"""

from typing import Callable, Optional

import torch

DenoiseFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def normal_sigmas(
    num_steps: int,
    shift: float = 1.0,
    num_train_timesteps: int = 1000,
    sigma_min: Optional[float] = None,
    sigma_max: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """ComfyUI "normal" sigma schedule for a flow model.

    With ``shift == 1`` this is a plain linear sigma ramp from ``sigma_max``
    to ``sigma_min`` followed by a terminal zero. For ``shift != 1`` each
    sigma is remapped by ``shift * t / (1 + (shift - 1) * t)``, the same
    time-shift used by ComfyUI's flow model sampling. The default
    ``sigma_min`` is the smallest sigma of the model's 1000-step shifted
    grid, which is what ComfyUI's normal scheduler uses as its lower
    timestep bound.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if shift <= 0:
        raise ValueError("shift must be > 0")
    if sigma_min is None:
        sigma_min = shift / (num_train_timesteps + shift - 1)
    t = torch.linspace(
        float(sigma_max), float(sigma_min), int(num_steps), device=device, dtype=dtype
    )
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def sample_dpmpp_2m(
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    denoise_fn: DenoiseFn,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """DPM-Solver++(2M) in sigma space.

    ``denoise_fn`` returns x0; ``callback(step, total_steps, x0)`` is called
    before each update (used for live previews). The terminal zero sigma is
    handled as the analytic limit ``x = x0`` rather than through log(0).
    """
    if len(sigmas) < 2:
        return latents
    sigmas = sigmas.to(device=latents.device, dtype=torch.float32)
    total_steps = len(sigmas) - 1
    old_denoised = None

    for i in range(total_steps):
        sigma = sigmas[i]
        denoised = denoise_fn(latents, sigma)
        if callback is not None:
            callback(i, total_steps, denoised)

        sigma_next = sigmas[i + 1]
        if float(sigma_next) == 0.0:
            latents = denoised
            break

        t = -torch.log(sigma)
        t_next = -torch.log(sigma_next)
        h = t_next - t
        if old_denoised is None:
            latents = torch.exp(t - t_next) * latents - torch.expm1(-h) * denoised
        else:
            t_prev = -torch.log(sigmas[i - 1])
            h_last = t - t_prev
            r = h_last / h
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (
                1.0 / (2.0 * r)
            ) * old_denoised
            latents = (
                torch.exp(t - t_next) * latents - torch.expm1(-h) * denoised_d
            )
        old_denoised = denoised

    return latents
