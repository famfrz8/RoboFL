import torch
import torch.nn.functional as F


def masked_token_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Pool valid temporal and spatial token features."""
    if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"Expected tokens [B,T,D] and mask [B,T], got {tuple(tokens.shape)} and {tuple(valid_mask.shape)}"
        )
    valid = valid_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def masked_spatial_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Normalize a spatial affordance factor over valid views and locations."""
    if logits.shape != valid_mask.shape or logits.ndim < 2:
        raise ValueError(
            f"Expected matching spatial logits and mask, got {tuple(logits.shape)} and "
            f"{tuple(valid_mask.shape)}"
        )
    flat_logits = logits.float().flatten(start_dim=1)
    flat_mask = valid_mask.to(device=logits.device, dtype=torch.bool).flatten(start_dim=1)
    masked_logits = flat_logits.masked_fill(~flat_mask, torch.finfo(flat_logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=-1) * flat_mask.to(flat_logits.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights.view_as(logits)


def factorized_future_effect_cosine_loss(
    spatial_factor: torch.Tensor,
    effect_factor: torch.Tensor,
    current_features: torch.Tensor,
    future_features: torch.Tensor,
    view_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a spatial-channel factorization with detached future latent change."""
    if current_features.shape != future_features.shape or current_features.ndim != 5:
        raise ValueError(
            "Expected matching current/future features [B,V,C,H,W], got "
            f"{tuple(current_features.shape)} and {tuple(future_features.shape)}"
        )
    batch_size, num_views, channels, height, width = current_features.shape
    if spatial_factor.shape != (batch_size, num_views, height, width):
        raise ValueError(
            f"Expected spatial factor {(batch_size, num_views, height, width)}, got "
            f"{tuple(spatial_factor.shape)}"
        )
    if effect_factor.shape != (batch_size, channels):
        raise ValueError(
            f"Expected effect factor {(batch_size, channels)}, got {tuple(effect_factor.shape)}"
        )

    target = (future_features.float() - current_features.float()).abs().detach()
    prediction = spatial_factor.float()[:, :, None] * effect_factor.float()[:, None, :, None, None]
    if view_mask is not None:
        if view_mask.shape != (batch_size, num_views):
            raise ValueError(
                f"Expected view mask {(batch_size, num_views)}, got {tuple(view_mask.shape)}"
            )
        mask = view_mask.to(device=target.device, dtype=target.dtype)[:, :, None, None, None]
        target = target * mask
        prediction = prediction * mask

    target = target.flatten(start_dim=1)
    prediction = prediction.flatten(start_dim=1)
    valid = target.norm(dim=-1) > eps
    valid_fraction = valid.float().mean()
    target = F.normalize(target, dim=-1)
    prediction = F.normalize(prediction, dim=-1)
    loss = 1.0 - (prediction * target).sum(dim=-1)
    valid_float = valid.to(loss.dtype)
    return (loss * valid_float).sum() / valid_float.sum().clamp_min(1.0), valid_fraction


def factorized_signed_future_effect_cosine_loss(
    spatial_factor: torch.Tensor,
    effect_factor: torch.Tensor,
    current_features: torch.Tensor,
    future_features: torch.Tensor,
    view_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a factorized prediction with direction-preserving future change."""
    if current_features.shape != future_features.shape or current_features.ndim != 5:
        raise ValueError(
            "Expected matching current/future features [B,V,C,H,W], got "
            f"{tuple(current_features.shape)} and {tuple(future_features.shape)}"
        )
    batch_size, num_views, channels, height, width = current_features.shape
    if spatial_factor.shape != (batch_size, num_views, height, width):
        raise ValueError(
            f"Expected spatial factor {(batch_size, num_views, height, width)}, got "
            f"{tuple(spatial_factor.shape)}"
        )
    if effect_factor.shape != (batch_size, 2 * channels):
        raise ValueError(
            f"Expected signed effect factor {(batch_size, 2 * channels)}, got {tuple(effect_factor.shape)}"
        )

    delta = (future_features.float() - current_features.float()).detach()
    target = torch.cat([F.relu(delta), F.relu(-delta)], dim=2)
    prediction = spatial_factor.float()[:, :, None] * F.softplus(effect_factor.float())[:, None, :, None, None]
    if view_mask is not None:
        if view_mask.shape != (batch_size, num_views):
            raise ValueError(
                f"Expected view mask {(batch_size, num_views)}, got {tuple(view_mask.shape)}"
            )
        mask = view_mask.to(device=target.device, dtype=target.dtype)[:, :, None, None, None]
        target = target * mask
        prediction = prediction * mask

    target = target.flatten(start_dim=1)
    prediction = prediction.flatten(start_dim=1)
    valid = target.norm(dim=-1) > eps
    valid_fraction = valid.float().mean()
    target = F.normalize(target, dim=-1)
    prediction = F.normalize(prediction, dim=-1)
    loss = 1.0 - (prediction * target).sum(dim=-1)
    valid_float = valid.to(loss.dtype)
    return (loss * valid_float).sum() / valid_float.sum().clamp_min(1.0), valid_fraction


def compute_router_lead_in_steps(total_steps: int, fraction: float, enabled: bool) -> int:
    """Return the initial server steps that keep uploaded LoRA experts fixed."""
    if total_steps < 0:
        raise ValueError(f"total_steps must be non-negative, got {total_steps}")
    if fraction < 0 or fraction >= 1:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    if not enabled:
        return 0
    return int(total_steps * fraction)


def should_accumulate_router_usage(step: int, lead_in_steps: int) -> bool:
    """Exclude lead-in routing from expert aggregation evidence."""
    if step < 0 or lead_in_steps < 0:
        raise ValueError(
            f"step and lead_in_steps must be non-negative, got {step} and {lead_in_steps}"
        )
    return step >= lead_in_steps


def make_affordance_action_horizon_mask(
    length: int,
    horizon: int,
    decay_end: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a smooth temporal mask for horizon-aware affordance conditioning.

    The mask stays at one through ``horizon``, follows a cosine decay until
    ``decay_end``, and is zero afterwards. It is returned as ``[1, length, 1]``
    so it can be multiplied directly with action-token residuals.
    """
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if horizon < 0 or decay_end < horizon or decay_end > length:
        raise ValueError(
            "Expected 0 <= horizon <= decay_end <= length, got "
            f"horizon={horizon}, decay_end={decay_end}, length={length}"
        )

    positions = torch.arange(length, device=device, dtype=torch.float32)
    mask = torch.zeros(length, device=device, dtype=torch.float32)
    mask = torch.where(positions < float(horizon), torch.ones_like(mask), mask)
    if decay_end > horizon:
        decay_positions = positions - float(horizon)
        decay_width = float(decay_end - horizon)
        decay_values = 0.5 * (1.0 + torch.cos(torch.pi * decay_positions / decay_width))
        in_decay = (positions >= float(horizon)) & (positions < float(decay_end))
        mask = torch.where(in_decay, decay_values, mask)
    return mask.to(dtype=dtype).view(1, length, 1)


def is_loramoe_expert_parameter(name: str) -> bool:
    """Identify expert A/B factors while excluding routers and affordance parameters."""
    return (
        ".lora_A." in name
        or ".lora_B." in name
        or "state_proj_lora_A_moe" in name
        or "state_proj_lora_B_moe" in name
    )


def clear_loramoe_expert_gradients(named_parameters) -> tuple[int, int]:
    """Discard expert gradients while preserving router and conditioning gradients."""
    cleared_tensors = 0
    cleared_numel = 0
    for name, parameter in named_parameters:
        if is_loramoe_expert_parameter(name) and parameter.grad is not None:
            cleared_tensors += 1
            cleared_numel += parameter.numel()
            parameter.grad = None
    return cleared_tensors, cleared_numel
