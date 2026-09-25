from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _normalized_router_probs(router_probs: torch.Tensor) -> torch.Tensor:
    probs = router_probs.float().clamp_min(1e-8)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _router_token_importance(router_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = _normalized_router_probs(router_probs)
    confidence = probs.amax(dim=-1)
    if probs.shape[-1] <= 1:
        certainty = torch.ones_like(confidence)
    else:
        entropy = -(probs * probs.log()).sum(dim=-1)
        certainty = 1.0 - entropy / math.log(probs.shape[-1])
        certainty = certainty.clamp(0.0, 1.0)
    return confidence * certainty, certainty


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.ones_like(values)
    minimum = values.min()
    return (values - minimum) / (values.max() - minimum).clamp_min(1e-6)


def _project_and_normalize_features(features: torch.Tensor, projection_dim: int) -> torch.Tensor:
    projected = features.detach().float()
    target_dim = min(max(1, int(projection_dim)), projected.shape[-1])
    if target_dim < projected.shape[-1]:
        projected = F.adaptive_avg_pool1d(projected.unsqueeze(1), target_dim).squeeze(1)
    return F.normalize(projected, p=2, dim=-1)


@dataclass
class VisualTokenPrunePlan:
    prefix_pad_masks: torch.Tensor
    visual_token_indices: list[list[int]]
    visual_token_counts: list[int]
    ot_losses: list[torch.Tensor] | None = None
    stats_call_count: int = 0
    prune_module_calls: int = 0
    prune_sample_decisions: int = 0
    prune_visual_before: int = 0
    prune_visual_after: int = 0
    prune_adapter_before: int = 0
    prune_adapter_after: int = 0
    prune_router_certainty_sum: float = 0.0
    prune_ot_correction_swaps: int = 0
    _visual_token_cache: dict[tuple[int, str, int | None], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    @staticmethod
    def _cache_key(batch_idx: int, device: torch.device) -> tuple[int, str, int | None]:
        return batch_idx, device.type, device.index

    def _visual_token_tensor(self, batch_idx: int, device: torch.device) -> torch.Tensor:
        key = self._cache_key(batch_idx, device)
        cached = self._visual_token_cache.get(key)
        if cached is not None:
            return cached
        tokens = self.visual_token_indices[batch_idx]
        tensor = torch.tensor(tokens, device=device, dtype=torch.long)
        self._visual_token_cache[key] = tensor
        return tensor

    def _select_visual_tokens(
        self,
        probs: torch.Tensor,
        batch_idx: int,
        token_features: torch.Tensor,
        apply_ot_correction: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
        visual_tokens = self._visual_token_tensor(batch_idx, probs.device)
        if visual_tokens.numel() == 0:
            return visual_tokens, visual_tokens, visual_tokens, 0.0, 0

        valid = self.prefix_pad_masks[batch_idx, visual_tokens]
        visual_tokens = visual_tokens[valid]
        if visual_tokens.numel() <= 1:
            return visual_tokens, visual_tokens, visual_tokens.new_empty((0,)), 1.0, 0

        token_probs = probs[batch_idx, visual_tokens]
        importance, certainty = _router_token_importance(token_probs)
        mean_certainty = float(certainty.detach().mean())
        detached_importance = importance.detach()
        top1_experts = token_probs.argmax(dim=-1)
        active_experts = top1_experts.unique()

        total_importance = detached_importance.sum()
        if float(total_importance) <= 1e-8:
            return visual_tokens, visual_tokens, visual_tokens.new_empty((0,)), mean_certainty, 0

        importance_mass = detached_importance / total_importance
        effective_tokens = torch.exp(
            -(importance_mass * importance_mass.clamp_min(1e-8).log()).sum()
        )
        uncertainty_tokens = visual_tokens.numel() * (1.0 - certainty.detach().mean())
        keep_count = max(
            math.ceil(visual_tokens.numel() / 2),
            active_experts.numel(),
            math.ceil(float(effective_tokens)),
            math.ceil(float(uncertainty_tokens)),
        )
        keep_count = min(visual_tokens.numel(), keep_count)
        order = torch.argsort(detached_importance, descending=True, stable=True)
        keep_local_mask = torch.zeros(visual_tokens.numel(), device=probs.device, dtype=torch.bool)
        keep_local_mask[order[:keep_count]] = True

        # Preserve at least one representative for every top-1 expert present in this sample.
        for expert_idx in active_experts:
            expert_local = torch.nonzero(top1_experts == expert_idx, as_tuple=False).flatten()
            best_local = expert_local[detached_importance[expert_local].argmax()]
            keep_local_mask[best_local] = True

        correction_swaps = 0
        if apply_ot_correction:
            keep_local_mask, correction_swaps = self._apply_ot_coverage_correction(
                token_features[visual_tokens],
                token_probs,
                detached_importance,
                keep_local_mask,
            )

        kept_tokens = visual_tokens[keep_local_mask]
        pruned_tokens = visual_tokens[~keep_local_mask]
        return visual_tokens, kept_tokens, pruned_tokens, mean_certainty, correction_swaps

    def _apply_ot_coverage_correction(
        self,
        visual_features: torch.Tensor,
        token_probs: torch.Tensor,
        router_importance: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        kept_local = torch.nonzero(keep_mask, as_tuple=False).flatten()
        pruned_local = torch.nonzero(~keep_mask, as_tuple=False).flatten()
        if kept_local.numel() <= 1 or pruned_local.numel() == 0:
            return keep_mask, 0

        with torch.no_grad():
            features = _project_and_normalize_features(visual_features, token_probs.shape[-1])

            kept_features = features[kept_local]
            pruned_features = features[pruned_local]
            pruned_residual = 1.0 - (pruned_features @ kept_features.transpose(0, 1)).amax(dim=1)

            kept_similarity = kept_features @ kept_features.transpose(0, 1)
            kept_similarity.fill_diagonal_(float("-inf"))
            kept_uniqueness = 1.0 - kept_similarity.amax(dim=1)

            router_score = _minmax_normalize(router_importance)
            coverage_score = _minmax_normalize(torch.cat([pruned_residual, kept_uniqueness]))
            candidate_score = router_score[pruned_local] + coverage_score[: pruned_local.numel()]
            retained_score = router_score[kept_local] + coverage_score[pruned_local.numel() :]

            protected = torch.zeros_like(keep_mask)
            top1_experts = token_probs.argmax(dim=-1)
            for expert_idx in top1_experts.unique():
                expert_local = torch.nonzero(top1_experts == expert_idx, as_tuple=False).flatten()
                best_local = expert_local[router_importance[expert_local].argmax()]
                protected[best_local] = True

            removable_positions = torch.nonzero(~protected[kept_local], as_tuple=False).flatten()
            if removable_positions.numel() == 0:
                return keep_mask, 0

            swap_count = min(pruned_local.numel(), removable_positions.numel())
            candidate_order = torch.argsort(candidate_score, descending=True, stable=True)[:swap_count]
            removable_order = removable_positions[
                torch.argsort(retained_score[removable_positions], descending=False, stable=True)[:swap_count]
            ]
            beneficial = candidate_score[candidate_order] > retained_score[removable_order]
            if not beneficial.any():
                return keep_mask, 0

            corrected = keep_mask.clone()
            corrected[kept_local[removable_order[beneficial]]] = False
            corrected[pruned_local[candidate_order[beneficial]]] = True
            return corrected, int(beneficial.sum())

    def reset_ot_losses(self) -> None:
        self.ot_losses = []

    def maybe_add_ot_loss(
        self,
        x_flat: torch.Tensor,
        original_shape: torch.Size | tuple[int, ...],
        router_probs: torch.Tensor,
        prune_state,
    ) -> None:
        if self.ot_losses is None or prune_state is None:
            return
        if len(original_shape) != 3:
            return

        batch_size, seq_len = original_shape[:2]
        hidden_dim = x_flat.shape[-1]
        if x_flat.shape[0] != batch_size * seq_len or router_probs.shape[0] != x_flat.shape[0]:
            return

        x_view = x_flat.reshape(batch_size, seq_len, hidden_dim)
        probs_view = router_probs.reshape(batch_size, seq_len, router_probs.shape[-1]).float()
        losses = []

        for batch_idx in range(batch_size):
            source_tokens = prune_state["source_visual_indices"][batch_idx]
            kept_tokens = prune_state["kept_visual_indices"][batch_idx]
            if source_tokens.numel() <= kept_tokens.numel() or kept_tokens.numel() == 0:
                continue

            source_points = _project_and_normalize_features(
                x_view[batch_idx, source_tokens],
                probs_view.shape[-1],
            )
            target_local = torch.searchsorted(source_tokens, kept_tokens)
            target_points = source_points[target_local]
            similarity = source_points @ target_points.transpose(0, 1)
            nearest_similarity, nearest_target = similarity.max(dim=1)

            source_probs = _normalized_router_probs(probs_view[batch_idx, source_tokens])
            target_probs = _normalized_router_probs(probs_view[batch_idx, kept_tokens])
            transported_probs = target_probs[nearest_target]
            pruned_mask = ~torch.isin(source_tokens, kept_tokens)
            if not pruned_mask.any():
                continue

            source_probs = source_probs[pruned_mask]
            transported_probs = transported_probs[pruned_mask]
            midpoint = (source_probs + transported_probs) * 0.5
            js_divergence = 0.5 * (
                (source_probs * (source_probs.log() - midpoint.log())).sum(dim=-1)
                + (transported_probs * (transported_probs.log() - midpoint.log())).sum(dim=-1)
            )
            if source_probs.shape[-1] > 1:
                js_divergence = js_divergence / math.log(source_probs.shape[-1])
            transport_distance = (1.0 - nearest_similarity[pruned_mask]).clamp_min(0.0)
            loss = (js_divergence * transport_distance).sum() / transport_distance.sum().clamp_min(1e-8)
            losses.append(loss)

        if losses:
            self.ot_losses.append(torch.stack([loss.reshape(()) for loss in losses]).mean())

    def get_ot_loss(self) -> torch.Tensor | None:
        if not self.ot_losses:
            return None
        return torch.stack([loss.reshape(()) for loss in self.ot_losses]).mean()

    def get_ot_loss_count(self) -> int:
        return len(self.ot_losses) if self.ot_losses is not None else 0

    def reset_prune_stats(self) -> None:
        self.prune_module_calls = 0
        self.prune_sample_decisions = 0
        self.prune_visual_before = 0
        self.prune_visual_after = 0
        self.prune_adapter_before = 0
        self.prune_adapter_after = 0
        self.prune_router_certainty_sum = 0.0
        self.prune_ot_correction_swaps = 0

    def _record_prune_stats(self, reduction_stats: list[dict]) -> None:
        self.prune_module_calls += 1
        self.prune_sample_decisions += len(reduction_stats)
        for stats in reduction_stats:
            self.prune_visual_before += stats["visual_before"]
            self.prune_visual_after += stats["visual_after"]
            self.prune_adapter_before += stats["total_before"]
            self.prune_adapter_after += stats["total_after"]
            self.prune_router_certainty_sum += stats["router_certainty"]
            self.prune_ot_correction_swaps += stats["ot_correction_swaps"]

    def get_prune_stats(self) -> dict[str, float | int]:
        visual_pruned = self.prune_visual_before - self.prune_visual_after
        adapter_pruned = self.prune_adapter_before - self.prune_adapter_after
        visual_pruned_pct = (
            100.0 * visual_pruned / self.prune_visual_before if self.prune_visual_before else 0.0
        )
        adapter_pruned_pct = (
            100.0 * adapter_pruned / self.prune_adapter_before if self.prune_adapter_before else 0.0
        )
        mean_router_certainty = (
            self.prune_router_certainty_sum / self.prune_sample_decisions
            if self.prune_sample_decisions
            else 0.0
        )
        return {
            "module_calls": self.prune_module_calls,
            "sample_decisions": self.prune_sample_decisions,
            "visual_before": self.prune_visual_before,
            "visual_after": self.prune_visual_after,
            "visual_pruned": visual_pruned,
            "visual_pruned_pct": visual_pruned_pct,
            "adapter_before": self.prune_adapter_before,
            "adapter_after": self.prune_adapter_after,
            "adapter_pruned": adapter_pruned,
            "adapter_pruned_pct": adapter_pruned_pct,
            "mean_router_certainty": mean_router_certainty,
            "ot_correction_swaps": self.prune_ot_correction_swaps,
        }

    def print_prune_summary(self) -> None:
        if not _env_flag("USE_VISUAL_TOKEN_PRUNE_PRINT_SUMMARY", True):
            return
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        stats = self.get_prune_stats()
        if stats["module_calls"] == 0:
            return
        print(
            "[VisualTokenPruneSummary] "
            f"module_calls={stats['module_calls']} sample_decisions={stats['sample_decisions']} "
            f"visual_tokens={stats['visual_before']}->{stats['visual_after']} "
            f"visual_pruned={stats['visual_pruned_pct']:.2f}% "
            f"router_certainty={stats['mean_router_certainty']:.4f} "
            f"ot_swaps={stats['ot_correction_swaps']}",
            flush=True,
        )

    @torch.compiler.disable
    def compact_moe_inputs(
        self,
        x_flat: torch.Tensor,
        original_shape: torch.Size | tuple[int, ...],
        router_probs: torch.Tensor,
        *weighted_tensors: torch.Tensor,
        apply_ot_correction: bool = False,
    ):
        if len(original_shape) != 3:
            return None
        batch_size, seq_len = original_shape[:2]
        hidden_dim = x_flat.shape[-1]
        if batch_size != self.prefix_pad_masks.shape[0] or seq_len != self.prefix_pad_masks.shape[1]:
            return None
        if x_flat.shape[0] != batch_size * seq_len or router_probs.shape[0] != x_flat.shape[0]:
            return None

        x_view = x_flat.reshape(batch_size, seq_len, hidden_dim)
        probs_view = router_probs.reshape(batch_size, seq_len, router_probs.shape[-1])
        tensor_views = [tensor.reshape(batch_size, seq_len, *tensor.shape[1:]) for tensor in weighted_tensors]
        compact_x = []
        compact_tensors = [[] for _ in weighted_tensors]
        restore_indices = []
        source_visual_indices = []
        kept_visual_indices = []
        reduction_stats = []
        compact_offset = 0

        for batch_idx in range(batch_size):
            visual_tokens, kept_visual, pruned_visual, mean_certainty, correction_swaps = (
                self._select_visual_tokens(
                    probs_view,
                    batch_idx,
                    x_view[batch_idx],
                    apply_ot_correction,
                )
            )
            pruned_mask = torch.zeros((seq_len,), device=x_flat.device, dtype=torch.bool)
            pruned_mask[pruned_visual] = True
            keep_tokens = torch.nonzero(~pruned_mask, as_tuple=False).flatten()

            compact_x.append(x_view[batch_idx, keep_tokens])
            for tensor_idx, tensor_view in enumerate(tensor_views):
                compact_tensors[tensor_idx].append(tensor_view[batch_idx, keep_tokens])

            local_restore = torch.full((seq_len,), -1, device=x_flat.device, dtype=torch.long)
            local_restore[keep_tokens] = compact_offset + torch.arange(keep_tokens.numel(), device=x_flat.device)
            restore_indices.append(local_restore)
            source_visual_indices.append(visual_tokens)
            kept_visual_indices.append(kept_visual)
            compact_offset += keep_tokens.numel()
            reduction_stats.append(
                {
                    "batch_idx": batch_idx,
                    "visual_before": int(visual_tokens.numel()),
                    "pruned_tokens": int(pruned_visual.numel()),
                    "visual_after": int(kept_visual.numel()),
                    "total_before": seq_len,
                    "total_after": int(keep_tokens.numel()),
                    "router_certainty": mean_certainty,
                    "ot_correction_swaps": correction_swaps,
                }
            )

        self._record_prune_stats(reduction_stats)
        if compact_offset == x_flat.shape[0]:
            return None

        prune_state = {
            "x_flat": torch.cat(compact_x, dim=0),
            "weighted_tensors": [torch.cat(values, dim=0) for values in compact_tensors],
            "restore_indices": torch.cat(restore_indices, dim=0),
            "original_rows": x_flat.shape[0],
            "source_visual_indices": source_visual_indices,
            "kept_visual_indices": kept_visual_indices,
            "reduction_stats": reduction_stats,
        }

        print_stats = (
            _env_flag("USE_VISUAL_TOKEN_PRUNE_PRINT_STATS", False)
            and int(os.environ.get("LOCAL_RANK", "0")) == 0
        )
        if print_stats:
            self.stats_call_count += 1
            max_calls = _env_int("USE_VISUAL_TOKEN_PRUNE_PRINT_STATS_MAX_CALLS", 1)
            print_stats = self.stats_call_count <= max(0, max_calls)
        if print_stats:
            print(
                "[VisualTokenPruneApplied] "
                f"call={self.stats_call_count} actual_adapter_rows={x_flat.shape[0]}->{prune_state['x_flat'].shape[0]} "
                f"removed={x_flat.shape[0] - prune_state['x_flat'].shape[0]}",
                flush=True,
            )
            for stats in reduction_stats:
                visual_pruned_pct = (
                    100.0 * stats["pruned_tokens"] / stats["visual_before"]
                    if stats["visual_before"]
                    else 0.0
                )
                print(
                    "[VisualTokenPruneStats] "
                    f"call={self.stats_call_count} batch={stats['batch_idx']} "
                    f"visual_before={stats['visual_before']} pruned={stats['pruned_tokens']} "
                    f"visual_after={stats['visual_after']} visual_pruned={visual_pruned_pct:.2f}% "
                    f"router_certainty={stats['router_certainty']:.4f} "
                    f"ot_swaps={stats['ot_correction_swaps']}",
                    flush=True,
                )
        return prune_state

    @torch.compiler.disable
    def restore_moe_output(self, compact_output: torch.Tensor, prune_state) -> torch.Tensor:
        restore_indices = prune_state["restore_indices"]
        out = compact_output.new_zeros((prune_state["original_rows"], compact_output.shape[-1]))
        valid = restore_indices >= 0
        out[valid] = compact_output[restore_indices[valid]]
        return out


_PATCHED_LORAMOE = False


def _get_router_probs(module, router_name: str = "router") -> torch.Tensor | None:
    observables = getattr(module, "_last_router_observables", None)
    if not observables or router_name not in observables:
        return None
    return observables[router_name].get("p")


def _clear_router_probs(module, router_name: str) -> None:
    observables = getattr(module, "_last_router_observables", None)
    if isinstance(observables, dict):
        observables.pop(router_name, None)


def _get_or_store_router_probs(
    module,
    router_name: str,
    fallback_probs: torch.Tensor,
) -> torch.Tensor:
    router_probs = _get_router_probs(module, router_name)
    if router_probs is not None:
        return router_probs

    observables = getattr(module, "_last_router_observables", None)
    if not isinstance(observables, dict):
        observables = {}
    observables[router_name] = {"p": fallback_probs}
    module._last_router_observables = observables
    return fallback_probs


def _dense_router_weights(
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.zeros(top_k_probs.shape[0], num_experts, device=top_k_probs.device, dtype=dtype)
    weights.scatter_(1, top_k_indices, top_k_probs.to(dtype))
    return weights


def install_loramoe_visual_token_prune_patch() -> bool:
    global _PATCHED_LORAMOE
    if _PATCHED_LORAMOE:
        return True

    try:
        from peft.tuners.lora.layer import LoraLayer
    except Exception:
        return False

    if not hasattr(LoraLayer, "lora_moe_forward"):
        return False

    original_moe_forward = LoraLayer.lora_moe_forward
    original_shared_a_forward = getattr(LoraLayer, "lora_moe_forward_shared_a", None)
    original_ab_forward = getattr(LoraLayer, "lora_moe_forward_ab", None)

    def lora_moe_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        plan = getattr(self, "_visual_token_prune_plan", None)
        if plan is None:
            return original_moe_forward(self, x, *args, **kwargs)

        original_dtype = x.dtype
        base_dtype = self.base_layer.weight.dtype
        x_for_base = x.to(base_dtype) if x.dtype != base_dtype else x
        result = self.base_layer(x_for_base, *args, **kwargs)
        result_flat = result.reshape(-1, result.shape[-1])

        _clear_router_probs(self, "router")
        top_k_probs, top_k_indices, x_flat = self.lora_router_forward(x)
        adapter_names = getattr(self, "_lora_adapter_names", [str(i) for i in range(len(self.lora_A))])
        lora_A_stack = torch.stack([self.lora_A[name].weight for name in adapter_names], dim=0).to(x_flat.dtype)
        lora_B_stack = torch.stack([self.lora_B[name].weight for name in adapter_names], dim=0).to(x_flat.dtype)
        if hasattr(self, "_lora_scaling_buffer"):
            scaling = self._lora_scaling_buffer.to(device=x_flat.device, dtype=x_flat.dtype)
        else:
            scaling = torch.tensor(
                [self.scaling.get(name, 1.0) for name in adapter_names],
                device=x_flat.device,
                dtype=x_flat.dtype,
            )

        router_weights = _dense_router_weights(top_k_probs, top_k_indices, len(adapter_names), x_flat.dtype)
        pruning_probs = _get_or_store_router_probs(self, "router", router_weights)
        prune_state = plan.compact_moe_inputs(
            x_flat,
            x.shape,
            pruning_probs,
            router_weights,
            apply_ot_correction=getattr(self, "_visual_token_prune_ot_correction_enabled", False),
        )
        if getattr(self, "_visual_token_prune_ot_enabled", False):
            plan.maybe_add_ot_loss(x_flat, x.shape, pruning_probs, prune_state)
        if prune_state is not None:
            x_flat = prune_state["x_flat"]
            router_weights = prune_state["weighted_tensors"][0]

        lora_A_out = torch.einsum("ti, eri -> ter", x_flat.to(lora_A_stack.dtype), lora_A_stack)
        lora_B_out = torch.einsum("ter, eor -> teo", lora_A_out, lora_B_stack)
        lora_B_out = lora_B_out * scaling.to(lora_B_out.dtype).view(1, -1, 1)
        moe_diff = (lora_B_out * router_weights.to(lora_B_out.dtype).unsqueeze(-1)).sum(dim=1)
        if prune_state is not None:
            moe_diff = plan.restore_moe_output(moe_diff, prune_state)

        result_flat = result_flat + moe_diff.to(result_flat.dtype)
        return result_flat.reshape(*result.shape).to(original_dtype)

    def lora_moe_forward_shared_a(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        plan = getattr(self, "_visual_token_prune_plan", None)
        if plan is None:
            return original_shared_a_forward(self, x, *args, **kwargs)

        original_dtype = x.dtype
        base_dtype = self.base_layer.weight.dtype
        x_for_base = x.to(base_dtype) if x.dtype != base_dtype else x
        result = self.base_layer(x_for_base, *args, **kwargs)
        result_flat = result.reshape(-1, result.shape[-1])

        shared_adapter = getattr(self, "_shared_lora_adapter", "shared")
        if shared_adapter not in self.lora_A:
            return result
        lora_A = self.lora_A[shared_adapter]
        dropout = self.lora_dropout[shared_adapter]
        x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
        x_flat = x_cast.reshape(-1, x_cast.shape[-1]) if len(x_cast.shape) == 3 else x_cast
        shared_a_out = lora_A(dropout(x_flat))

        _clear_router_probs(self, "router")
        top_k_probs, top_k_indices = self.lora_router_forward_from_features(shared_a_out)
        adapter_names = getattr(self, "_lora_adapter_names", [str(i) for i in range(len(self.lora_B))])
        lora_B_stack = torch.stack([self.lora_B[name].weight for name in adapter_names], dim=0).to(shared_a_out.dtype)
        if hasattr(self, "_lora_scaling_buffer"):
            scaling = self._lora_scaling_buffer.to(device=shared_a_out.device, dtype=shared_a_out.dtype)
        else:
            scaling = torch.tensor(
                [self.scaling.get(name, 1.0) for name in adapter_names],
                device=shared_a_out.device,
                dtype=shared_a_out.dtype,
            )

        router_weights = _dense_router_weights(top_k_probs, top_k_indices, len(adapter_names), shared_a_out.dtype)
        router_probs = _get_or_store_router_probs(self, "router", router_weights)
        prune_state = plan.compact_moe_inputs(
            shared_a_out,
            x.shape,
            router_probs,
            router_weights,
            apply_ot_correction=getattr(self, "_visual_token_prune_ot_correction_enabled", False),
        )
        if getattr(self, "_visual_token_prune_ot_enabled", False):
            plan.maybe_add_ot_loss(shared_a_out, x.shape, router_probs, prune_state)
        if prune_state is not None:
            shared_a_out = prune_state["x_flat"]
            router_weights = prune_state["weighted_tensors"][0]

        lora_B_out = torch.einsum("tr, eor -> teo", shared_a_out.to(lora_B_stack.dtype), lora_B_stack)
        lora_B_out = lora_B_out * scaling.to(lora_B_out.dtype).view(1, -1, 1)
        moe_diff = (lora_B_out * router_weights.to(lora_B_out.dtype).unsqueeze(-1)).sum(dim=1)
        if prune_state is not None:
            moe_diff = plan.restore_moe_output(moe_diff, prune_state)

        result_flat = result_flat + moe_diff.to(result_flat.dtype)
        return result_flat.reshape(*result.shape).to(original_dtype)

    def lora_moe_forward_ab(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        plan = getattr(self, "_visual_token_prune_plan", None)
        if plan is None:
            return original_ab_forward(self, x, *args, **kwargs)

        original_dtype = x.dtype
        base_dtype = self.base_layer.weight.dtype
        x_for_base = x.to(base_dtype) if x.dtype != base_dtype else x
        result = self.base_layer(x_for_base, *args, **kwargs)
        result_flat = result.reshape(-1, result.shape[-1])

        _clear_router_probs(self, "router_a")
        _clear_router_probs(self, "router_b")
        top_k_probs_a, top_k_indices_a, top_k_probs_b, top_k_indices_b, x_flat = self.lora_router_forward_ab(x)
        adapter_names = getattr(self, "_lora_adapter_names", [str(i) for i in range(len(self.lora_A))])
        lora_A_stack = torch.stack([self.lora_A[name].weight for name in adapter_names], dim=0).to(x_flat.dtype)
        lora_B_stack = torch.stack([self.lora_B[name].weight for name in adapter_names], dim=0).to(x_flat.dtype)
        if hasattr(self, "_lora_scaling_buffer"):
            scaling = self._lora_scaling_buffer.to(device=x_flat.device, dtype=x_flat.dtype)
        else:
            scaling = torch.tensor(
                [self.scaling.get(name, 1.0) for name in adapter_names],
                device=x_flat.device,
                dtype=x_flat.dtype,
            )

        weights_a = _dense_router_weights(top_k_probs_a, top_k_indices_a, len(adapter_names), x_flat.dtype)
        weights_b = _dense_router_weights(top_k_probs_b, top_k_indices_b, len(adapter_names), x_flat.dtype)
        _get_or_store_router_probs(self, "router_a", weights_a)
        pruning_probs = _get_or_store_router_probs(self, "router_b", weights_b)
        prune_state = plan.compact_moe_inputs(
            x_flat,
            x.shape,
            pruning_probs,
            weights_a,
            weights_b,
            apply_ot_correction=getattr(self, "_visual_token_prune_ot_correction_enabled", False),
        )
        if getattr(self, "_visual_token_prune_ot_enabled", False):
            plan.maybe_add_ot_loss(x_flat, x.shape, pruning_probs, prune_state)
        if prune_state is not None:
            x_flat = prune_state["x_flat"]
            weights_a, weights_b = prune_state["weighted_tensors"]

        lora_A_out = torch.einsum("ti, eri -> ter", x_flat.to(lora_A_stack.dtype), lora_A_stack)
        lora_A_weighted = (lora_A_out * weights_a.to(lora_A_out.dtype).unsqueeze(-1)).sum(dim=1)
        lora_B_out = torch.einsum(
            "ter, eor -> teo",
            lora_A_weighted.unsqueeze(1).expand(-1, lora_A_out.shape[1], -1),
            lora_B_stack,
        )
        lora_B_out = lora_B_out * scaling.to(lora_B_out.dtype).view(1, -1, 1)
        moe_diff = (lora_B_out * weights_b.to(lora_B_out.dtype).unsqueeze(-1)).sum(dim=1)
        if prune_state is not None:
            moe_diff = plan.restore_moe_output(moe_diff, prune_state)

        result_flat = result_flat + moe_diff.to(result_flat.dtype)
        return result_flat.reshape(*result.shape).to(original_dtype)

    LoraLayer.lora_moe_forward = lora_moe_forward
    if original_shared_a_forward is not None:
        LoraLayer.lora_moe_forward_shared_a = lora_moe_forward_shared_a
    if original_ab_forward is not None:
        LoraLayer.lora_moe_forward_ab = lora_moe_forward_ab
    _PATCHED_LORAMOE = True
    return True


@contextmanager
def apply_visual_token_prune_to_loramoe_modules(model: nn.Module, plan: VisualTokenPrunePlan | None):
    if plan is None:
        yield
        return

    if not install_loramoe_visual_token_prune_patch():
        raise RuntimeError(
            "Visual token pruning requires PEFT LoraLayer.lora_moe_forward; "
            "the installed PEFT LoRA-MoE patch is incompatible."
        )
    plan.reset_prune_stats()
    patched_modules = []
    correction_modules = []
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_moe_forward"):
            module._visual_token_prune_plan = plan
            patched_modules.append(module)
            if module_name.rsplit(".", 1)[-1] == "q_proj":
                correction_modules.append(module)

    if not patched_modules:
        raise RuntimeError(
            "Visual token pruning is enabled, but no Transformer LoRA-MoE modules were found. "
            "Verify that the server policy applied LoRA-MoE before entering Phase 3."
        )

    if not correction_modules and patched_modules:
        correction_modules.append(patched_modules[0])
    for module in correction_modules:
        module._visual_token_prune_ot_correction_enabled = True
        if model.training:
            module._visual_token_prune_ot_enabled = True
    try:
        yield
    finally:
        for module in patched_modules:
            if hasattr(module, "_visual_token_prune_plan"):
                delattr(module, "_visual_token_prune_plan")
            if hasattr(module, "_visual_token_prune_ot_enabled"):
                delattr(module, "_visual_token_prune_ot_enabled")
            if hasattr(module, "_visual_token_prune_ot_correction_enabled"):
                delattr(module, "_visual_token_prune_ot_correction_enabled")
        plan.print_prune_summary()


@torch.compiler.disable
def build_visual_token_prune_plan(
    *,
    lang_tokens: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    image_token_id: int,
    cfg,
) -> VisualTokenPrunePlan | None:
    if not bool(getattr(cfg, "use_visual_token_prune", False)):
        return None

    visual_token_indices = []
    visual_token_counts = []
    for batch_idx in range(lang_tokens.shape[0]):
        tokens = (lang_tokens[batch_idx] == image_token_id).nonzero(as_tuple=False).flatten().tolist()
        visual_token_indices.append(tokens)
        visual_token_counts.append(len(tokens))

    if not any(visual_token_indices):
        return None

    return VisualTokenPrunePlan(
        prefix_pad_masks=prefix_pad_masks,
        visual_token_indices=visual_token_indices,
        visual_token_counts=visual_token_counts,
    )
