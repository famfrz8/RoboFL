import math

import torch


def foresight_consensus(
    q_und: torch.Tensor,
    q_gen: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return geometric teacher, path agreement, specificity, and reliability."""
    consensus = torch.sqrt(q_und.clamp_min(eps) * q_gen.clamp_min(eps))
    agreement = consensus.sum(dim=-1)
    teacher = consensus / agreement[:, None].clamp_min(eps)
    entropy = -(teacher * teacher.clamp_min(eps).log()).sum(dim=-1)
    specificity = (1.0 - entropy / math.log(float(teacher.shape[-1]))).clamp_min(0.0)
    reliability = agreement * specificity
    return teacher, agreement, specificity, reliability


def three_path_consensus(
    q_und: torch.Tensor,
    q_gen: torch.Tensor,
    q_act: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return the symmetric Hellinger barycenter evidence of all three paths."""
    del eps
    mean_root = (
        q_und.clamp_min(0.0).sqrt()
        + q_gen.clamp_min(0.0).sqrt()
        + q_act.clamp_min(0.0).sqrt()
    ) / 3.0
    return mean_root.square()


def path_consensus_expert_weights(
    score_sum: torch.Tensor,
    sample_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize pure PCEA evidence and return its mean path-agreement mass."""
    total_score = score_sum.sum()
    mean_agreement = total_score / sample_count
    return score_sum / total_score, mean_agreement


def smooth_and_impute_pcea_weights(
    expert_weights: dict[str, torch.Tensor],
    eligible_modules: set[str],
    module_agreements: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str], torch.Tensor]:
    """Smooth modules toward global PCEA in Hellinger geometry and fill missing ones."""
    if not expert_weights:
        raise ValueError("Cannot impute PCEA weights without observed path-consensus evidence")
    completed = dict(expert_weights)
    missing_modules = sorted(eligible_modules - set(completed))

    module_names = sorted(completed)
    agreement_weights = torch.tensor(
        [float((module_agreements or {}).get(name, 1.0)) for name in module_names],
        dtype=torch.float32,
    )
    stacked_weights = torch.stack([completed[name].float() for name in module_names])
    global_consensus = (agreement_weights[:, None] * stacked_weights).sum(dim=0)
    global_consensus = global_consensus / global_consensus.sum()

    global_root = global_consensus.clamp_min(0.0).sqrt()
    for module_name in module_names:
        module_root = completed[module_name].float().clamp_min(0.0).sqrt()
        midpoint = (module_root + global_root).square()
        completed[module_name] = midpoint / midpoint.sum()
    for module_name in missing_modules:
        completed[module_name] = global_consensus.clone()
    return completed, missing_modules, global_consensus
