"""
Weight mapping functions for LoRA-MoE Federated Learning.

This module provides functions to map weights between:
- Standard LoRA (client mode)
- LoRA-MoE (server mode with multiple experts)
"""

import itertools

import torch
import torch.nn as nn
import torch.distributed as dist


def _is_shared_a_mode(model) -> bool:
    target_model = model.model if hasattr(model, 'model') else model
    return getattr(target_model, '_use_shared_a_moe', False)


def _get_shared_adapter_name(module) -> str:
    return getattr(module, '_shared_lora_adapter', 'shared')


def _iter_router_tensors(router):
    if router is None:
        return []
    if isinstance(router, nn.Module):
        return [param.data for param in router.parameters()]
    return [router.data]


def get_lora_weights(state_dict, adapter_name="default"):
    """Extract standard LoRA weights from state dict."""
    lora_A_key = f"lora_A.{adapter_name}.weight"
    lora_B_key = f"lora_B.{adapter_name}.weight"

    if lora_A_key in state_dict and lora_B_key in state_dict:
        return {
            "lora_A": state_dict[lora_A_key],
            "lora_B": state_dict[lora_B_key],
        }
    return None


def validate_complete_lora_state_dict(state_dict: dict, context: str = "LoRA state") -> int:
    """Validate that every serialized LoRA update contains a compatible A/B pair."""
    pairs = {}
    for key, value in state_dict.items():
        if "lora_A" in key:
            pair_id = key.replace("lora_A", "lora_factor", 1)
            pairs.setdefault(pair_id, {})["lora_A"] = (key, value)
        elif "lora_B" in key:
            pair_id = key.replace("lora_B", "lora_factor", 1)
            pairs.setdefault(pair_id, {})["lora_B"] = (key, value)

    incomplete = [pair_id for pair_id, pair in pairs.items() if set(pair) != {"lora_A", "lora_B"}]
    if incomplete:
        sample = ", ".join(incomplete[:3])
        raise ValueError(f"{context} contains incomplete LoRA adapters: {sample}")

    for pair_id, pair in pairs.items():
        _, lora_a = pair["lora_A"]
        _, lora_b = pair["lora_B"]
        _validate_lora_adapter({"lora_A": lora_a, "lora_B": lora_b}, f"{context}:{pair_id}")
    return len(pairs)


def init_expert_slots(model, num_experts: int, r: int = None, lora_alpha: int = None, lora_dropout: float = 0.0):
    """Initialize expert slots for all LoRA layers in the model.

    This creates adapter slots for all expert IDs (0, 1, ..., num_experts-1) on each LoRA layer.

    Args:
        model: The model with LoRA layers
        num_experts: Number of expert slots to create
        r: LoRA rank (if None, inferred from existing adapters)
        lora_alpha: LoRA alpha (if None, inferred from existing adapters)
        lora_dropout: LoRA dropout probability

    Returns:
        Number of layers that had expert slots created
    """
    import logging
    logger = logging.getLogger()

    layers_initialized = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'update_layer'):
            continue
        if not hasattr(module, 'lora_A'):
            continue

        # Get rank and alpha from existing adapter if not provided
        if r is None or lora_alpha is None:
            existing_r = list(module.r.values())[0] if module.r else 8
            existing_alpha = list(module.lora_alpha.values())[0] if module.lora_alpha else 8

            if r is None:
                r = existing_r
            if lora_alpha is None:
                lora_alpha = existing_alpha

        # Create expert adapters
        for exp_id in range(num_experts):
            exp_key = str(exp_id)
            if exp_key not in module.lora_A:
                try:
                    module.update_layer(
                        adapter_name=exp_key,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        init_lora_weights=True,
                        use_rslora=False,
                    )
                    # CRITICAL: Ensure all LoRA weights are float32 for stable training.
                    # nn.Linear creates weights in float32 by default, but if base layer is bfloat16,
                    # the weights might be converted. We need to keep them in float32.
                    if module.lora_A[exp_key].weight.dtype != torch.float32:
                        module.lora_A[exp_key].weight.data = module.lora_A[exp_key].weight.data.to(torch.float32)
                        module.lora_B[exp_key].weight.data = module.lora_B[exp_key].weight.data.to(torch.float32)
                except ValueError:
                    # Adapter already exists
                    pass

        layers_initialized += 1

    logger.info(f"[init_expert_slots] Initialized {num_experts} expert slots on {layers_initialized} layers")
    return layers_initialized


def init_lora_routers(model, num_experts: int, router_top_k: int = 2, router_hidden_dim: int | None = 16):
    """Initialize LoRA routers for all LoRA layers in the model.

    This initializes the router weights for all LoRA layers to enable MoE mode.

    Args:
        model: The model with LoRA layers
        num_experts: Number of experts for LoRA-MoE
        router_top_k: Top-K experts to select
        router_hidden_dim: Optional bottleneck dimension for MLP routers. If None, use direct routers.

    Returns:
        Number of layers that had routers initialized
    """
    import logging
    logger = logging.getLogger()

    layers_initialized = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'init_lora_router'):
            continue

        try:
            module.init_lora_router(num_experts=num_experts, top_k=router_top_k, hidden_dim=router_hidden_dim)
            layers_initialized += 1
        except Exception as e:
            logger.warning(f"[init_lora_routers] Failed to initialize router for {name}: {e}")

    logger.info(f"[init_lora_routers] Initialized routers on {layers_initialized} layers")
    return layers_initialized


def inject_shared_a_to_server(server_model, shared_a_state):
    """Inject a shared lora_A state dict into server model."""
    import logging
    logger = logging.getLogger()

    if not shared_a_state:
        return 0

    target_model = server_model.model if hasattr(server_model, 'model') else server_model
    inject_count = 0
    sample_logs = []

    for name, module in target_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'base_layer'):
            base_name = name.replace('.base_layer', '')
            if not base_name.startswith('model.'):
                state_key = f"model.{base_name}.lora_A.default.weight"
            else:
                state_key = f"{base_name}.lora_A.default.weight"
            if state_key in shared_a_state:
                shared_adapter = _get_shared_adapter_name(module)
                if shared_adapter in module.lora_A:
                    module.lora_A[shared_adapter].weight.data.copy_(shared_a_state[state_key].to(torch.float32))
                    inject_count += 1
                    if len(sample_logs) < 3:
                        sample_logs.append((state_key, tuple(module.lora_A[shared_adapter].weight.shape), float(module.lora_A[shared_adapter].weight.data.norm().item())))
        if hasattr(module, '_stack_lora_weights') and hasattr(module, 'lora_A'):
            module._stack_lora_weights()

    if hasattr(target_model, 'state_proj_lora_A_shared') and 'model.state_proj_lora_A.weight' in shared_a_state:
        target_model.state_proj_lora_A_shared.weight.data.copy_(shared_a_state['model.state_proj_lora_A.weight'].to(torch.float32))
        inject_count += 1
        if len(sample_logs) < 3:
            sample_logs.append(('model.state_proj_lora_A.weight', tuple(target_model.state_proj_lora_A_shared.weight.shape), float(target_model.state_proj_lora_A_shared.weight.data.norm().item())))

    if sample_logs:
        logger.info(f"[inject_shared_a_to_server] Injected shared A into {inject_count} layers")
        for key, shape, norm in sample_logs:
            logger.info(f"  shared_A {key}: shape={shape}, norm={norm:.6f}")

    return inject_count


def inject_client_b_to_expert(server_model, client_states, client_id):
    """Inject only client-specific lora_B weights into expert slots."""
    target_model = server_model.model if hasattr(server_model, 'model') else server_model
    if client_id not in client_states:
        return 0

    state_dict = client_states[client_id]
    exp_key = str(client_id)
    inject_count = 0

    for name, module in target_model.named_modules():
        if not hasattr(module, 'lora_B') or not hasattr(module, 'base_layer'):
            continue
        if exp_key not in module.lora_B:
            continue
        base_name = name.replace('.base_layer', '')
        if not base_name.startswith('model.'):
            b_key = f"model.{base_name}.lora_B.default.weight"
        else:
            b_key = f"{base_name}.lora_B.default.weight"
        if b_key in state_dict:
            module.lora_B[exp_key].weight.data.copy_(state_dict[b_key].to(torch.float32))
            inject_count += 1
        if _is_shared_a_mode(target_model):
            shared_adapter = _get_shared_adapter_name(module)
            if shared_adapter in module.lora_A and exp_key in module.lora_A:
                module.lora_A[exp_key].weight.data.copy_(module.lora_A[shared_adapter].weight.data)
        if hasattr(module, '_stack_lora_weights'):
            module._stack_lora_weights()

    if hasattr(target_model, 'state_proj_lora_B_moe') and client_id < len(target_model.state_proj_lora_B_moe):
        b_key = 'model.state_proj_lora_B.weight'
        if b_key in state_dict:
            target_model.state_proj_lora_B_moe[client_id].weight.data.copy_(state_dict[b_key].to(torch.float32))
            inject_count += 1

    return inject_count


def inject_state_to_expert(server_model, client_states, client_id, create_slots_if_missing=True):
    """Inject client states (saved weights) to server model's expert slot.

    This function extracts weights from client_states dict instead of client_model,
    ensuring each client gets their own weights even after multiple clients
    have trained on the same client_model.

    Args:
        server_model: The model with LoRA-MoE layers (QwenA1Policy wrapper)
        client_states: Dict of client_id -> state_dict with trainable weights
        client_id: The expert slot ID to inject into
        create_slots_if_missing: If True, create expert slots if they don't exist
    """
    import logging
    logger = logging.getLogger()

    exp_key = str(client_id)

    if client_id not in client_states:
        logger.warning(f"[inject_state_to_expert] No state found for client {client_id}")
        return

    state_dict = client_states[client_id]
    logger.info(f"[inject_state_to_expert] Injecting weights for client {client_id}, {len(state_dict)} parameters")

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(server_model, 'model'):
        target_model = server_model.model
    else:
        target_model = server_model

    # Print sample state_dict keys for debugging
    sample_keys = list(state_dict.keys())[:10]
    logger.info(f"[inject_state_to_expert] state_dict sample keys: {sample_keys}")

    # Collect all LoRA keys in the state_dict, indexed by base layer name
    # state_dict format: {full_path}.lora_A.default.weight or {full_path}.lora_A.{exp_id}.weight
    state_dict_lora_keys = {}  # layer_path -> {'lora_A': key, 'lora_B': key}
    for key in state_dict.keys():
        if '.lora_A.' in key:
            # Extract the base layer name: strip .lora_A.{adapter}.weight
            parts = key.split('.lora_A.')
            if len(parts) == 2:
                layer_path = parts[0]
                if layer_path not in state_dict_lora_keys:
                    state_dict_lora_keys[layer_path] = {}
                state_dict_lora_keys[layer_path]['lora_A'] = key
        elif '.lora_B.' in key:
            parts = key.split('.lora_B.')
            if len(parts) == 2:
                layer_path = parts[0]
                if layer_path not in state_dict_lora_keys:
                    state_dict_lora_keys[layer_path] = {}
                state_dict_lora_keys[layer_path]['lora_B'] = key

    logger.info(f"[inject_state_to_expert] state_dict has {len(state_dict_lora_keys)} layers with LoRA")
    logger.info(f"[inject_state_to_expert] Sample layer_paths in state_dict: {list(state_dict_lora_keys.keys())[:3]}")

    # Iterate over all LoRA layers in target_model
    inject_count = 0
    expert_lora_a_count = 0
    expert_lora_b_count = 0
    matched_layers = 0
    unmatched_modules = 0
    sample_unmatched = []

    for name, module in target_model.named_modules():
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        if not hasattr(module, 'base_layer'):
            continue

        # Get the module base layer name (strip base_layer)
        # module name: model.xxx.yyy.base_layer
        base_name = name.replace('.base_layer', '')

        # Find the matching layer in the state_dict
        matched_layer_path = None
        for layer_path, lora_keys in state_dict_lora_keys.items():
            # layer_path may or may not carry the model. prefix
            # base_name usually carries the model. prefix
            if layer_path == base_name:
                matched_layer_path = layer_path
                break
            if layer_path == base_name.replace('model.', '', 1):
                matched_layer_path = layer_path
                break
            # Also check whether layer_path is a substring of base_name
            if base_name.endswith(layer_path) or layer_path.endswith(base_name):
                matched_layer_path = layer_path
                break

        if matched_layer_path is None:
            unmatched_modules += 1
            if len(sample_unmatched) < 5:
                sample_unmatched.append(base_name)
            continue

        matched_layers += 1
        lora_keys = state_dict_lora_keys[matched_layer_path]

        if set(lora_keys) != {'lora_A', 'lora_B'}:
            raise ValueError(f"[inject_state_to_expert] Incomplete LoRA adapter for {matched_layer_path}")

        source_a = state_dict[lora_keys['lora_A']]
        source_b = state_dict[lora_keys['lora_B']]
        target_a = module.lora_A[exp_key].weight.data
        target_b = module.lora_B[exp_key].weight.data
        if source_a.shape != target_a.shape or source_b.shape != target_b.shape:
            raise ValueError(
                f"[inject_state_to_expert] LoRA shape mismatch for {name}: "
                f"target A/B={tuple(target_a.shape)}/{tuple(target_b.shape)}, "
                f"source A/B={tuple(source_a.shape)}/{tuple(source_b.shape)}"
            )
        target_a.copy_(source_a.to(torch.float32))
        target_b.copy_(source_b.to(torch.float32))
        inject_count += 2
        expert_lora_a_count += 1
        expert_lora_b_count += 1

    logger.info(f"[inject_state_to_expert] Matched layers: {matched_layers}, Unmatched modules: {unmatched_modules}")
    if sample_unmatched:
        logger.info(f"[inject_state_to_expert] Sample unmatched module names: {sample_unmatched}")
    logger.info(f"[inject_state_to_expert] Injected {inject_count} weights (lora_A: {expert_lora_a_count}, lora_B: {expert_lora_b_count}) to expert {client_id}")

    # NOTE: _stack_lora_weights() is NOT called here.
    # Caller should call it ONCE after all expert injections are complete.
    # See Phase 2 in lerobot_fl_robotwin_moe.py for proper usage.

    # Handle state_proj injection separately (ModuleList with integer indices)
    # client state_dict format: model.state_proj_lora_A.weight (standard LoRA)
    # server format: model.state_proj_lora_A_moe.{client_id}.weight (LoRA-MoE)
    has_state_proj_moe = hasattr(target_model, 'state_proj_lora_A_moe') and hasattr(target_model, 'state_proj_lora_B_moe')
    logger.info(f"[inject_state_to_expert] target_model has state_proj_lora_A_moe: {has_state_proj_moe}")

    if has_state_proj_moe:
        num_experts = len(target_model.state_proj_lora_A_moe)
        logger.info(f"[inject_state_to_expert] num_experts: {num_experts}, client_id: {client_id}")
        if client_id < num_experts:
            # Look for state_proj LoRA keys in the client state_dict
            # Standard format: model.state_proj_lora_A.weight
            lora_A_key = None
            lora_B_key = None
            for key in state_dict:
                if 'state_proj' in key and 'lora_A' in key and key.endswith('.weight'):
                    # Skip MoE-format keys (those are for the server)
                    if 'lora_A_moe' not in key:
                        lora_A_key = key
                        break
            for key in state_dict:
                if 'state_proj' in key and 'lora_B' in key and key.endswith('.weight'):
                    if 'lora_B_moe' not in key:
                        lora_B_key = key
                        break

            logger.info(f"[inject_state_to_expert] Found state_proj keys: lora_A={lora_A_key}, lora_B={lora_B_key}")

            if bool(lora_A_key) != bool(lora_B_key):
                raise ValueError("[inject_state_to_expert] Incomplete state_proj LoRA adapter")
            if lora_A_key and lora_B_key:
                target_a = target_model.state_proj_lora_A_moe[client_id].weight.data
                target_b = target_model.state_proj_lora_B_moe[client_id].weight.data
                source_a = state_dict[lora_A_key]
                source_b = state_dict[lora_B_key]
                if source_a.shape != target_a.shape or source_b.shape != target_b.shape:
                    raise ValueError(
                        "[inject_state_to_expert] state_proj LoRA shape mismatch: "
                        f"target A/B={tuple(target_a.shape)}/{tuple(target_b.shape)}, "
                        f"source A/B={tuple(source_a.shape)}/{tuple(source_b.shape)}"
                    )
                target_a.copy_(source_a.to(torch.float32))
                target_b.copy_(source_b.to(torch.float32))
                logger.info(f"[inject_state_to_expert] Injected complete state_proj LoRA to expert {client_id}")


def load_client_state_to_model(client_model, client_states, client_id):
    """Load client states (saved weights) to client model for training.

    This function loads weights from client_states dict to client_model,
    ensuring each client uses their own weights (from previous round's mixing).

    Args:
        client_model: The model with standard LoRA layers
        client_states: Dict of client_id -> state_dict with trainable weights
        client_id: The client ID to load weights for
    """
    import logging
    logger = logging.getLogger()

    if client_id not in client_states:
        logger.warning(f"[load_client_state_to_model] No state found for client {client_id}, using initial weights")
        return

    state_dict = client_states[client_id]
    logger.info(f"[load_client_state_to_model] Loading weights for client {client_id}, {len(state_dict)} parameters in state_dict")

    # Get model's parameter names
    model_param_names = set(name for name, _ in client_model.named_parameters())

    # Check key format mismatch
    if len(state_dict) > 0:
        sample_key = list(state_dict.keys())[0]
        logger.info(f"[load_client_state_to_model] Sample state_dict key: {sample_key}")
        logger.info(f"[load_client_state_to_model] Sample model param: {list(model_param_names)[:3]}")

        # Find matching and non-matching keys
        matching_keys = [k for k in state_dict.keys() if k in model_param_names]
        non_matching_keys = [k for k in state_dict.keys() if k not in model_param_names]
        logger.info(f"[load_client_state_to_model] Matching keys: {len(matching_keys)}, Non-matching: {len(non_matching_keys)}")
        if non_matching_keys:
            logger.info(f"[load_client_state_to_model] Non-matching sample: {non_matching_keys[:5]}")

    # Load weights to client_model
    load_count = 0
    skip_count = 0

    # Debug: log client_policy param names for comparison
    model_param_names = [n for n, _ in client_model.named_parameters()]
    if logger and len(model_param_names) > 0:
        lora_params = [n for n in model_param_names if 'lora_A' in n or 'lora_B' in n]
        logger.info(f"[load_client_state_to_model] client_policy has {len(lora_params)} LoRA params")
        logger.info(f"[load_client_state_to_model] Sample client_policy LoRA params: {lora_params[:3]}")
        logger.info(f"[load_client_state_to_model] Sample state_dict keys: {list(state_dict.keys())[:3]}")
        # Check direct match
        direct_matches = sum(1 for k in state_dict.keys() if k in model_param_names)
        logger.info(f"[load_client_state_to_model] Direct matches: {direct_matches}/{len(state_dict)}")

    # Preprocess state_dict: build mapping from base layer to data
    # Handle two formats: .lora_A.default.weight and .lora_A.{client_id}.weight
    base_to_data = {}  # base_layer_path -> {lora_A: tensor, lora_B: tensor}
    for state_key, state_value in state_dict.items():
        # Extract base layer name
        base = None
        for suffix in ['.lora_A.default.weight', '.lora_B.default.weight',
                       '.lora_A.0.weight', '.lora_B.0.weight',
                       '.lora_A.1.weight', '.lora_B.1.weight',
                       '.lora_A.2.weight', '.lora_B.2.weight']:
            if state_key.endswith(suffix):
                base = state_key[:-len(suffix)]
                break
        if base is None:
            continue

        if base not in base_to_data:
            base_to_data[base] = {}
        if '.lora_A.' in state_key:
            base_to_data[base]['lora_A'] = state_value
        elif '.lora_B.' in state_key:
            base_to_data[base]['lora_B'] = state_value

    logger.info(f"[load_client_state_to_model] Preprocessed {len(base_to_data)} base layers from state_dict")

    # Debug: check base_to_data format
    if logger and len(base_to_data) > 0:
        sample_base = list(base_to_data.keys())[0]
        logger.info(f"[load_client_state_to_model] Sample base in base_to_data: {sample_base}")

    for name, param in client_model.named_parameters():
        if not param.requires_grad:
            continue

        if name in state_dict:
            # Direct match
            param.data.copy_(state_dict[name].to(param.device))
            load_count += 1
        elif name == 'model.state_proj_lora_A.weight' and 'model.state_proj_lora_A.weight' in state_dict:
            param.data.copy_(state_dict['model.state_proj_lora_A.weight'].to(param.device))
            load_count += 1
        elif name == 'model.state_proj_lora_B.weight' and 'model.state_proj_lora_B.weight' in state_dict:
            param.data.copy_(state_dict['model.state_proj_lora_B.weight'].to(param.device))
            load_count += 1
        else:
            # Try fuzzy matching
            matched = False
            # Extract the base layer name of the model parameter
            model_base = None
            for suffix in ['.lora_A.default.weight', '.lora_B.default.weight']:
                if name.endswith(suffix):
                    model_base = name[:-len(suffix)]
                    break

            if model_base and model_base in base_to_data:
                # Check lora_A and lora_B
                if '.lora_A.' in name and 'lora_A' in base_to_data[model_base]:
                    param.data.copy_(base_to_data[model_base]['lora_A'].to(param.device))
                    load_count += 1
                    matched = True
                elif '.lora_B.' in name and 'lora_B' in base_to_data[model_base]:
                    param.data.copy_(base_to_data[model_base]['lora_B'].to(param.device))
                    load_count += 1
                    matched = True

            if not matched:
                skip_count += 1

    logger.info(f"[load_client_state_to_model] Loaded {load_count} parameters for client {client_id}, skipped {skip_count}")


def inject_standard_to_expert(server_model, client_model, client_id):
    """Inject client model LoRA weights to server model's expert slot.

    Args:
        server_model: The model with LoRA-MoE layers
        client_model: The model with standard LoRA (source of weights)
        client_id: The expert slot ID to inject into
    """
    import logging
    logger = logging.getLogger()

    exp_key = str(client_id)
    logger.info(f"[inject_standard_to_expert] Starting for client {client_id}")

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(server_model, 'model'):
        target_model = server_model.model
    else:
        target_model = server_model

    # Get the inner client model
    if hasattr(client_model, 'model'):
        client_inner = client_model.model
    else:
        client_inner = client_model

    # Collect LoRA weights from client model
    client_lora_A = {}
    client_lora_B = {}

    for name, module in client_inner.named_modules():
        # Check for PEFT LoRA layer
        if not hasattr(module, 'base_layer'):
            continue
        if not hasattr(module, 'lora_A') or not hasattr(module.lora_A, 'default'):
            continue
        if not hasattr(module.lora_A.default, 'weight'):
            continue

        key = name.replace('.base_layer', '')
        client_lora_A[key] = module.lora_A.default.weight.data
        client_lora_B[key] = module.lora_B.default.weight.data

    logger.info(f"[inject_standard_to_expert] Collected {len(client_lora_A)} LoRA layers from client")

    # Inject into server model - PEFT LoRA-MoE uses lora_A/lora_B ModuleDict
    inject_count = 0
    for name, module in target_model.named_modules():
        if not hasattr(module, 'lora_A'):
            continue
        if not hasattr(module, 'lora_B'):
            continue
        if not hasattr(module, 'base_layer'):
            continue
        if exp_key not in module.lora_A:
            continue

        base_name = name.replace('.base_layer', '')

        if base_name in client_lora_A:
            # Ensure float32 for stable training
            module.lora_A[exp_key].weight.data.copy_(client_lora_A[base_name].to(torch.float32))
            inject_count += 1
        if base_name in client_lora_B:
            # Ensure float32 for stable training
            module.lora_B[exp_key].weight.data.copy_(client_lora_B[base_name].to(torch.float32))

    # IMPORTANT: Update stacked weights after modifying expert weights
    for name, module in target_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, '_stack_lora_weights'):
            module._stack_lora_weights()

    logger.info(f"[inject_standard_to_expert] Injected {inject_count} layers to expert {client_id}")


def inject_weights_to_expert(model, client_id, weights_dict, layer_name=None):
    """Inject pre-mixed weights dict to expert slot.

    Args:
        model: The model with LoRA-MoE layers (can be QwenA1Policy wrapper)
        client_id: The expert slot ID to inject into
        weights_dict: Dict with 'lora_A' and 'lora_B' tensors
        layer_name: Optional layer name to match. Supports both:
                   - Full path from model.named_modules() (e.g., 'base_model.model.layers.0.q_proj')
                   - Relative path from model.model.named_modules() (e.g., 'model.layers.0.q_proj' or 'layers.0.q_proj')
                   If None, use shape matching (legacy behavior).
    """
    exp_key = str(client_id)

    if 'lora_A' not in weights_dict or 'lora_B' not in weights_dict:
        return 0

    lora_A_data = weights_dict['lora_A']
    lora_B_data = weights_dict['lora_B']

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    # Find corresponding layers in model and inject
    inject_count = 0
    for name, module in target_model.named_modules():
        # Check if this is a LoRA layer with MoE support
        # New PEFT LoRA-MoE: uses lora_A/lora_B ModuleDict
        if not hasattr(module, 'lora_A'):
            continue
        if not hasattr(module, 'lora_B'):
            continue
        if not hasattr(module, 'base_layer'):
            continue

        # If layer_name is provided, match against both full path and relative path
        if layer_name is not None:
            module_name_full = name.replace('.base_layer', '')

            match = False
            if module_name_full == layer_name:
                match = True
            elif module_name_full.endswith('.' + layer_name):
                match = True
            elif '.' + layer_name in module_name_full:
                match = True

            if not match:
                continue

        # Check if expert slot exists
        if exp_key not in module.lora_A:
            continue

        # Match by shape to ensure correct dimensions
        # Ensure float32 for stable training
        if module.lora_A[exp_key].weight.shape == lora_A_data.shape:
            module.lora_A[exp_key].weight.data.copy_(lora_A_data.to(torch.float32))
            inject_count += 1
        if module.lora_B[exp_key].weight.shape == lora_B_data.shape:
            module.lora_B[exp_key].weight.data.copy_(lora_B_data.to(torch.float32))

    # NOTE: _stack_lora_weights() is NOT called here to avoid over-calling.
    # Caller should call it ONCE after all expert weights are injected.
    # See Phase 5 in lerobot_fl_robotwin_moe.py for proper usage.

    return inject_count


def extract_expert_to_standard(model, client_id):
    """Extract expert weights from LoRA-MoE to standard LoRA format.

    Args:
        model: The model with LoRA-MoE layers
        client_id: The expert slot ID to extract from

    Returns:
        Dict with 'lora_A' and 'lora_B' tensors (per-layer dict format)
    """
    exp_key = str(client_id)
    layer_weights = {}

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    for name, module in target_model.named_modules():
        # PEFT LoRA-MoE uses lora_A/lora_B ModuleDict with string keys
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if exp_key in module.lora_A and exp_key in module.lora_B:
                lora_A = module.lora_A[exp_key].weight.data.clone()
                lora_B = module.lora_B[exp_key].weight.data.clone()
                layer_weights[name] = {
                    "lora_A": lora_A,
                    "lora_B": lora_B,
                }

    if layer_weights:
        return layer_weights
    return None


def _normalize_expert_weights(weights: torch.Tensor, num_experts: int, device: torch.device) -> torch.Tensor:
    weights = weights.to(device=device, dtype=torch.float32).flatten()
    if weights.numel() != num_experts:
        raise ValueError(f"Expected {num_experts} expert weights, got {weights.numel()}")
    if not bool(torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0:
        raise ValueError("Expert weights must be finite with a positive sum")
    if bool((weights < 0).any()):
        raise ValueError("Expert weights must be non-negative")
    return weights / weights.sum().clamp_min(1e-12)


def get_lora_adapter_scaling(module, adapter_name: str) -> float:
    """Return the effective PEFT scaling for one adapter."""
    scaling = getattr(module, "scaling", 1.0)
    if isinstance(scaling, dict):
        scaling = scaling.get(adapter_name, 1.0)
    if isinstance(scaling, torch.Tensor):
        scaling = scaling.detach().item()
    return float(scaling)


def _validate_lora_adapter(adapter: dict, label: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    if "lora_A" not in adapter or "lora_B" not in adapter:
        raise ValueError(f"{label} must contain both lora_A and lora_B")
    lora_a = adapter["lora_A"]
    lora_b = adapter["lora_B"]
    if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
        raise TypeError(f"{label} LoRA factors must be tensors")
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError(
            f"{label} has incompatible LoRA shapes: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
        )
    scaling = float(adapter.get("scaling", 1.0))
    if not bool(torch.isfinite(torch.tensor(scaling))) or scaling <= 0:
        raise ValueError(f"{label} has invalid LoRA scaling {scaling}")
    return lora_a, lora_b, scaling


@torch.no_grad()
def average_complete_lora_adapters(
    adapters: list[dict],
    weights: torch.Tensor | None = None,
) -> dict:
    """Average LoRA update matrices and return their best rank-r factorization."""
    if not adapters:
        raise ValueError("At least one complete LoRA adapter is required")

    validated = [_validate_lora_adapter(adapter, f"adapter[{index}]") for index, adapter in enumerate(adapters)]
    first_a, first_b, first_scaling = validated[0]
    device = first_a.device

    for index, (lora_a, lora_b, scaling) in enumerate(validated[1:], start=1):
        if lora_a.shape != first_a.shape or lora_b.shape != first_b.shape:
            raise ValueError(
                f"adapter[{index}] shapes do not match adapter[0]: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )
        if scaling != first_scaling:
            raise ValueError(
                f"adapter[{index}] scaling {scaling} does not match adapter[0] scaling {first_scaling}"
            )

    if weights is None:
        normalized_weights = torch.full(
            (len(validated),), 1.0 / len(validated), device=device, dtype=torch.float32
        )
    else:
        normalized_weights = _normalize_expert_weights(weights, len(validated), device)

    sqrt_weights = normalized_weights.sqrt()
    left_factors = torch.cat(
        [
            lora_b.to(device=device, dtype=torch.float32) * sqrt_weights[index]
            for index, (_, lora_b, _) in enumerate(validated)
        ],
        dim=1,
    )
    right_factors = torch.cat(
        [
            lora_a.to(device=device, dtype=torch.float32) * sqrt_weights[index]
            for index, (lora_a, _, _) in enumerate(validated)
        ],
        dim=0,
    )

    q_left, r_left = torch.linalg.qr(left_factors, mode="reduced")
    q_right, r_right = torch.linalg.qr(right_factors.transpose(0, 1), mode="reduced")
    core = r_left @ r_right.transpose(0, 1)
    u, singular_values, vh = torch.linalg.svd(core, full_matrices=False)

    target_rank = first_a.shape[0]
    kept_rank = min(target_rank, singular_values.numel())
    singular_root = singular_values[:kept_rank].sqrt()
    lora_b = (q_left @ u[:, :kept_rank]) * singular_root[None, :]
    lora_a = singular_root[:, None] * (vh[:kept_rank] @ q_right.transpose(0, 1))

    if kept_rank < target_rank:
        lora_b = torch.cat(
            [lora_b, lora_b.new_zeros(lora_b.shape[0], target_rank - kept_rank)],
            dim=1,
        )
        lora_a = torch.cat(
            [lora_a, lora_a.new_zeros(target_rank - kept_rank, lora_a.shape[1])],
            dim=0,
        )

    return {
        "lora_A": lora_a,
        "lora_B": lora_b,
        "scaling": first_scaling,
    }


def aggregate_experts_to_global(
    model,
    expert_weights: dict[str, torch.Tensor] | None = None,
):
    """Aggregate all expert weights to create a global standard LoRA.

    Returns:
        Per-layer complete LoRA records with paired factors and scaling.
    """
    target_model = model.model if hasattr(model, 'model') else model

    if _is_shared_a_mode(target_model):
        return {}

    # Collect weights per layer (module name)
    layer_weights = {}

    for name, module in target_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            expert_keys = sorted(
                (key for key in module.lora_A if key in module.lora_B and str(key).isdigit()),
                key=lambda key: int(key),
            )
            if len(expert_keys) > 1:
                adapters = [
                    {
                        "lora_A": module.lora_A[key].weight.data,
                        "lora_B": module.lora_B[key].weight.data,
                        "scaling": get_lora_adapter_scaling(module, key),
                    }
                    for key in expert_keys
                ]
                weights = expert_weights.get(name) if expert_weights is not None else None
                layer_weights[name] = average_complete_lora_adapters(
                    adapters,
                    weights=weights,
                )

    if hasattr(target_model, 'state_proj_lora_A_moe') and hasattr(target_model, 'state_proj_lora_B_moe'):
        num_experts = len(target_model.state_proj_lora_A_moe)
        if num_experts > 0:
            adapters = []
            state_proj_scaling = float(getattr(target_model, "state_proj_lora_scaling", 1.0))
            for e in range(num_experts):
                adapters.append(
                    {
                        "lora_A": target_model.state_proj_lora_A_moe[e].weight.data,
                        "lora_B": target_model.state_proj_lora_B_moe[e].weight.data,
                        "scaling": state_proj_scaling,
                    }
                )

            if adapters:
                weights = expert_weights.get("model.state_proj") if expert_weights is not None else None
                layer_weights["model.state_proj"] = average_complete_lora_adapters(
                    adapters,
                    weights=weights,
                )

    return layer_weights


def mix_local_and_global(model, client_id, global_weights, mix_ratio=0.5):
    """Directly average complete local and global LoRA adapters.

    Args:
        model: The model with LoRA-MoE layers
        client_id: The expert slot ID to mix
        global_weights: Dict with global 'lora_A' and 'lora_B'
        mix_ratio: Weight for local expert (default 0.5)
    """
    exp_key = str(client_id)
    if not 0.0 <= float(mix_ratio) <= 1.0:
        raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    modules_to_update = []
    for module in target_model.modules():
        # PEFT LoRA-MoE: lora_A/lora_B ModuleDict with string keys
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if exp_key in module.lora_A and exp_key in module.lora_B:
                local_A = module.lora_A[exp_key].weight
                local_B = module.lora_B[exp_key].weight
                local_scaling = get_lora_adapter_scaling(module, exp_key)
                merged = average_complete_lora_adapters(
                    [
                        {
                            "lora_A": local_A.data,
                            "lora_B": local_B.data,
                            "scaling": local_scaling,
                        },
                        {
                            "lora_A": global_weights["lora_A"].to(local_A.device),
                            "lora_B": global_weights["lora_B"].to(local_B.device),
                            "scaling": float(global_weights.get("scaling", local_scaling)),
                        },
                    ],
                    weights=torch.tensor([mix_ratio, 1.0 - mix_ratio], device=local_A.device),
                )
                local_A.data.copy_(merged["lora_A"])
                local_B.data.copy_(merged["lora_B"])
                modules_to_update.append(module)

    # IMPORTANT: Update stacked weights after modifying expert weights
    for module in modules_to_update:
        if hasattr(module, '_stack_lora_weights'):
            module._stack_lora_weights()


def _broadcast_lora_adapter(lora_a: torch.Tensor, lora_b: torch.Tensor, src_rank: int) -> None:
    """Broadcast one complete LoRA adapter as a single packed payload."""
    if lora_a.device != lora_b.device or lora_a.dtype != lora_b.dtype:
        raise ValueError("LoRA factors must share device and dtype for atomic broadcast")
    packed = torch.cat((lora_a.reshape(-1), lora_b.reshape(-1)))
    dist.broadcast(packed, src=src_rank)
    split = lora_a.numel()
    lora_a.copy_(packed[:split].reshape_as(lora_a))
    lora_b.copy_(packed[split:].reshape_as(lora_b))


def sync_expert_weights(model, num_clients, world_size):
    """Broadcast all expert weights across all ranks.

    All ranks must participate in all broadcasts to avoid deadlock.
    """
    if not dist.is_initialized() or world_size <= 1:
        return

    target_model = model.model if hasattr(model, 'model') else model
    shared_a_mode = _is_shared_a_mode(target_model)

    for expert_id in range(num_clients):
        src_rank = expert_id % world_size
        exp_key = str(expert_id)

        complete_adapters = []
        expert_params = []
        for module in target_model.modules():
            has_a = hasattr(module, 'lora_A') and exp_key in module.lora_A
            has_b = hasattr(module, 'lora_B') and exp_key in module.lora_B
            if not shared_a_mode:
                if has_a != has_b:
                    raise ValueError(f"Expert {expert_id} contains an incomplete LoRA adapter")
                if has_a:
                    complete_adapters.append(
                        (module.lora_A[exp_key].weight.data, module.lora_B[exp_key].weight.data)
                    )
            elif has_b:
                expert_params.append(module.lora_B[exp_key].weight.data)
            if expert_id == 0 and shared_a_mode and hasattr(module, 'lora_A') and getattr(module, '_use_shared_lora_a', False):
                shared_adapter = _get_shared_adapter_name(module)
                if shared_adapter in module.lora_A:
                    expert_params.append(module.lora_A[shared_adapter].weight.data)
            if expert_id == 0 and hasattr(module, 'lora_router') and module.lora_router is not None:
                expert_params.extend(_iter_router_tensors(module.lora_router))

        for lora_a, lora_b in complete_adapters:
            _broadcast_lora_adapter(lora_a, lora_b, src_rank)
        for param in expert_params:
            dist.broadcast(param, src=src_rank)

        if shared_a_mode:
            if hasattr(target_model, 'state_proj_lora_A_shared') and expert_id == 0:
                dist.broadcast(target_model.state_proj_lora_A_shared.weight.data, src=src_rank)
            if hasattr(target_model, 'state_proj_lora_B_moe') and expert_id < len(target_model.state_proj_lora_B_moe):
                dist.broadcast(target_model.state_proj_lora_B_moe[expert_id].weight.data, src=src_rank)
        else:
            if hasattr(target_model, 'state_proj_lora_A_moe') and hasattr(target_model, 'state_proj_lora_B_moe'):
                if expert_id < len(target_model.state_proj_lora_A_moe):
                    _broadcast_lora_adapter(
                        target_model.state_proj_lora_A_moe[expert_id].weight.data,
                        target_model.state_proj_lora_B_moe[expert_id].weight.data,
                        src_rank,
                    )

        if expert_id == 0 and hasattr(target_model, 'state_proj_router'):
            for param in _iter_router_tensors(target_model.state_proj_router):
                dist.broadcast(param, src=src_rank)


def sync_all_experts_allreduce(model):
    """AllReduce sync all expert weights across all ranks (for averaging).

    After MoE training, use this to sync updated weights.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    world_size = dist.get_world_size()
    target_model = model.model if hasattr(model, 'model') else model
    shared_a_mode = _is_shared_a_mode(target_model)

    expert_params = []
    for module in target_model.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if hasattr(module, 'lora_router') and module.lora_router is not None:
                expert_params.extend(_iter_router_tensors(module.lora_router))

            if shared_a_mode and getattr(module, '_use_shared_lora_a', False):
                shared_adapter = _get_shared_adapter_name(module)
                if shared_adapter in module.lora_A:
                    expert_params.append(module.lora_A[shared_adapter].weight.data)
            else:
                for exp_key in module.lora_A:
                    expert_params.append(module.lora_A[exp_key].weight.data)

            for exp_key in module.lora_B:
                expert_params.append(module.lora_B[exp_key].weight.data)

    for param in expert_params:
        dist.all_reduce(param, op=dist.ReduceOp.SUM)
        param.div_(world_size)

    if shared_a_mode:
        if hasattr(target_model, 'state_proj_lora_A_shared'):
            dist.all_reduce(target_model.state_proj_lora_A_shared.weight.data, op=dist.ReduceOp.SUM)
            target_model.state_proj_lora_A_shared.weight.data.div_(world_size)
        if hasattr(target_model, 'state_proj_lora_B_moe'):
            for e in range(len(target_model.state_proj_lora_B_moe)):
                dist.all_reduce(target_model.state_proj_lora_B_moe[e].weight.data, op=dist.ReduceOp.SUM)
                target_model.state_proj_lora_B_moe[e].weight.data.div_(world_size)
    else:
        if hasattr(target_model, 'state_proj_lora_A_moe') and hasattr(target_model, 'state_proj_lora_B_moe'):
            for e in range(len(target_model.state_proj_lora_A_moe)):
                dist.all_reduce(target_model.state_proj_lora_A_moe[e].weight.data, op=dist.ReduceOp.SUM)
                target_model.state_proj_lora_A_moe[e].weight.data.div_(world_size)
                dist.all_reduce(target_model.state_proj_lora_B_moe[e].weight.data, op=dist.ReduceOp.SUM)
                target_model.state_proj_lora_B_moe[e].weight.data.div_(world_size)

    if hasattr(target_model, 'state_proj_router'):
        for param in _iter_router_tensors(target_model.state_proj_router):
            dist.all_reduce(param, op=dist.ReduceOp.SUM)
            param.div_(world_size)
