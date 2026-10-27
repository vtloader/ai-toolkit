"""Tensor parallelism for the MOSS-Image DiT on Ascend NPUs.

Only the transformer's four per-block linears are sharded, which is the
standard Megatron-style layout:

    qkv / ffn_up_gate : column parallel (output split, input replicated)
    o   / ffn_down    : row parallel    (input split, output all-reduced)

The residual stream, norms, modulation, embeddings and the text refiner stay
replicated, so only the two attention/FFN projections need collectives. LoRA
adapters are sharded to match their base linear: ``lora_up`` rows for column
parallel layers and ``lora_down`` columns for row parallel layers.

This module is only active when ``AITK_TP=1`` and the process group is
initialized (torchrun with one process per NPU). The single-device path never
imports any distributed state into the model.
"""

import os
import torch
import torch.distributed as dist

_TP_FLAG = os.environ.get("AITK_TP", "0") == "1"


def is_tp_enabled() -> bool:
    if not _TP_FLAG:
        return False
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def tp_rank() -> int:
    return dist.get_rank() if is_tp_enabled() else 0


def tp_world_size() -> int:
    if is_tp_enabled():
        return dist.get_world_size()
    return int(os.environ.get("AITK_TP_SIZE", "1") or 1)


class _AllReduceForward(torch.autograd.Function):
    """Row-parallel output: sum partial products in forward, pass grads back."""

    @staticmethod
    def forward(ctx, x):
        y = x.contiguous().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad


class _AllReduceBackward(torch.autograd.Function):
    """Column-parallel input: identity forward, sum the input gradient."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


def tp_all_reduce_forward(x):
    return _AllReduceForward.apply(x) if is_tp_enabled() else x


def tp_all_reduce_backward(x):
    return _AllReduceBackward.apply(x) if is_tp_enabled() else x


def _base_module(module):
    org = getattr(module, "org_module", None)
    if org is not None:
        return org[0]
    return module


def _slice_groups(tensor, groups, rank, world, dim=0):
    """Take this rank's slice of every contiguous group along ``dim``.

    Used for fused qkv (3 groups) and ffn_up_gate (2 groups), which must be
    sharded per projection rather than by splitting the fused output in half.
    """
    total = tensor.shape[dim] // groups
    per = total // world
    if per * world != total:
        raise ValueError(f"TP cannot split {tensor.shape[dim]} into {groups}x{world}")
    parts = [
        tensor.narrow(dim, group * total + rank * per, per) for group in range(groups)
    ]
    return torch.cat(parts, dim=dim)


def _slice_row_parallel(tensor, rank, world, dim):
    total = tensor.shape[dim]
    per = total // world
    if per * world != total:
        raise ValueError(f"TP cannot split {total} columns into {world}")
    return tensor.narrow(dim, rank * per, per)


def _shard_base(base, mode, groups, rank, world):
    if getattr(base, "cr8_qdata", None) is not None:
        qdata = base.cr8_qdata
        if mode == "col":
            base.cr8_qdata = _slice_groups(qdata, groups, rank, world, 0).contiguous()
            scales = base.cr8_scales.view(torch.float32)
            scales = _slice_groups(scales, groups, rank, world, 0).contiguous()
            base.cr8_scales = scales.view(torch.uint8)
            base.out_features = base.cr8_qdata.shape[0]
        else:
            local_in = qdata.shape[1] // world
            rot = int(getattr(base, "cr8_rot_size", 1))
            if local_in % rot != 0:
                raise ValueError(
                    f"TP row shard of {qdata.shape[1]} -> {local_in} is not a "
                    f"multiple of the ConvRot block {rot}"
                )
            base.cr8_qdata = qdata.narrow(1, rank * local_in, local_in).contiguous()
            base.in_features = local_in
        base._cr8_qdata_t = None
        base._cr8_dequant_cache = None
        return
    weight = base.weight
    if mode == "col":
        new = _slice_groups(weight.data, groups, rank, world, 0).contiguous()
        base.out_features = new.shape[0]
    else:
        new = _slice_row_parallel(weight.data, rank, world, 1).contiguous()
        base.in_features = new.shape[1]
    base.weight = torch.nn.Parameter(new, requires_grad=weight.requires_grad)
    if base.bias is not None:
        raise NotImplementedError(
            "TP currently supports bias-free MOSS projections (got a biased linear)"
        )


def _shard_lora(lora, mode, groups, rank, world):
    if getattr(lora, "lora_mid", None) is not None:
        raise NotImplementedError("TP does not support LoCon/lora_mid adapters")
    up, down = lora.lora_up, lora.lora_down
    if mode == "col":
        new = _slice_groups(up.weight.data, groups, rank, world, 0).contiguous()
        up.weight = torch.nn.Parameter(new, requires_grad=True)
        up.weight._tp_sharded = True
        up.out_features = new.shape[0]
        if up.bias is not None:
            up.bias = torch.nn.Parameter(
                _slice_groups(up.bias.data, groups, rank, world, 0).contiguous(),
                requires_grad=True,
            )
        # lora_down is replicated but each rank only sees one output slice, so
        # its gradient is a partial sum and must be all-reduced before the step.
        down.weight._tp_partial_grad = True
    else:
        new = _slice_row_parallel(down.weight.data, rank, world, 1).contiguous()
        down.weight = torch.nn.Parameter(new, requires_grad=True)
        down.weight._tp_sharded = True
        down.in_features = new.shape[1]
        if down.bias is not None:
            down.bias = torch.nn.Parameter(
                _slice_row_parallel(down.bias.data, rank, world, 1).contiguous(),
                requires_grad=True,
            )
        up.weight._tp_partial_grad = True
    lora._tp_shard = (mode, groups)


def apply_tensor_parallel(model, network, rank=None, world=None):
    """Shard the MOSS DiT blocks in place. Call before the optimizer is built."""
    if not is_tp_enabled():
        return False
    rank = tp_rank() if rank is None else rank
    world = tp_world_size() if world is None else world
    layers = getattr(model, "layers", None)
    if layers is None:
        raise NotImplementedError(
            "TP is only implemented for the MOSS-Image transformer (model.layers)"
        )
    plan = {}
    for layer in layers:
        heads = int(layer.heads)
        if heads % world != 0:
            raise ValueError(f"TP needs heads divisible by {world}, got {heads}")
        for name, mode, groups in (
            ("qkv", "col", 3),
            ("o", "row", 1),
            ("ffn_up_gate", "col", 2),
            ("ffn_down", "row", 1),
        ):
            base = layer.proj[name]
            _shard_base(base, mode, groups, rank, world)
            plan[id(base)] = (mode, groups)
        layer.heads = heads // world

    # LoRA modules patch the base linear's forward instead of replacing it in
    # layer.proj, so find them through the network and match by the base module.
    loras = list(getattr(network, "unet_loras", ())) + list(
        getattr(network, "text_encoder_loras", ())
    )
    for lora in loras:
        org = getattr(lora, "org_module", None)
        if not org:
            continue
        meta = plan.get(id(org[0]))
        if meta is None:
            # adapter on a replicated module (e.g. the text refiner)
            continue
        _shard_lora(lora, meta[0], meta[1], rank, world)
    if network is not None:
        # Sampling must not merge sharded adapters into the sharded base.
        network.can_merge_in = False
    return True


def _iter_params(params):
    for item in params:
        if isinstance(item, dict):
            yield from item.get("params", ())
        else:
            yield item


def tp_sync_replicated_grads(network):
    """All-reduce grads of adapter params that are replicated but partial."""
    if not is_tp_enabled() or network is None:
        return
    for param in network.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if getattr(param, "_tp_partial_grad", False):
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


def tp_clip_grad_norm_(params, max_norm):
    """Global gradient-norm clipping across TP shards."""
    if not is_tp_enabled():
        return None
    if max_norm is None:
        return None
    rank = tp_rank()
    grads = []
    local_sq = torch.zeros(
        (), device=torch.device("npu", torch.npu.current_device()), dtype=torch.float32
    )
    for param in _iter_params(params):
        if param.grad is None:
            continue
        sharded = getattr(param, "_tp_sharded", False)
        if not sharded and rank != 0:
            # replicated parameters are identical on every rank; count once
            continue
        grads.append(param.grad)
        local_sq += param.grad.detach().float().pow(2).sum()
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    total_norm = local_sq.sqrt()
    coef = float(max_norm) / (float(total_norm) + 1e-6)
    if coef < 1.0:
        for grad in grads:
            grad.mul_(coef)
    return float(total_norm)


def _gather_rows(local, groups, world):
    per = local.shape[0] // groups
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.contiguous())
    return torch.cat(
        [
            torch.cat(
                [gathered[r][group * per : (group + 1) * per] for r in range(world)],
                dim=0,
            )
            for group in range(groups)
        ],
        dim=0,
    )


def _gather_cols(local, world):
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.contiguous())
    return torch.cat(gathered, dim=1)


def tp_unshard_network(network):
    """Gather sharded LoRA weights so rank 0 can save a normal file."""
    if not is_tp_enabled() or network is None:
        return
    world = tp_world_size()
    for module in network.modules():
        meta = getattr(module, "_tp_shard", None)
        if meta is None:
            continue
        mode, groups = meta
        if mode == "col":
            up = module.lora_up
            up.weight.data = _gather_rows(up.weight.data, groups, world)
            up.out_features = up.weight.shape[0]
            if up.bias is not None:
                up.bias.data = _gather_rows(up.bias.data, groups, world)
        else:
            down = module.lora_down
            down.weight.data = _gather_cols(down.weight.data, world)
            down.in_features = down.weight.shape[1]
            if down.bias is not None:
                down.bias.data = _gather_cols(down.bias.data, world)


def tp_reshard_network(network):
    """Restore this rank's shards after a save."""
    if not is_tp_enabled() or network is None:
        return
    rank = tp_rank()
    world = tp_world_size()
    for module in network.modules():
        meta = getattr(module, "_tp_shard", None)
        if meta is None:
            continue
        mode, groups = meta
        if mode == "col":
            up = module.lora_up
            up.weight.data = _slice_groups(
                up.weight.data, groups, rank, world, 0
            ).contiguous()
            up.out_features = up.weight.shape[0]
            if up.bias is not None:
                up.bias.data = _slice_groups(
                    up.bias.data, groups, rank, world, 0
                ).contiguous()
        else:
            down = module.lora_down
            down.weight.data = _slice_row_parallel(
                down.weight.data, rank, world, 1
            ).contiguous()
            down.in_features = down.weight.shape[1]
            if down.bias is not None:
                down.bias.data = _slice_row_parallel(
                    down.bias.data, rank, world, 1
                ).contiguous()
