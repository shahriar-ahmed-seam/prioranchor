"""Unit tests for parameter-efficient LoRA adapter layers and container."""

import tempfile

from prioranchor.core.models import LoRAConfig
from prioranchor.peft.lora import LoRAContainer, LoRALinear
from prioranchor.peft.merger import (
    calculate_compression_ratio,
    load_lora_checkpoint,
    save_lora_checkpoint,
)


def test_lora_initial_delta_zero():
    layer = LoRALinear(name="to_q", in_features=4, out_features=4, rank=2, alpha=4.0)
    x = [1.0, 2.0, 3.0, 4.0]

    # Initial forward before adaptation should equal base projection because B is 0
    y_init = layer.forward(x)
    assert len(y_init) == 4

    # Manually check delta weight is 0
    delta = layer.compute_delta_weight()
    for row in delta:
        for val in row:
            assert abs(val) < 1e-9


def test_lora_adaptation_and_merge_equivalence():
    layer = LoRALinear(name="to_k", in_features=3, out_features=3, rank=2, alpha=2.0)
    # Give non-zero values to B
    layer.lora_b = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    x = [0.5, 1.0, -0.5]
    y_unmerged = layer.forward(x)

    # Merge weights into base
    layer.merge_weights()
    assert layer.merged is True
    y_merged = layer.forward(x)

    # Check numerical equivalence
    for u, m in zip(y_unmerged, y_merged):
        assert abs(u - m) < 1e-6

    # Unmerge weights
    layer.unmerge_weights()
    assert layer.merged is False
    y_re_unmerged = layer.forward(x)

    for u, r in zip(y_unmerged, y_re_unmerged):
        assert abs(u - r) < 1e-6


def test_lora_container_and_checkpoint():
    cfg = LoRAConfig(rank=4, alpha=8.0)
    container = LoRAContainer(cfg)

    l1 = container.inject_layer("unet.cross_attn.to_q", in_features=8, out_features=8)
    l2 = container.inject_layer("unet.cross_attn.to_v", in_features=8, out_features=8)

    # Param count: 2 layers * (4*8 + 8*4) = 2 * 64 = 128
    assert container.get_trainable_param_count() == 128

    ratio = calculate_compression_ratio(
        total_base_params=1_000_000,
        trainable_lora_params=container.get_trainable_param_count(),
    )
    assert ratio < 0.001

    # Modify weights and save checkpoint
    l1.lora_b[0][0] = 0.42
    l2.lora_b[1][1] = 0.99

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/lora_weights.json"
        save_lora_checkpoint(container, ckpt_path)

        # Create fresh container and reload
        fresh_container = LoRAContainer(cfg)
        fresh_l1 = fresh_container.inject_layer(
            "unet.cross_attn.to_q", in_features=8, out_features=8
        )
        fresh_l2 = fresh_container.inject_layer(
            "unet.cross_attn.to_v", in_features=8, out_features=8
        )

        load_lora_checkpoint(fresh_container, ckpt_path)
        assert abs(fresh_l1.lora_b[0][0] - 0.42) < 1e-6
        assert abs(fresh_l2.lora_b[1][1] - 0.99) < 1e-6
