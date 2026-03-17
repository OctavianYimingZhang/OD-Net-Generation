"""旧训练数据入口的薄包装。"""

from __future__ import annotations

from typing import Any

from od_zero_shot.data.dataset import sample_to_tensor_dict


def to_torch_sample(sample: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    """兼容旧调用名，内部直接委托给 canonical data.dataset。"""

    return sample_to_tensor_dict(sample, device=device)
