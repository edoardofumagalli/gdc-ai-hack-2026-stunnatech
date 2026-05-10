from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import PeopleCounterConfig


@dataclass(frozen=True)
class PeopleCounterResult:
    raw_count: float
    smooth_count: float
    density_map: np.ndarray


class PeopleCounter:
    def __init__(self, config: PeopleCounterConfig) -> None:
        self.config = config
        self._history: deque[float] = deque(maxlen=config.smoothing_frames)

    def update_from_nn(self, nn_message) -> PeopleCounterResult | None:
        tensor = _get_tensor(nn_message, self.config.tensor_name)
        if tensor is None:
            return None
        try:
            nn_data = np.array(tensor.data)
        except Exception:
            nn_data = np.array(tensor)

        density_map = reshape_dmcount_output(nn_data)
        if density_map is None:
            return None

        raw_count = float(density_map.sum()) / self.config.raw_scale
        self._history.append(raw_count)
        smooth_count = sum(self._history) / len(self._history)
        return PeopleCounterResult(
            raw_count=raw_count,
            smooth_count=smooth_count,
            density_map=density_map.astype(np.float32, copy=False),
        )


def reshape_dmcount_output(nn_data: np.ndarray) -> np.ndarray | None:
    if len(nn_data.shape) == 2:
        return nn_data.astype(np.float32, copy=False)
    if len(nn_data.shape) >= 3:
        return nn_data.squeeze().astype(np.float32, copy=False)

    flat = nn_data.flatten()
    fallback_h, fallback_w = 30, 53
    if len(flat) != fallback_h * fallback_w:
        return None
    return flat.reshape((fallback_h, fallback_w)).astype(np.float32, copy=False)


def _get_tensor(nn_message, tensor_name: str):
    try:
        return nn_message.getTensor(tensor_name)
    except Exception:
        try:
            layer_names = nn_message.getAllLayerNames()
        except Exception:
            return None
        if not layer_names:
            return None
        try:
            return nn_message.getTensor(layer_names[0])
        except Exception:
            return None
