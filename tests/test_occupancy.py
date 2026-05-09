from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exitclear.config import load_config
from exitclear.occupancy import OccupancyEngine


def main() -> None:
    zone = load_config(ROOT / "config/zones.yaml").zones[0]

    shape = (120, 160)
    baseline = np.full(shape, 2200, dtype=np.uint16)

    clear_engine = OccupancyEngine(zone, shape)
    clear = clear_engine.evaluate(np.full(shape, 2200, dtype=np.uint16), baseline)
    assert clear.zone_pixel_count == 7200
    assert clear.valid_pixel_count == 7200
    assert clear.occupied_pixel_count == 0
    assert clear.occupied_bins == []
    assert clear.depth_valid_pct == 100.0
    assert clear.measured_free_width_mm == 900

    tiny_engine = OccupancyEngine(zone, shape)
    tiny = np.full(shape, 2200, dtype=np.uint16)
    tiny[50:55, 70:75] = 1400
    tiny_result = tiny_engine.evaluate(tiny, baseline)
    assert tiny_result.occupied_pixel_count == 0
    assert tiny_result.components == []

    blocked_engine = OccupancyEngine(zone, shape)
    blocked = np.full(shape, 2200, dtype=np.uint16)
    blocked[48:104, 62:100] = 1450
    blocked_result = blocked_engine.evaluate(blocked, baseline)
    assert blocked_result.occupied_pixel_count > 0
    assert len(blocked_result.components) == 1
    assert blocked_result.occupied_bins
    assert blocked_result.measured_free_width_mm < zone.required_clear_width_mm

    invalid_engine = OccupancyEngine(zone, shape)
    invalid = np.full(shape, 2200, dtype=np.uint16)
    invalid[30:50, 45:75] = 0
    invalid_result = invalid_engine.evaluate(invalid, baseline)
    assert invalid_result.depth_valid_pct < 100.0
    assert invalid_result.occupied_pixel_count == 0

    print("occupancy synthetic tests passed")


if __name__ == "__main__":
    main()
