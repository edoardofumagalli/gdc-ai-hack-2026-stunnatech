from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exitclear.config import load_config
from exitclear.models import ExitPosition
from exitclear.occupancy import OccupancyEngine


def main() -> None:
    zone = load_config(ROOT / "config/zones.yaml").zones[0]
    exit_position = ExitPosition(x=0.0, y=-300.0, z=2300.0)

    shape = (120, 160)
    baseline = np.full(shape, 2200, dtype=np.uint16)

    clear_engine = OccupancyEngine(zone, shape, exit_position, fx=500.0, fy=500.0)
    assert clear_engine.bounds.x_min == -250.0
    assert clear_engine.bounds.x_max == 250.0
    assert clear_engine.bounds.y_min == -550.0
    assert clear_engine.bounds.y_max == -50.0
    assert clear_engine.bounds.z_min == 1300.0
    assert clear_engine.bounds.z_max == 2300.0
    assert zone.monitored_depth_mm == 1000
    clear = clear_engine.evaluate(np.full(shape, 2200, dtype=np.uint16), baseline)
    assert clear.valid_pixel_count == 19200
    assert clear.occupied_pixel_count == 0
    assert clear.occupied_bins == []
    assert clear.depth_valid_pct == 100.0
    assert clear.measured_free_width_mm == 500

    tiny_engine = OccupancyEngine(zone, shape, exit_position, fx=500.0, fy=500.0)
    tiny = np.full(shape, 2200, dtype=np.uint16)
    tiny[50:55, 70:75] = 1400
    tiny_result = tiny_engine.evaluate(tiny, baseline)
    assert tiny_result.occupied_pixel_count == 0
    assert tiny_result.components == []

    blocked_engine = OccupancyEngine(zone, shape, exit_position, fx=500.0, fy=500.0)
    blocked = np.full(shape, 2200, dtype=np.uint16)
    blocked[20:100, 40:120] = 1450
    blocked_result = blocked_engine.evaluate(blocked, baseline)
    assert blocked_result.occupied_pixel_count > 0
    assert len(blocked_result.components) == 1
    assert blocked_result.components[0].area_mm2 > zone.occupancy.min_component_area_mm2
    assert blocked_result.occupied_bins
    assert blocked_result.measured_free_width_mm < zone.required_clear_width_mm

    invalid_engine = OccupancyEngine(zone, shape, exit_position, fx=500.0, fy=500.0)
    invalid = np.full(shape, 2200, dtype=np.uint16)
    invalid[30:50, 45:75] = 0
    invalid_result = invalid_engine.evaluate(invalid, baseline)
    assert invalid_result.depth_valid_pct < 100.0
    assert invalid_result.occupied_pixel_count == 0

    print("occupancy synthetic tests passed")


if __name__ == "__main__":
    main()
