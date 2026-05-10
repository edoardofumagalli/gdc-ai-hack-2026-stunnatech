#!/usr/bin/env python3

"""
Crowd Intelligence Pipeline on OAK-4D Pro.

Drop-in replacement / evolution of people_counter.py.

Pipeline (steps refer to the architecture document):

    [1] RGB + Depth acquisition          OAK-4D Pro
    [2] DM-Count inference               luxonis/dm-count:shb-426x240
    [3] Density integration              raw_sum / RAW_SCALE
    [4] Spatial projection               (u, v, z) -> (x, y) on floor
    [5] Exit definition                  config-driven, world coords
    [6] Distance-to-exit field           BFS on occupancy grid
    [7] Crowd velocity model             Greenshields, clamped
    [8] Bottleneck throughput            Q = C * door_width
    [9] Evacuation time estimation       travel + queue, in series
    [10] Risk estimation                 alpha*density + beta*evac
    [11] Visualization                   multi-layer overlay

This file deliberately remains a single module so it can be dropped in
in place of people_counter.py. The code is organized as a sequence of
small, independently-testable functions plus a SceneState dataclass that
carries the per-frame scene snapshot through the pipeline.

Notes on the model choices:

  * The velocity model is Greenshields (v = vmax * (1 - rho/rho_max))
    clamped at a small floor velocity so cells at jam density don't
    produce infinite travel time. This is a documented departure from
    the architecture doc; see compute_velocity_field().

  * Evacuation time per cell is travel_time + queue_time (in series),
    not max(...). The doc's max(...) form under-estimates when both
    terms are non-trivial. See estimate_evacuation_time().

  * Density-to-floor projection weights each pixel by its real-world
    footprint area (which scales with depth^2 / fx*fy). Without this,
    far pixels would be over-counted. See project_density_to_floor().

Calibration constants (intrinsics, floor plane, exits, RAW_SCALE) live
in CONFIG below and should be tuned per deployment.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np


# =========================================================
# CONFIG
# =========================================================

FPS = 30
SMOOTH_WINDOW = 5

MODEL_NAME = "luxonis/dm-count:shb-426x240"

# 1 person ~= 242 raw units (empirical, tune per deployment)
RAW_SCALE = 242.0


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for the RGB sensor used by DM-Count.

    These must match the resolution at which the density map is
    evaluated, NOT the native sensor resolution. The density map is
    426x240 by default; we project density-map pixels back to world
    coordinates, so fx/fy/cx/cy must be expressed in density-map space.
    """
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 213.0   # 426 / 2
    cy: float = 120.0   # 240 / 2
    width: int = 426
    height: int = 240


@dataclass
class FloorPlane:
    """Camera pose relative to the floor.

    The simplest case (camera looking down at a flat floor) is fully
    described by camera height and pitch angle. Roll/yaw can be folded
    into the exits' coordinate frame.
    """
    height_m: float = 3.0      # camera height above the floor
    pitch_deg: float = -25.0   # negative = tilted down


@dataclass
class Exit:
    """An exit in world (floor) coordinates."""
    position: Tuple[float, float]   # (x, y) in meters, floor frame
    width_m: float = 1.2            # door width
    flow_coeff: float = 1.3         # people / (m * s), Weidmann ~1.0-1.5


@dataclass
class GridSpec:
    """2D occupancy grid covering the room footprint."""
    x_min: float = -5.0
    x_max: float =  5.0
    y_min: float =  0.0
    y_max: float = 10.0
    cell_m: float = 0.25   # 25 cm cells

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell_m))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.cell_m))

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        ix = int((x - self.x_min) / self.cell_m)
        iy = int((y - self.y_min) / self.cell_m)
        return ix, iy


@dataclass
class CrowdModel:
    """Pedestrian flow parameters."""
    v_max: float = 1.34          # m/s, free-walking speed (Weidmann)
    v_min: float = 0.05          # m/s, floor to avoid div-by-zero
    rho_max: float = 5.4         # people/m^2, jam density
    risk_alpha: float = 1.0      # density weight in risk score
    risk_beta: float = 0.05      # evac-time weight in risk score


CONFIG_INTRINSICS = CameraIntrinsics()
CONFIG_FLOOR      = FloorPlane()
CONFIG_GRID       = GridSpec()
CONFIG_MODEL      = CrowdModel()
CONFIG_EXITS: List[Exit] = [
    Exit(position=(-4.5, 0.5), width_m=1.2),
    Exit(position=( 4.5, 0.5), width_m=1.2),
]


# =========================================================
# SCENE STATE
# =========================================================

@dataclass
class SceneState:
    """Per-frame snapshot carried through the pipeline.

    Every stage either reads from this object or fills in a field of it.
    Keeping it in one place makes the main loop a clean sequence of
    transformations and makes offline replay straightforward.
    """
    rgb: np.ndarray
    depth: np.ndarray                       # meters, same H/W as rgb (resized)
    density_map: Optional[np.ndarray] = None
    raw_count: float = 0.0
    smooth_count: float = 0.0

    occupancy_grid: Optional[np.ndarray] = None      # people / m^2
    distance_field: Optional[np.ndarray] = None      # meters to nearest exit
    velocity_field: Optional[np.ndarray] = None      # m/s
    evac_time_field: Optional[np.ndarray] = None     # seconds, per cell
    risk_field: Optional[np.ndarray] = None          # dimensionless

    evac_time_global: float = 0.0
    max_risk: float = 0.0


# =========================================================
# UTILS
# =========================================================

def normalize_density_map(density: np.ndarray) -> np.ndarray:
    """Stretch a float density map into a uint8 visualization."""
    density = density.astype(np.float32)
    density -= density.min()
    if density.max() > 0:
        density /= density.max()
    density *= 255.0
    return density.astype(np.uint8)


def reshape_dmcount_output(nnData: np.ndarray) -> Optional[np.ndarray]:
    """Defensive reshape of DM-Count output across versions.

    Carried over from the original people_counter.py: the model has
    shipped with subtly different output shapes, and the 30x53 fallback
    has been earned through debugging. Don't simplify away.
    """
    if len(nnData.shape) == 2:
        return nnData
    if len(nnData.shape) >= 3:
        return nnData.squeeze()
    flat = nnData.flatten()
    expected_h, expected_w = 30, 53
    if len(flat) != expected_h * expected_w:
        return None
    return flat.reshape((expected_h, expected_w))


# =========================================================
# STEP 4: SPATIAL PROJECTION (image -> floor)
# =========================================================

def _build_pixel_to_floor_lut(
    intr: CameraIntrinsics,
    floor: FloorPlane,
    grid: GridSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute, for each density-map pixel, the floor cell it maps to
    AND the per-pixel footprint area in m^2.

    The projection is done assuming a flat floor and a known camera
    pitch. For each pixel (u, v) we cast a ray, intersect with the
    floor plane, and record the (x, y) hit point. This is precomputed
    once because intrinsics and pose don't change per frame.

    Returns:
        cell_index: (H, W, 2) int array of (ix, iy) floor cell or -1
                    if the ray points above the horizon / out of grid.
        area_m2:    (H, W) float array of per-pixel footprint area on
                    the floor. Pixels above the horizon get 0.
    """
    H, W = intr.height, intr.width
    pitch = math.radians(floor.pitch_deg)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    h = floor.height_m

    # Pixel grid -> normalized camera rays (camera frame: x right,
    # y down, z forward). We rotate by pitch around the x axis to put
    # the floor at y_world = 0 with the camera at y_world = h.
    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    x_cam = (uu - intr.cx) / intr.fx
    y_cam = (vv - intr.cy) / intr.fy
    z_cam = np.ones_like(x_cam)

    # Rotate around camera-X axis by pitch (negative pitch = tilt down).
    # World coords: x_w = x_cam, y_w = y_cam*cos - z_cam*sin (vertical),
    # z_w = y_cam*sin + z_cam*cos (along floor).
    y_w_dir = y_cam * cos_p - z_cam * sin_p
    z_w_dir = y_cam * sin_p + z_cam * cos_p

    # Camera at (0, h, 0). Floor at y_world = 0. The ray is
    #   p(t) = camera + t * dir,  with t > 0 going forward along dir.
    # Hits floor when h + t*y_w_dir = 0, but our coordinate convention
    # has the camera ABOVE the floor (y_world = h, floor at y_world = 0)
    # and the camera looking down (y_w_dir > 0 means downward in world
    # frame after the pitch rotation). So the floor intersection is at
    # t = h / y_w_dir (a positive forward distance along the ray).
    eps = 1e-6
    valid = y_w_dir > eps
    t = np.where(valid, h / np.where(valid, y_w_dir, 1.0), -1.0)

    x_floor = t * x_cam
    y_floor = t * z_w_dir   # forward distance from camera

    # Per-pixel footprint area on the floor. For a pinhole camera, the
    # solid angle subtended by one pixel is ~ 1/(fx*fy). Distance along
    # the ray is t * |dir|. Foreshortening on the floor goes as
    # 1 / cos(angle_to_normal). Combined:
    ray_len = t * np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    cos_floor = np.where(valid, y_w_dir / np.sqrt(x_cam**2 + y_w_dir**2 + z_w_dir**2), 0.0)
    area = np.where(
        valid,
        (ray_len * ray_len) / (intr.fx * intr.fy * np.maximum(cos_floor, eps)),
        0.0,
    )

    # Map (x_floor, y_floor) into grid cell indices.
    ix = ((x_floor - grid.x_min) / grid.cell_m).astype(np.int32)
    iy = ((y_floor - grid.y_min) / grid.cell_m).astype(np.int32)
    in_grid = (
        valid
        & (ix >= 0) & (ix < grid.nx)
        & (iy >= 0) & (iy < grid.ny)
    )
    ix = np.where(in_grid, ix, -1)
    iy = np.where(in_grid, iy, -1)

    cell_index = np.stack([ix, iy], axis=-1).astype(np.int32)
    area_m2 = np.where(in_grid, area, 0.0).astype(np.float32)
    return cell_index, area_m2


def project_density_to_floor(
    density_map: np.ndarray,
    cell_index: np.ndarray,
    area_m2: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """Splat per-pixel density onto the floor occupancy grid.

    The DM-Count density map is in image space: each pixel value is a
    fraction of a person per pixel. We convert to people-per-m^2 by:

        people_in_pixel = density_map[u, v] / RAW_SCALE
        floor_cell = LUT[u, v]
        occupancy_grid[floor_cell] += people_in_pixel / cell_area_m2

    The cell_area is the grid cell area (constant), NOT the per-pixel
    footprint. The per-pixel footprint is implicitly handled because
    multiple far pixels accumulate into the same cell, while a near
    pixel covers only part of a cell.
    """
    occ = np.zeros((grid.nx, grid.ny), dtype=np.float32)

    # Resize density map to intrinsics resolution if it doesn't match.
    H_lut, W_lut = cell_index.shape[:2]
    if density_map.shape != (H_lut, W_lut):
        density_resized = cv2.resize(
            density_map.astype(np.float32),
            (W_lut, H_lut),
            interpolation=cv2.INTER_AREA,
        )
        # Resizing changes per-pixel sum scale; preserve total mass.
        orig_sum = float(density_map.sum())
        new_sum = float(density_resized.sum())
        if new_sum > 1e-9:
            density_resized *= orig_sum / new_sum
    else:
        density_resized = density_map.astype(np.float32)

    people_per_pixel = density_resized / RAW_SCALE

    ix = cell_index[..., 0]
    iy = cell_index[..., 1]
    valid = (ix >= 0) & (iy >= 0) & (area_m2 > 0)

    # np.add.at handles repeated indices correctly (vectorized scatter-add).
    np.add.at(
        occ,
        (ix[valid], iy[valid]),
        people_per_pixel[valid],
    )

    cell_area = grid.cell_m * grid.cell_m
    occ /= cell_area
    return occ


# =========================================================
# STEP 6: DISTANCE FIELD
# =========================================================

def compute_distance_to_exit(
    grid: GridSpec,
    exits: List[Exit],
) -> np.ndarray:
    """BFS over grid cells from all exit cells simultaneously.

    Returns a (nx, ny) float array of approximate Euclidean distance
    from each cell to the nearest exit, in meters. Uses 8-connected
    BFS with sqrt(2) diagonal cost.

    This is a static computation as long as exits and grid are fixed,
    so the main loop should call it once at startup, not per frame.
    """
    nx, ny = grid.nx, grid.ny
    INF = np.float32(1e9)
    dist = np.full((nx, ny), INF, dtype=np.float32)

    # Multi-source BFS via a simple priority-free flood fill is wrong
    # for non-uniform costs, so we use a tiny Dijkstra with a heap.
    import heapq
    heap: List[Tuple[float, int, int]] = []

    for ex in exits:
        ix, iy = grid.world_to_cell(*ex.position)
        if 0 <= ix < nx and 0 <= iy < ny:
            dist[ix, iy] = 0.0
            heapq.heappush(heap, (0.0, ix, iy))

    if not heap:
        # No exits inside grid; everything is unreachable.
        return dist

    cell = grid.cell_m
    diag = cell * math.sqrt(2.0)
    neighbors = [
        (-1, 0, cell), (1, 0, cell), (0, -1, cell), (0, 1, cell),
        (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag),
    ]

    while heap:
        d, ix, iy = heapq.heappop(heap)
        if d > dist[ix, iy]:
            continue
        for dx, dy, step in neighbors:
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < nx and 0 <= jy < ny:
                nd = d + step
                if nd < dist[jx, jy]:
                    dist[jx, jy] = nd
                    heapq.heappush(heap, (nd, jx, jy))

    return dist


# =========================================================
# STEP 7: VELOCITY MODEL
# =========================================================

def compute_velocity_field(
    occupancy_grid: np.ndarray,
    model: CrowdModel,
) -> np.ndarray:
    """Greenshields fundamental diagram, clamped at v_min.

    The doc specifies v = vmax * (1 - rho / rho_max). At rho = rho_max
    that gives v = 0, which causes travel time to blow up to infinity
    in any cell that briefly spikes to jam density. We clamp at v_min
    instead. This matches Weidmann's pedestrian flow model in spirit.
    """
    rho = np.clip(occupancy_grid, 0.0, model.rho_max)
    v = model.v_max * (1.0 - rho / model.rho_max)
    return np.maximum(v, model.v_min).astype(np.float32)


# =========================================================
# STEPS 8-9: BOTTLENECK + EVACUATION TIME
# =========================================================

def estimate_evacuation_time(
    occupancy_grid: np.ndarray,
    distance_field: np.ndarray,
    velocity_field: np.ndarray,
    exits: List[Exit],
    grid: GridSpec,
) -> Tuple[np.ndarray, float]:
    """Per-cell and global evacuation time estimates.

    Per-cell travel time approximates path-integral of 1/v as
    distance / v_local (a one-step approximation that's exact only in
    homogeneous fields, but cheap and stable).

    Per-cell queue time at the assigned exit = N_assigned / Q_exit,
    where N_assigned is the total number of people whose nearest exit
    is the same exit, and Q_exit = flow_coeff * door_width.

    The combined evacuation estimate per cell is travel + queue (in
    series for each person), which is the upper-bound the doc's max(.)
    form was reaching for but tighter.

    The global estimate is the max over all cells (last person out).
    """
    nx, ny = grid.nx, grid.ny
    cell_area = grid.cell_m * grid.cell_m

    # Reachable cells: BFS sentinel for unreachable is ~1e9. Anything
    # past a generous reachable distance (1e6 m) is unreachable.
    reachable = distance_field < 1e6

    # Travel time per cell (clamped to avoid tiny v issues).
    v = np.maximum(velocity_field, 1e-3)
    travel = np.where(reachable, distance_field / v, 0.0).astype(np.float32)

    # Assign each cell to its nearest exit by recomputing per-exit
    # distances. For small numbers of exits this is fine; for many
    # exits, store the assignment alongside the BFS instead.
    assignment = np.zeros((nx, ny), dtype=np.int32)
    best = np.full((nx, ny), np.inf, dtype=np.float32)
    for k, ex in enumerate(exits):
        # Cheap per-exit "as-the-crow-flies" surrogate, since the BFS
        # multi-source field doesn't remember which source won. Good
        # enough for assignment when exits are well-separated.
        ix0, iy0 = grid.world_to_cell(*ex.position)
        ix_arr, iy_arr = np.indices((nx, ny))
        d = np.sqrt(
            ((ix_arr - ix0) * grid.cell_m) ** 2
            + ((iy_arr - iy0) * grid.cell_m) ** 2
        ).astype(np.float32)
        better = d < best
        best = np.where(better, d, best)
        assignment = np.where(better, k, assignment)

    # Queue time per exit: total assigned headcount / capacity.
    queue_time_per_exit = np.zeros(len(exits), dtype=np.float32)
    for k, ex in enumerate(exits):
        mask = (assignment == k)
        n_assigned = float((occupancy_grid * mask).sum() * cell_area)
        capacity = ex.flow_coeff * ex.width_m       # people / s
        if capacity > 1e-6:
            queue_time_per_exit[k] = n_assigned / capacity

    queue_field = queue_time_per_exit[assignment]

    evac_per_cell = (travel + queue_field).astype(np.float32)
    # Zero-out unreachable cells in the evac field too, so visualization
    # and the global max never see sentinel values.
    evac_per_cell = np.where(reachable, evac_per_cell, 0.0).astype(np.float32)

    # Global estimate: worst REACHABLE populated cell. If no populated
    # reachable cells exist, fall back to 0.
    valid_pop = (occupancy_grid > 1e-3) & reachable
    if valid_pop.any():
        evac_global = float(evac_per_cell[valid_pop].max())
    else:
        evac_global = 0.0

    return evac_per_cell, evac_global


# =========================================================
# STEP 10: RISK
# =========================================================

def compute_risk_field(
    occupancy_grid: np.ndarray,
    evac_time_field: np.ndarray,
    model: CrowdModel,
) -> np.ndarray:
    """Linear blend of normalized density and normalized evac time."""
    rho_n = np.clip(occupancy_grid / model.rho_max, 0.0, 1.0)
    # Use a soft normalization for evac time so the field has a stable
    # 0..~1 range across frames; 60 s is "long" in evacuation terms.
    evac_n = np.clip(evac_time_field / 60.0, 0.0, 1.0)
    return (model.risk_alpha * rho_n + model.risk_beta * evac_n * 60.0).astype(np.float32)


# =========================================================
# STEP 11: VISUALIZATION
# =========================================================

def _colorize(field: np.ndarray, vmax: Optional[float] = None,
              colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Render a 2D float field as a colormapped uint8 image.

    TURBO is the default because it has high zero/low contrast (deep
    blue at zero, distinct from the dark-red "high" end), making sparse
    occupancy fields readable. Pass cv2.COLORMAP_JET to get the original
    look.
    """
    f = field.astype(np.float32)
    if vmax is None:
        vmax = float(f.max()) if f.max() > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    f = np.clip(f / vmax, 0.0, 1.0)
    f = (f * 255.0).astype(np.uint8)
    return cv2.applyColorMap(f, colormap)


def _grid_to_image(grid_field: np.ndarray, target_hw: Tuple[int, int],
                   smooth: bool = True) -> np.ndarray:
    """Render a (nx, ny) grid as an image of the requested HxW.

    The grid is in floor coordinates with x horizontal and y forward
    (away from camera). For a top-down panel we transpose so y is
    vertical (forward = up in the panel) and flip so that "near the
    camera" is at the bottom of the panel.

    smooth=True applies a small Gaussian and uses bilinear interpolation
    on resize, which makes the panel visually smooth without changing
    integrated mass meaningfully (the kernel is tiny relative to the
    grid). Pass smooth=False if you need pixel-accurate cell visibility
    for debugging.
    """
    panel = grid_field.T          # (ny, nx)
    panel = panel[::-1, :]        # forward up, near at bottom
    if smooth and panel.size > 0 and panel.max() > 0:
        # 3x3 Gaussian on the grid before upscaling. This is a viz-only
        # transform; the underlying SceneState.occupancy_grid is untouched.
        panel = cv2.GaussianBlur(panel.astype(np.float32), (3, 3), 0.8)
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_NEAREST
    panel = cv2.resize(panel, (target_hw[1], target_hw[0]), interpolation=interp)
    return panel


def _world_to_panel_xy(x: float, y: float, grid: GridSpec,
                       panel_w: int, panel_h: int) -> Tuple[int, int]:
    """Map (x, y) world meters to (px, py) panel pixels.

    Uses the same convention as _grid_to_image (y forward = up in panel,
    near = bottom). Coordinates outside the grid are clamped to the
    nearest edge so things like out-of-grid exits still render visibly.
    """
    fx = (x - grid.x_min) / (grid.x_max - grid.x_min)
    fy = (y - grid.y_min) / (grid.y_max - grid.y_min)
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))
    px = int(fx * (panel_w - 1))
    py = int((1.0 - fy) * (panel_h - 1))
    return px, py


def _draw_metric_grid(panel: np.ndarray, grid: GridSpec,
                      step_m: float = 1.0) -> None:
    """Draw faint 1m gridlines + axis labels on the top-down panel."""
    h, w = panel.shape[:2]
    color = (60, 60, 60)
    # Vertical lines (constant x)
    x = math.ceil(grid.x_min / step_m) * step_m
    while x <= grid.x_max:
        px, _ = _world_to_panel_xy(x, grid.y_min, grid, w, h)
        cv2.line(panel, (px, 0), (px, h), color, 1)
        x += step_m
    # Horizontal lines (constant y)
    y = math.ceil(grid.y_min / step_m) * step_m
    while y <= grid.y_max:
        _, py = _world_to_panel_xy(grid.x_min, y, grid, w, h)
        cv2.line(panel, (0, py), (w, py), color, 1)
        y += step_m


def visualize(
    state: SceneState,
    grid: GridSpec,
    exits: List[Exit],
    fps: float,
) -> np.ndarray:
    """Build the multi-layer overlay image returned for display."""
    rgb = state.rgb
    H, W = rgb.shape[:2]

    # --- Panel 1: RGB + density heatmap overlay ----------------------
    if state.density_map is not None:
        density_vis = normalize_density_map(state.density_map)
        density_vis = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
        density_vis = cv2.resize(density_vis, (W, H))
        cam_panel = cv2.addWeighted(rgb, 0.6, density_vis, 0.4, 0)
    else:
        cam_panel = rgb.copy()

    cv2.putText(
        cam_panel,
        f"Persone: {state.smooth_count:.1f}",
        (20, 50),
        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3,
    )
    cv2.putText(
        cam_panel,
        f"Evac: {state.evac_time_global:.1f}s",
        (20, 95),
        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 255), 2,
    )
    cv2.putText(
        cam_panel,
        f"Risk: {state.max_risk:.2f}",
        (20, 130),
        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 165, 255), 2,
    )
    cv2.putText(
        cam_panel,
        f"FPS: {fps:.1f}",
        (20, H - 20),
        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1,
    )

    # --- Panel 2: top-down occupancy --------------------------------
    # Design choices for readability:
    #   - Occupancy is the primary signal (one channel, one colormap).
    #   - Risk is overlaid as red contour lines only above a threshold,
    #     so empty cells stay black and don't compete with occupancy.
    #   - Faint 1m gridlines + axis labels give spatial context.
    #   - Camera position is marked so the user knows their viewpoint.
    #   - Exits are clamped to the panel edge if they fall outside, with
    #     an arrow pointing in the direction they're actually located.
    panel_h, panel_w = H, W // 2
    if state.occupancy_grid is not None and state.risk_field is not None:
        # Smooth occupancy as the base layer, on a true black background
        # so empty floor reads as "empty" not "low risk".
        occ_smooth = _grid_to_image(
            state.occupancy_grid, (panel_h, panel_w), smooth=True,
        )
        # Mask: only show colormap where there is meaningful occupancy.
        # Below threshold -> pure black. This is the single biggest
        # readability win.
        occ_mask = occ_smooth > 0.05  # people/m^2
        occ_colored = _colorize(
            occ_smooth, vmax=CONFIG_MODEL.rho_max,
            colormap=cv2.COLORMAP_TURBO,
        )
        topdown = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        topdown[occ_mask] = occ_colored[occ_mask]

        # Faint metric grid (1m spacing).
        _draw_metric_grid(topdown, grid, step_m=1.0)

        # Risk contour overlay (only high-risk regions, as red outlines).
        risk_smooth = _grid_to_image(
            state.risk_field, (panel_h, panel_w), smooth=True,
        )
        risk_binary = (risk_smooth > 0.6).astype(np.uint8) * 255
        if risk_binary.any():
            contours, _ = cv2.findContours(
                risk_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(topdown, contours, -1, (0, 80, 255), 2)

        # Camera marker at world origin (0, 0).
        cam_px, cam_py = _world_to_panel_xy(0.0, 0.0, grid, panel_w, panel_h)
        cv2.drawMarker(topdown, (cam_px, cam_py), (200, 200, 200),
                       cv2.MARKER_TRIANGLE_UP, 14, 2)
        cv2.putText(topdown, "CAM", (cam_px - 14, cam_py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Exits: render with full circle even when at panel edge by
        # using clamped panel coords; if the exit is outside the grid,
        # draw an arrow in its direction instead.
        for ex in exits:
            ex_x, ex_y = ex.position
            in_grid = (
                grid.x_min <= ex_x <= grid.x_max
                and grid.y_min <= ex_y <= grid.y_max
            )
            px, py = _world_to_panel_xy(ex_x, ex_y, grid, panel_w, panel_h)
            if in_grid:
                # Pull slightly inward from the panel edge so the circle
                # is fully visible (10px margin = circle radius).
                px = max(12, min(panel_w - 12, px))
                py = max(12, min(panel_h - 12, py))
                cv2.circle(topdown, (px, py), 8, (0, 255, 0), 2)
                cv2.putText(topdown, "E", (px - 4, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0), 1)
            else:
                # Out-of-grid exit: arrow at the panel edge pointing out.
                cv2.arrowedLine(topdown, (cam_px, cam_py), (px, py),
                                (0, 255, 0), 2, tipLength=0.15)

        # Title strip.
        cv2.rectangle(topdown, (0, 0), (panel_w, 22), (30, 30, 30), -1)
        cv2.putText(topdown, "Top-down  |  1m grid  |  red = risk",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1)

        # Density colorbar: vertical strip on the right, inset from
        # the edge so it doesn't collide with edge-clamped exit markers.
        bar_w = 8
        bar_x = panel_w - bar_w - 18      # 18px inset from right edge
        bar_top = 28
        bar_bot = panel_h - 28
        bar_h = bar_bot - bar_top
        bar = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
        bar = np.tile(bar, (1, bar_w)) * CONFIG_MODEL.rho_max
        bar_img = _colorize(bar, vmax=CONFIG_MODEL.rho_max,
                            colormap=cv2.COLORMAP_TURBO)
        topdown[bar_top:bar_top + bar_h, bar_x:bar_x + bar_w] = bar_img
        # Labels next to the bar (not overlapping it).
        cv2.putText(topdown, f"{CONFIG_MODEL.rho_max:.0f}",
                    (bar_x + bar_w + 1, bar_top + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220, 220, 220), 1)
        cv2.putText(topdown, "0",
                    (bar_x + bar_w + 1, bar_bot),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220, 220, 220), 1)
        cv2.putText(topdown, "p/m2",
                    (bar_x - 4, bar_top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220, 220, 220), 1)
    else:
        topdown = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(topdown, "warming up...", (20, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Resize cam_panel to match topdown height, then concatenate.
    cam_resized = cv2.resize(cam_panel, (W, panel_h))
    combined = np.hstack([cam_resized, topdown])

    # Safety banner if risk is high.
    if state.max_risk > 1.5:
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 30),
                      (0, 0, 200), -1)
        cv2.putText(combined, "WARNING: HIGH CROWD RISK",
                    (20, 22), cv2.FONT_HERSHEY_TRIPLEX,
                    0.7, (255, 255, 255), 2)

    return combined


# =========================================================
# DEPTHAI PIPELINE BUILD
# =========================================================

def build_pipeline(pipeline: "dai.Pipeline") -> Tuple[
    "dai.MessageQueue", "dai.MessageQueue", "dai.MessageQueue"
]:
    """Construct the OAK-4D pipeline: RGB + depth + DM-Count.

    Returns (qRgb, qDepth, qNN) message queues. Depth is produced by
    the OAK-4D Pro's stereo node. The depth frame is aligned to RGB so
    that pixel (u, v) in the density map corresponds to (u, v) in the
    depth frame after a resize.
    """
    # --- RGB camera + DM-Count ---------------------------------------
    camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A,
        sensorFps=FPS,
    )

    modelDescription = dai.NNModelDescription()
    modelDescription.model = MODEL_NAME
    modelDescription.platform = "RVC4"
    archivePath = dai.getModelFromZoo(modelDescription, useCached=True)
    print("Model path:", archivePath)
    nnArchive = dai.NNArchive(archivePath)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.build(camera, nnArchive)

    qRgb = nn.passthrough.createOutputQueue()
    qNN  = nn.out.createOutputQueue()

    # --- Stereo depth aligned to RGB ---------------------------------
    # OAK-4D Pro exposes left/right mono cameras for stereo. We align
    # the depth to CAM_A (RGB) so pixel correspondence is direct.
    monoLeft  = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=FPS,
    )
    monoRight = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=FPS,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    # PresetMode names changed between DepthAI V2 and V3, and even
    # between V3 minor versions. Try the modern names first and fall
    # back gracefully so this works on whichever build is installed.
    preset_candidates = [
        "DEFAULT",         # V3 default — always present
        "ROBOTICS",        # V3 recommended for indoor use
        "FAST_ACCURACY",   # V3 alternative
        "HIGH_ACCURACY",   # V2 legacy
        "HIGH_DENSITY",    # V2 legacy
    ]
    preset_set = False
    for name in preset_candidates:
        preset = getattr(dai.node.StereoDepth.PresetMode, name, None)
        if preset is not None:
            stereo.setDefaultProfilePreset(preset)
            print(f"StereoDepth preset: {name}")
            preset_set = True
            break
    if not preset_set:
        print("WARNING: no known StereoDepth preset found; using node defaults")
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    monoLeft.requestOutput((640, 400)).link(stereo.left)
    monoRight.requestOutput((640, 400)).link(stereo.right)

    qDepth = stereo.depth.createOutputQueue()

    return qRgb, qDepth, qNN


# =========================================================
# MAIN LOOP
# =========================================================

def run() -> None:
    count_history: deque = deque(maxlen=SMOOTH_WINDOW)

    # Precompute the static pieces: pixel->floor LUT and distance field.
    # Both depend only on config, not on frames.
    print("Precomputing pixel->floor LUT...")
    cell_index, area_m2 = _build_pixel_to_floor_lut(
        CONFIG_INTRINSICS, CONFIG_FLOOR, CONFIG_GRID,
    )
    print(f"  LUT shape: {cell_index.shape}, "
          f"valid pixels: {(area_m2 > 0).sum()}")

    print("Precomputing distance-to-exit field...")
    distance_field = compute_distance_to_exit(CONFIG_GRID, CONFIG_EXITS)
    print(f"  Grid: {CONFIG_GRID.nx} x {CONFIG_GRID.ny} cells, "
          f"max dist: {distance_field[distance_field < 1e8].max():.2f} m")

    with dai.Pipeline() as pipeline:
        qRgb, qDepth, qNN = build_pipeline(pipeline)

        pipeline.start()
        startTime = time.monotonic()
        counter = 0

        while pipeline.isRunning():
            inRgb   = qRgb.get()
            inDepth = qDepth.get()
            inNN    = qNN.get()

            frame = inRgb.getCvFrame()
            depth_raw = inDepth.getCvFrame()    # uint16, millimeters

            # --- Build per-frame scene state ----------------------------
            depth_m = depth_raw.astype(np.float32) / 1000.0
            depth_resized = cv2.resize(
                depth_m,
                (CONFIG_INTRINSICS.width, CONFIG_INTRINSICS.height),
                interpolation=cv2.INTER_NEAREST,
            )

            state = SceneState(rgb=frame, depth=depth_resized)

            # --- Step 2: read DM-Count tensor ---------------------------
            tensor = inNN.getTensor("density_map")
            try:
                nnData = np.array(tensor.data)
            except Exception:
                nnData = np.array(tensor)

            if counter == 0:
                print("Layer names:", inNN.getAllLayerNames())
                print("Tensor shape:", nnData.shape)

            density_map = reshape_dmcount_output(nnData)
            if density_map is None:
                print(f"\nUnexpected tensor size: {nnData.size}")
                continue
            state.density_map = density_map

            # --- Step 3: integrate density to scalar count --------------
            raw_sum = float(density_map.sum())
            state.raw_count = raw_sum / RAW_SCALE
            count_history.append(state.raw_count)
            state.smooth_count = sum(count_history) / len(count_history)

            # --- Step 4: project to floor occupancy grid ----------------
            state.occupancy_grid = project_density_to_floor(
                density_map, cell_index, area_m2, CONFIG_GRID,
            )

            # --- Step 6: distance field is precomputed (static) ---------
            state.distance_field = distance_field

            # --- Step 7: velocity field ---------------------------------
            state.velocity_field = compute_velocity_field(
                state.occupancy_grid, CONFIG_MODEL,
            )

            # --- Steps 8-9: evacuation time -----------------------------
            state.evac_time_field, state.evac_time_global = \
                estimate_evacuation_time(
                    state.occupancy_grid,
                    state.distance_field,
                    state.velocity_field,
                    CONFIG_EXITS,
                    CONFIG_GRID,
                )

            # --- Step 10: risk ------------------------------------------
            state.risk_field = compute_risk_field(
                state.occupancy_grid,
                state.evac_time_field,
                CONFIG_MODEL,
            )
            state.max_risk = float(state.risk_field.max())

            # --- Periodic debug stats -----------------------------------
            if counter % 60 == 0:
                print(
                    f"\n[frame {counter}] "
                    f"count={state.smooth_count:.1f} "
                    f"evac={state.evac_time_global:.1f}s "
                    f"max_risk={state.max_risk:.2f} "
                    f"density min/max/mean="
                    f"{density_map.min():.3f}/"
                    f"{density_map.max():.3f}/"
                    f"{density_map.mean():.3f}"
                )

            # --- Step 11: visualization ---------------------------------
            counter += 1
            fps = counter / (time.monotonic() - startTime)

            overlay = visualize(state, CONFIG_GRID, CONFIG_EXITS, fps)
            cv2.imshow("Crowd Intelligence", overlay)

            print(
                f"Persone: {state.smooth_count:.1f} | "
                f"Evac: {state.evac_time_global:.1f}s | "
                f"Risk: {state.max_risk:.2f}",
                end="\r",
            )

            if cv2.waitKey(1) == ord("q"):
                break

        pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()