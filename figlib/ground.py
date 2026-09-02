"""A ground surface (DTM) estimated from the point cloud, for draping overlays.

Network graphs are 2D. Laying them on a constant-z plane makes them float above
valleys and cut into hills. Draping them onto the ground follows the terrain
instead.

Ground height is estimated per raster cell as a low percentile of z, which is
the standard way a DTM is derived from lidar: even under closed canopy some
returns reach the ground, so the bottom of the z distribution in a small cell is
the surface. This uses only the cached cloud — no re-parse of the source PLY.

If the raw `elevation` field (height above the delivered MNT) is available, it is
used instead: `ground = z - elevation` is exact rather than estimated. It is not
in the cache today, so the percentile route is the normal path.

Known limitation: a bridge or viaduct has ground *under* it, so a road crossing
one is draped down to the river bed rather than following the deck. Nothing in
the point cloud distinguishes the two without the deck being labelled.
"""
import numpy as np


class GroundGrid:
    """Sampleable ground surface on a regular XY raster."""

    def __init__(self, grid, origin, cell):
        self.grid = grid          # (H, W) float32, NaN where never filled
        self.origin = np.asarray(origin, dtype=np.float64)   # (x_min, y_min)
        self.cell = float(cell)

    @property
    def coverage(self):
        return float(np.isfinite(self.grid).mean())

    def sample(self, xy):
        """Bilinear ground height at each XY, clamped to the raster."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        height, width = self.grid.shape

        fx = (xy[:, 0] - self.origin[0]) / self.cell - 0.5
        fy = (xy[:, 1] - self.origin[1]) / self.cell - 0.5
        fx = np.clip(fx, 0, width - 1)
        fy = np.clip(fy, 0, height - 1)

        x0 = np.floor(fx).astype(int); x1 = np.minimum(x0 + 1, width - 1)
        y0 = np.floor(fy).astype(int); y1 = np.minimum(y0 + 1, height - 1)
        tx = (fx - x0)[:, None].ravel(); ty = (fy - y0).ravel()

        g = self.grid
        top = g[y0, x0] * (1 - tx) + g[y0, x1] * tx
        bottom = g[y1, x0] * (1 - tx) + g[y1, x1] * tx
        return top * (1 - ty) + bottom * ty


def _fill_holes(grid):
    """Fill empty cells from their filled neighbours, spreading outward."""
    filled = grid.copy()
    missing = ~np.isfinite(filled)
    if not missing.any():
        return filled
    for _ in range(200):
        if not missing.any():
            break
        padded = np.pad(filled, 1, constant_values=np.nan)
        stack = np.stack([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
            padded[:-2, :-2], padded[:-2, 2:],
            padded[2:, :-2], padded[2:, 2:],
        ])
        with np.errstate(invalid="ignore"):
            neighbour = np.nanmean(stack, axis=0)
        take = missing & np.isfinite(neighbour)
        if not take.any():
            break
        filled[take] = neighbour[take]
        missing &= ~take
    if missing.any():                      # nothing nearby at all
        filled[missing] = np.nanmedian(filled)
    return filled


def build(pos, elevation=None, cell=2.0, percentile=5.0):
    """Estimate the ground surface from a point cloud.

    Args:
        pos: (N, 3) points, in the same frame the graph will be drawn in.
        elevation: optional (N,) height above ground. When given, the ground is
            `z - elevation`, which is exact; otherwise a low percentile of z.
        cell: raster resolution in scene units.
        percentile: which quantile of z counts as ground (ignored if
            `elevation` is given).
    """
    pos = np.asarray(pos, dtype=np.float64)
    origin = pos[:, :2].min(axis=0)
    span = pos[:, :2].max(axis=0) - origin
    width = max(int(np.ceil(span[0] / cell)) + 1, 1)
    height = max(int(np.ceil(span[1] / cell)) + 1, 1)

    ix = np.clip(((pos[:, 0] - origin[0]) / cell).astype(int), 0, width - 1)
    iy = np.clip(((pos[:, 1] - origin[1]) / cell).astype(int), 0, height - 1)
    flat = iy * width + ix

    if elevation is not None:
        ground_z = pos[:, 2] - np.asarray(elevation, dtype=np.float64)
        finite = np.isfinite(ground_z)
        flat, ground_z = flat[finite], ground_z[finite]
        # mean per cell is fine once each sample is already a ground height
        total = np.bincount(flat, weights=ground_z, minlength=width * height)
        count = np.bincount(flat, minlength=width * height)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    else:
        # low percentile of z per cell: sort by (cell, z) and index into each run
        order = np.lexsort((pos[:, 2], flat))
        sorted_cell = flat[order]
        sorted_z = pos[order, 2]
        starts = np.searchsorted(sorted_cell, np.arange(width * height), "left")
        ends = np.searchsorted(sorted_cell, np.arange(width * height), "right")
        counts = ends - starts
        pick = starts + np.floor((counts - 1) * percentile / 100.0).astype(int)
        grid = np.full(width * height, np.nan)
        occupied = counts > 0
        grid[occupied] = sorted_z[pick[occupied]]

    return GroundGrid(_fill_holes(grid.reshape(height, width)), origin, cell)


def resample_polyline(points, max_segment):
    """Split a polyline so no segment is longer than `max_segment`.

    Straight edges between distant nodes would otherwise cut through terrain
    between their endpoints, however well the endpoints are draped.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    out = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        steps = max(int(np.ceil(distance / max_segment)), 1)
        for k in range(1, steps + 1):
            out.append(start + (end - start) * (k / steps))
    return np.asarray(out)
