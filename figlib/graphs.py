"""Read network graphs from GeoPackage files, with no geospatial dependencies.

MALIBU3D ships road / railroad / transmission-line networks as GPKG files
holding `nodes`, `edges` and `metadata` layers. A GPKG is just SQLite, and its
geometry blobs are a small header followed by standard WKB, so `sqlite3` and
`struct` are enough — no GDAL, fiona or shapely required.

Graph coordinates are **absolute** (EPSG:2154 Lambert-93) and **2D**: these are
bird's-eye centrelines with no Z. Placing them in a scene therefore means
choosing a height; see `align_to_cloud`.
"""
import os.path as osp
import sqlite3
import struct

import numpy as np

WKB_LINESTRING = 2
WKB_MULTILINESTRING = 5

# envelope indicator (flags bits 1-3) -> number of doubles in the envelope
_ENVELOPE_DOUBLES = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}


def _parse_wkb_linestrings(blob):
    """Extract every LineString from a GPKG geometry blob as (M, 2) arrays."""
    if blob[:2] != b"GP":
        raise ValueError("Not a GeoPackage geometry blob")
    flags = blob[3]
    little_header = bool(flags & 0x01)
    envelope = _ENVELOPE_DOUBLES[(flags >> 1) & 0x07]
    offset = 8 + envelope * 8  # magic+version+flags+srs_id, then envelope
    if not little_header:
        pass  # header endianness only affects srs_id/envelope, which we skip

    def read_geometry(pos):
        byte_order = "<" if blob[pos] == 1 else ">"
        pos += 1
        (raw_type,) = struct.unpack_from(f"{byte_order}I", blob, pos)
        pos += 4
        # strip SRID flag (0x20000000) and dimension offsets (1000/2000/3000)
        base_type = (raw_type & 0xFF) % 1000
        has_z = ((raw_type & 0xFFFF) // 1000) in (1, 3) or bool(raw_type & 0x80000000)
        has_m = ((raw_type & 0xFFFF) // 1000) in (2, 3) or bool(raw_type & 0x40000000)
        if raw_type & 0x20000000:  # SRID present
            pos += 4
        ndim = 2 + int(has_z) + int(has_m)

        if base_type == WKB_LINESTRING:
            (n_points,) = struct.unpack_from(f"{byte_order}I", blob, pos)
            pos += 4
            coords = np.frombuffer(
                blob, dtype=np.dtype(f"{byte_order}f8"),
                count=n_points * ndim, offset=pos).reshape(n_points, ndim)
            pos += n_points * ndim * 8
            return [coords[:, :2].copy()], pos

        if base_type == WKB_MULTILINESTRING:
            (n_parts,) = struct.unpack_from(f"{byte_order}I", blob, pos)
            pos += 4
            parts = []
            for _ in range(n_parts):
                sub, pos = read_geometry(pos)
                parts.extend(sub)
            return parts, pos

        raise ValueError(f"Unsupported WKB geometry type {base_type}")

    lines, _ = read_geometry(offset)
    return lines


def read_gpkg_graph(path):
    """Read a network GPKG into plain numpy structures.

    Returns a dict with:
        edges: list of (M, 2) float64 arrays of absolute XY polylines
        nodes: (N, 2) float64 array of absolute XY node positions
        metadata: dict of the file's key/value metadata layer
        name: file stem, e.g. 'D075_UU-S1-3_ROADS_graph'
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"No such graph file: {path}")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        tables = {row[0] for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

        metadata = {}
        if "metadata" in tables:
            metadata = {k: v for k, v in cursor.execute(
                "SELECT key, value FROM metadata")}

        edges = []
        if "edges" in tables:
            for (blob,) in cursor.execute("SELECT geom FROM edges"):
                if blob is not None:
                    edges.extend(_parse_wkb_linestrings(blob))

        nodes = np.empty((0, 2))
        if "nodes" in tables:
            columns = {d[1] for d in cursor.execute("PRAGMA table_info(nodes)")}
            if {"x", "y"} <= columns:
                nodes = np.array(list(cursor.execute("SELECT x, y FROM nodes")),
                                 dtype=np.float64).reshape(-1, 2)
    finally:
        connection.close()

    return {
        "edges": edges,
        "nodes": nodes,
        "metadata": metadata,
        "name": osp.splitext(osp.basename(path))[0],
    }


def align_to_cloud(graph, coord_translation, offset, height):
    """Bring an absolute-coordinate graph into the rendered scene's frame.

    Two transforms compose:
      1. absolute -> source-relative, by subtracting the PLY's `coord_translation`
      2. source-relative -> scene, by adding the recentring `offset` the point
         cloud applied

    The graph has no Z, so every vertex is placed on one horizontal plane at
    `height` above the cloud's ground (z = 0 after centring).

    Returns the graph dict with `edges` and `nodes` replaced by (M, 3) arrays.
    """
    translation = np.asarray(coord_translation, dtype=np.float64)[:2]
    shift = np.zeros(2) if offset is None else np.asarray(offset, dtype=np.float64)[:2]

    def to_scene(xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2) - translation + shift
        return np.column_stack([xy, np.full(len(xy), float(height))])

    return {
        **graph,
        "edges": [to_scene(e) for e in graph["edges"]],
        "nodes": to_scene(graph["nodes"]) if len(graph["nodes"]) else np.empty((0, 3)),
    }


def find_meta_json(data_path):
    """Locate the `<roi>_meta.json` sidecar next to a MALIBU3D point cloud."""
    directory = osp.dirname(data_path)
    stem = osp.basename(data_path).split(".")[0]
    candidate = osp.join(directory, f"{stem}_meta.json")
    return candidate if osp.exists(candidate) else None
