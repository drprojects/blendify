# Research Log

## Goal
Produce Blender figures for MALIBU3D, a large-scale aerial 3D point cloud
benchmark. Priority: a convenient, re-runnable setup for figure generation that
Claude can easily interface with — not a polished code product.

## Experiments
<!-- Log each A-B comparison: what was tried, which won, and why -->

## Findings
- **The 6432 elevation NaNs are a subtile-seam artifact.** Every one lies within
  0.2 m of a subtile boundary; the subtile pitch is 102.4 m (not the 100 m the
  README claims), and 102.4 / 0.2 = 512, so subtiles are 512x512 rasters at 0.2 m
  and the artifact is exactly one pixel wide. Confirmed by Louis Geist: "des
  points qui tombent au bord de la subtile, et qui ne sont pas dans le geotiff".
  Same 6432 points are void in `elevation`, `forest` and `natural_habitat`.
- **Void convention** (per Louis): `#000000` is void in every layer; `#808080`
  (unknown) means a label outside `[0, n_label]` and in practice only appears for
  elevation. The v19/v20 mismatch is harmless — v20 only drops "Other Infra" in
  urban zones, no palette impact.
- **Loading dominates render time.** 4 M-point ROI: ~55 s to parse the gzipped
  PLY, ~4 s to render at 1050x700 / 32 samples on the 4090. Caching to `.npz`
  cuts the same render from 60 s to 9 s.
- **Sphere radius must track point spacing.** 0.15 m for a full ROI; scale by
  `sqrt(n_full / n_kept)` when subsampling. Earlier renders looked empty because
  the radius was sized for a much denser cloud.
- The MALIBU3D ROI `D075_UU-S1-3` is central Paris — the Eiffel Tower is clearly
  visible in the first render. Useful as a sanity check that geometry, centering
  and camera convention are all correct.
- The semantic palette appears correctly aligned despite the version mismatch
  below: the tower renders as class 0 = "Building".

## Dead Ends
<!-- What didn't work and why — saves re-trying the same thing -->

## Next Steps
- Capture a proper camera pose for `D075_UU-S1-3` in the Blender GUI
- Tune sun / world strength: the first RGB render is washed out
- Ask the colleague about the `semantic` definition mismatch (v19 vs v20)

---

## Session 2026-07-30

### Context loaded
First session on this repo with Claude. No `CLAUDE.md`, no `RESEARCH.md`, no
memory. Repo is a fork of `ptrvilya/blendify` with custom figure scripts from
two past papers (EZSP, LitePT) in `examples/00_custom.py`.

### Experiments run
- Explored the new MALIBU3D drop `data/malibu3d/send_29_07_v2/` (one ROI,
  `D075_UU-S1-3`, V1 format with raw label fields).
- Validated the PLY: 3 979 349 points, 512 x 512 m, fields `x/y/z`,
  `red/green/blue`, `semantic` (uint8, 11 of 16 classes present), `elevation`
  (float32, 6432 NaN), `natural_habitat` (uint8), `forest` (uint8).
- Refactored the five `if args.mode` chains into YAML configs; migrated all 41
  existing figures mechanically via AST parsing, then spot-checked two against
  the original source.
- First end-to-end render: 150 k points, 8 samples, 640x428, Cycles CPU. Took
  ~0.5 s per colorization.

### Findings
- **Elevation NaNs are not noise.** The 6432 NaN elevations coincide exactly
  with `forest == 2` and `natural_habitat == 43` — the same void/unknown points.
- **Label definition mismatch.** `meta.json` says `label_definitions.segment =
  "v20"`, `palettes.json` says `semantic.definition = "v19"`. Indices stay
  within the 16-class palette and the Eiffel Tower lands on class 0 =
  "Building", so rendering is fine, but class *names* are worth confirming.
- **A `.pt`-only interface was too narrow.** Replaced with `figlib/data.py`,
  which reads `.pt` / `.ply(.gz)` / `.npz` / `.las` / `.laz` into one
  `PointCloud` object (XYZ + named RGB colorizations). MALIBU3D PLY is now read
  directly, no conversion step.
- **Pre-existing bug found.** `--video` could only ever have worked for the four
  `paper_ezsp_*` modes: the trajectory chain defined `start_position` etc. only
  for those, and the code used them unconditionally. Now a clear config error.

### Dead ends
- Writing a MALIBU3D-specific `.pt` converter — abandoned mid-way on the (good)
  objection that `.pt` is a dirty legacy interchange format, not a standard to
  build on. Generalized into the format-agnostic reader instead.

### Next steps
1. Capture a real camera pose for `D075_UU-S1-3` in the Blender GUI; the current
   one is computed and only roughly frames the ROI.
2. Tune lighting — the first RGB render is washed out; try `sun.energy` ~2.0 and
   `world.strength` ~0.6.
3. Decide the point budget: 150 k looks sparse, 4 M may be too slow in Cycles.
   Sweep `data.subsample` against `data.voxel`.
4. Ask the colleague for the remaining ROIs and about the v19/v20 mismatch.
5. Consider a project-local `blendify-figure` skill now that the config system
   is stable.


## Session 2026-07-30 (continued)

### Experiments run
- Cross-tabulated the NaN elevations against semantic class, XY position and
  subtile grid. Ruled out a geographic MNT hole (NaNs are spread uniformly across
  every class at ~0.15%), then found the 0.2 m seam band.
- Timed the render pipeline with and without an `.npz` cache.
- Built and verified a lossless GUI round-trip for camera / sun / world.

### Findings
- `data.drop_void: [elevation]` removes exactly the 6432 seam points and is set
  at the MALIBU3D dataset level, since forest and natural_habitat share the same
  void set and would otherwise speckle the same 102.4 m grid.
- `drop_void` deliberately does **not** include `semantic`: its Void class is
  712 855 real points (18% of the cloud).
- `scene_to_config.py` round-trips config -> .blend -> config with zero numeric
  drift at 7 significant digits (float32 precision). Rounding to 10 digits
  surfaces Blender's float32 representation noise.

### Next steps
1. Capture a real camera pose for `D075_UU-S1-3` via the round-trip; the current
   one is computed and frames the ROI too obliquely.
2. Tune sun / world in the GUI at the same time.
3. Ask Louis for the remaining ROIs, and mention the 100 m vs 102.4 m subtile
   discrepancy in the README.
4. Decide whether elevation figures want a custom ramp range — the 2-98
   percentile clip lands at [-0.01, 65.27] m, which may crush building detail.

## Session 2026-07-30 (continued, 3)

### Built
- `data.add_xyz`: synthetic `xyz` colorization (position min-max scaled as RGB).
- `export.all_layers`: every colorization written into the `.blend` as a colour
  attribute on the ONE mesh, so layers can be switched in the GUI without
  re-exporting. Deliberately not one object per layer — geometry and the
  geometry-nodes sphere instancing are shared, which is the expensive part.
  Cost is file size: 122 MB -> 486 MB for six layers.
- `figlib/graphs.py`: GPKG reader with no geospatial deps (sqlite3 + struct on
  the WKB blobs). `figlib/blender_graph.py` draws graphs as bevelled curve tubes.
- `void:` section: recolour and fade unannotated points.
- `point_cloud.alpha`: uniform cloud opacity, for pushing the scene behind overlays.

### Findings
- Graph alignment verified: transformed ROADS graph spans
  [-273.0, -255.8] -> [238.0, 255.2] against the cloud's
  [-273.5, -256.3] -> [238.5, 255.7]. Roads visibly follow real streets.
- The ROADS graph is all 2-point segments (284 edges, 756 nodes) — no polylines.
- blender_plots already wires the colour attribute's Alpha into the Principled
  BSDF, so per-point opacity needs only an RGBA colour array. No material surgery.
- Rendering all 5 layers at 1400x940 / 48 samples takes 68 s total on the 4090.
  Adding transparency roughly doubles per-layer cost (~20 s each).

### Open issues to fix next
1. **Elevation ramp is crushed.** 2-98 percentile clips to [-0.01, 65.27] m but
   almost all points are under 20 m, so the city reads flat blue. Try
   `percentile_high` ~85 or an explicit range.
2. **XYZ blue channel is crushed** by the 347 m tower — Z min-max is dominated by
   one outlier. Per-axis percentile normalization would fix it.
3. Natural habitat multi-label GT and `land_use` are missing from the drop.
4. Only ROADS delivered; no RAILROADS / TRANSMISSION_LINES.

## Session 2026-08-03

### Built
- **Node-based grading.** Saturation / brightness / contrast / exposure / gamma /
  opacity moved out of the baked colour arrays into named shader nodes, so the
  same numbers drive a CLI render and a live GUI slider and can be read back out
  of a saved `.blend`. Neutral values: saturation 1, contrast 0, brightness 0,
  exposure 0, gamma 1.
- **`color.variants`** — a variant is the same source colours under a different
  grading preset, so it costs no extra colour array. `grayscale` (saturation 0
  from rgb) and `rgb_muted` (saturation 0.5, alpha 0.05) ship dataset-wide.
- **GUI panel** (`scripts/blender_layer_switcher.py`, embedded in every export):
  layer buttons, six grading sliders, greyscale toggle, per-graph colour / edge
  radius / node radius / height / glow / visibility, transparent-film toggle.
- **Graphs actually drawn.** `prepare_rois.py` was writing GPKG paths as YAML
  comments instead of `graphs.items`. 12 of 13 ROIs now carry their networks.
- **`scripts/selfcheck.py`** — 20 checks over the render and export paths.

### Findings
- `blender_plots` builds a **new material on every `scatter.color = ...`**, which
  orphans anything spliced into the material. Capture it once and keep it as the
  mesh's only material.
- The **render and export paths use different colour attributes**: `--image`
  writes into `marker_color`, an exported `.blend` carries `color_<layer>` per
  layer. Pointing the shader at a missing attribute renders black, silently.
- In bpy, `link.to_socket is node.inputs["Base Color"]` never matches — each
  attribute access builds a fresh wrapper. Compare by **name**. This produced a
  false "shader unwired" diagnosis that cost a lot of time.
- Blender's Bright/Contrast node is neutral at **0.0**, whereas the old Python
  power-law contrast was neutral at 1.0. Moving a value between two systems
  requires checking the neutral value in both.
- A desaturation test on a dark-forest tile under warm light cannot distinguish
  "working" from "no-op". Measured on a chromatic layer under neutral light, the
  saturation node cuts chromaticity by 95% (0.069 -> 0.0035).

### State
13 MALIBU3D ROIs parsed and cached (~2 GB), each with a `.blend` (8 layers,
grading chain, embedded panel, networks where delivered) and 6 PNG previews.
All 20 self-checks pass; all 13 `.blend` files verified individually.

### Open
1. Elevation ramp still crushed (2-98 percentile spans 0-65 m, most points <20 m).
2. XYZ blue channel flattened by tall outliers; per-axis percentile would fix it.
3. Palette editing in the GUI — ColorRamp LUT works for <=32 classes (landcover
   16, forest 3); natural habitat has 44 and needs family grouping.
4. Which classes are locked, and is colourblind-safety hard or best-effort?
5. Cameras for the three badly framed ROIs (two 512x1024 strips, two mountains).
