# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal fork of [ptrvilya/blendify](https://github.com/ptrvilya/blendify) used to
generate Blender figures and videos for computer-vision papers on large 3D point clouds.

This is **research/figure-generation code, not a software product.** There are no tests,
no linter, and no build step. The deliverable is a rendered PNG or MP4. Optimize for
"I can re-run this and tweak it," not for abstraction or robustness.

## Environment

`bpy` is only installed in the `blendify` conda env — **not** in base python.
Always invoke scripts with that interpreter explicitly:

```bash
~/miniconda3/envs/blendify/bin/python examples/00_custom.py ...
# or: conda activate blendify
```

Pinned: python 3.10.9, `bpy==3.5.0`, plus `blender_plots`, `torch`, `trimesh`,
`videoio`, `matplotlib`, `mathutils`. See `installation.md` — the dependency set is
fragile, so prefer not to upgrade anything.

`bpy` prints a harmless `WARN (bgl): ... imported without an OpenGL backend` on every
import. Ignore it; it is not an error.

**Version split, and it matters.** Rendering uses the pinned `bpy` **3.5.0**; the
GUI on this machine is Blender **5.2.0 LTS**. A 5.2 `.blend` cannot be opened by
bpy 3.5 (it fails with the misleading "not a blend file"), and Blender cannot
save backwards. So `scripts/scene_to_config.py` shells out to the `blender`
binary to read `.blend` files, and only falls back to in-process bpy if none is
on PATH.

Camera, sun and world are stable concepts across versions, so tuning them in 5.2
and rendering in 3.5 is fine. **Material/BSDF tweaks are not** — the Principled
BSDF was reorganized in 4.x, and `00_custom.py` addresses its sockets by index
(`inputs[7]` specular, `inputs[9]` roughness), which are 3.x positions. Set
those via `point_cloud.specularity` / `roughness` in the config, not in the GUI.

## Running a figure

`examples/00_custom.py` is the workhorse script. Everything else in `examples/` is
upstream demo code.

**Every figure is a YAML config in `configs/figures/`.** Argparse only carries
run-time concerns; nothing about the figure's appearance lives in code.

```bash
# Still images: renders one PNG per colorization found in the data
~/miniconda3/envs/blendify/bin/python examples/00_custom.py \
    --config configs/figures/malibu3d_D075_UU-S1-3.yaml --image

# Fast preview: override any config entry with --set
~/miniconda3/envs/blendify/bin/python examples/00_custom.py \
    --config configs/figures/malibu3d_D075_UU-S1-3.yaml --image \
    --set data.subsample=150000 "data.colors=['rgb','semantic']" \
          render.n_samples=8 "render.resolution=[640,363]" render.cpu=True

# Video along the config's camera trajectory
~/miniconda3/envs/blendify/bin/python examples/00_custom.py \
    --config configs/figures/paper_ezsp_dales.yaml --video
```

Other flags: `--export` writes a `.blend` to open in the GUI, `--path` /
`--path_bbox` override the input files without editing the config.

### Config layout

`configs/base.yaml` holds every default and documents every key. Figure configs
declare `extends:` (a path relative to themselves) and override only what differs;
inheritance is a recursive deep merge, so `configs/figures/malibu3d_*.yaml` ->
`configs/malibu3d.yaml` -> `configs/base.yaml`. Loading lives in
`figlib/config.py`.

**Adding a figure means adding one YAML file, never a code branch.** Unknown keys
in `--set` and missing required keys both fail loudly rather than silently
defaulting.

## Input data format

`figlib/data.py` reads any supported format into a single `PointCloud` object:
`pos` as `(N, 3)` float32, plus `colors`, a dict of named `(N, 3)` uint8
colorizations. The figure script only ever sees that object, so supporting a new
format means writing one reader function.

| Suffix | Notes |
|---|---|
| `.pt` | Legacy dict: `pos` tensor + `*_colors` arrays. The `_colors` suffix is stripped, so `pred_colors` becomes the colorization `pred`. |
| `.ply`, `.ply.gz` | Uses `red/green/blue` as `rgb`, precomputed `<name>_red/green/blue` triplets, and raw scalar/label fields colorized via `data.palettes`. |
| `.npz` | `pos` + `*_colors`. Written by `scripts/inspect_pointcloud.py --cache`. |
| `.las`, `.laz` | Needs `laspy` (**not currently installed**). |

`--image` renders **one PNG per colorization**, named `<stem>_<name>.png`. Set
`data.colors` to restrict the set, and `data.default_color` to pick which one
drives the video and the initial scene.

`data.add_xyz: true` synthesizes an extra `xyz` colorization: position min-max
scaled per axis and shown as RGB, which reads as scene structure with no semantic
content.

To see what a new file yields before writing a config:

```bash
~/miniconda3/envs/blendify/bin/python scripts/inspect_pointcloud.py <file> \
    --palettes <palettes.json>
```

### Palettes

`figlib/palettes.py` turns raw fields into RGB from a palette JSON — either
categorical (`names` + `colors` + `unknown_color`) or continuous (`type:
continuous`, `color_stops_rgb`, percentile clipping). The format comes from
MALIBU3D's `palettes.json` but nothing in the module is dataset-specific.

### Model predictions (figure 4)

The `predictions/` sub-drop ships the same clouds as the main drop plus one
predicted field per benchmark task. Ground truth and prediction are two
colorizations of one cloud, so a GT/pred pair is a layer switch, not a second
render.

| task | ground truth field | predicted field |
|---|---|---|
| semantic | `semantic` | `pred_semantic` |
| forest | `forest` | `pred_forest` |
| elevation | `elevation` | `pred_elevation` |
| habitat type | `natural_habitat` **remapped** | `pred_nathab_habitat_type` |
| moisture regime | `natural_habitat` **remapped** | `pred_nathab_moisture_regime` |
| soil chemistry | `natural_habitat` **remapped** | `pred_nathab_soil_chemistry` |
| bioclimatic zone | `natural_habitat` **remapped** | `pred_nathab_bioclimatic_zone` |

**The four habitat tasks are asymmetric and it matters.** Ground truth arrives as
one 44-class `natural_habitat` field that each task `remap`s into its own
classes; predictions arrive *already* in each task's index space. So the
prediction palettes inherit the GT palette but drop the `remap` — applying it a
second time would scramble the labels. Verified against ground truth: 92.5%
identity agreement, and no relabelling of the predicted classes fits better.

Prediction palettes are declared in `configs/palettes/malibu3d_predictions.json`
using `"like"`, which deep-copies another layer's palette:

```json
"pred_semantic":     {"like": "semantic", "field": "pred_semantic"},
"pred_habitat_type": {"like": "habitat_type",
                      "field": "pred_nathab_habitat_type", "remap": null}
```

Inheriting rather than copying is deliberate: a prediction figure is only
readable if it uses *exactly* the GT colours, so an edit to the semantic palette
must reach both. An explicit `null` deletes an inherited key.
`data.palette_overrides` accepts a list, so the shared restyle and the
prediction layers compose without either copying the other.

Road networks come as both `<roi>_ROADS_graph.gpkg` and
`<roi>_ROADS_pred_graph.gpkg`; `prepare_rois.py` names them `roads_gt` and
`roads_pred` and gives them contrasting hues (blue / orange) rather than two
shades of one colour, which would read as one network drawn twice.

Prediction configs are generated with a suffix so they do not collide with the
originals:

```bash
python scripts/prepare_rois.py \
    --drop data/malibu3d/send_29_07_v2/blender_export/predictions \
    --suffix _pred \
    --palette-overrides "configs/palettes/malibu3d_extra.json,configs/palettes/malibu3d_predictions.json" \
    --camera-from malibu3d_D075_UU-S1-4
```

In the GUI, the Layer panel pairs each task on one row — ground truth left,
prediction right — with a *Flip ground truth / prediction* button.

### Network graphs

`figlib/graphs.py` reads GPKG network files with **no geospatial dependencies** —
a GPKG is SQLite, and its geometry blobs are a small header plus standard WKB, so
`sqlite3` + `struct` suffice. `figlib/blender_graph.py` draws them as bevelled
curves with a translucent, faintly emissive material.

Graphs are 2D bird's-eye centrelines with no Z, so they are placed on one
horizontal plane at `graphs.height` above the cloud's ground. Alignment composes
two transforms: absolute EPSG:2154 -> source-relative (subtract the PLY's
`coord_translation`, read automatically from the `<roi>_meta.json` sidecar) ->
scene (add the recentring `PointCloud.offset`).

`graphs.items` is a list, so one scene can carry several networks — the intended
use is three network types x (ground truth, prediction). Each item may override
`color`, `height`, `radius`, `alpha`, `emission`. See
`configs/figures/malibu3d_D075_UU-S1-3_graphs.yaml`.

### Colour grading

`figlib/grading.py` grades photographic layers: white balance, then exposure and
contrast in **linear light** (where they mean physical gain and a power law about
mid-grey 0.18), then luma-preserving saturation in the display-encoded space,
then gamma. Configured under `color:`.

`color.apply_to` lists which colorizations get graded — default `[rgb]`. Never
add a categorical layer (semantic, forest, natural_habitat): grading a palette
breaks the match between a figure and its legend.

### Void points

`figlib/palettes.py` records, per colorization, which points carry no real label
(an explicit `Void` / `N/A` class, or a non-finite value in a continuous field).

There are two ways to handle them, and they answer different questions.

**Mute them** (the default). Void points are geometry that was never annotated
and on which no metrics are computed, so they must read as context, not as a
class. The palette paints them `#000000`, which is the highest-contrast thing on
the page and grabs the eye. The `void:` section overrides that per render:
`void.color` recolours them neutral grey, `void.alpha` fades them. The scatter
material wires the colour attribute's alpha into the BSDF, so this is genuine
per-point opacity. MALIBU3D uses grey `0.74` at alpha `0.55`.

**Drop them.** `data.drop_void: [layer, ...]` removes them outright.

The drop is applied to **every** colorization at once, so all figures of a scene
keep an identical point set and stay visually comparable. Choose the layer
deliberately: for MALIBU3D, `drop_void: [elevation]` removes 6432 subtile-seam
points, whereas `drop_void: [semantic]` would remove a genuine 18% Void class.

### Bounding boxes

`bbox.path` points at a **PLY** whose vertices are groups of 8 corners per box,
with vertex colour encoding the class. See `blendify/utils/bounding_box.py`. This
format is specific to the Waymo detection files and is acknowledged as hacky.

## Architecture

**Upstream library (`blendify/`)** — do not modify unless necessary. A thin
object-oriented wrapper over `bpy`: a `scene` singleton (`blendify/scene.py`) owning
`scene.renderables`, `scene.lights`, and a camera; plus `colors/`, `materials/`,
`cameras/`. Useful here mainly for `scene.set_perspective_camera`, `scene.render`,
`scene.export`, and `utils/camera_trajectory.Trajectory`.

**Fork-specific additions** — these are the only files that differ from upstream:

- `examples/00_custom.py` — the actual figure script (see below).
- `figlib/` — config loading, format-agnostic point cloud reading, palettes.
- `configs/` — `base.yaml`, dataset defaults, and one YAML per figure.
- `scripts/inspect_pointcloud.py` — inspect/cache a cloud without rendering.
- `examples/00_point_clouds.py` — an earlier, mostly superseded version. Treat as legacy.
- `blendify/utils/bounding_box.py` — PLY bbox loading, 3D IoU + NMS, and Blender
  cylinder/sphere/face drawing for box wireframes.

Note that `00_custom.py` bypasses much of the blendify abstraction and uses `bpy`
directly (render engine settings, sun, world background) plus `blender_plots.Scatter`
for the point cloud, because `Scatter` handles millions of points far better than
`scene.renderables.add_pointcloud`.

## Choosing a camera pose / lighting: the GUI round-trip

Do **not** read values off the scripting console and paste them by hand — that is
what `scripts/scene_to_config.py` exists to eliminate.

```bash
# 1. export a light scene to open in the GUI (subsample keeps the .blend small)
python examples/00_custom.py --config configs/figures/X.yaml --export \
    --set data.subsample=50000

# 2. open the .blend in Blender, move the camera, tweak the sun and the world
#    background, then SAVE the file.
#    To place the camera: navigate the viewport, select the camera object,
#    press Ctrl + Alt + Numpad 0 to snap it to the view.

# 3. pull camera + sun + world straight back into the YAML
python scripts/scene_to_config.py --blend data/.../X.blend \
    --config configs/figures/X.yaml
```

It pulls `camera:`, `sun:`, `world:` and `data.voxel` (the sphere radius, which
lives in the geometry-nodes "Mesh to Points" node). The round-trip is lossless to
float32 precision (verified). `--dry-run` prints what it would write; `--only
camera` restricts what is pulled.

Comments *inside* a replaced `camera:` / `sun:` / `world:` block are lost;
everything else survives. `data.voxel` is written as a targeted single-key edit,
and only if it actually changed — an unmodified radius is left inheriting from
the dataset config rather than being pinned at figure level.

### Colour variants

`color.variants` builds extra graded versions of a layer, carried alongside it as
their own colorizations. This is how the muted backdrop used under graph
overlays stays available *in the same scene* as the normal photo — grading is
baked into the colour array, so a differently-graded look is a separate layer,
not a separate config:

```yaml
color:
  variants:
    - name: rgb_muted
      from: rgb
      saturation: 0.5
```

Variants grade from the **ungraded** source, so they fully specify their own look
rather than compounding the base grade.

### The studio scene

`configs/figures/malibu3d_D075_UU-S1-3_graphs.yaml` extends the plain figure
config and adds the graphs plus the muted variant. **Export from that one** when
you want a `.blend` to tune in the GUI: it carries every colour layer *and* the
graph objects, so you can toggle networks with the Outliner eye icon and switch
point cloud layer independently.

Caveat when pulling changes back: `scene_to_config.py --config` should usually
point at the **parent** (`malibu3d_D075_UU-S1-3.yaml`), not the studio config.
Camera, sun and world live in the parent; writing them into the studio config
would shadow the parent and leave the plain figures un-updated.

### Switching layers in the GUI

`export.all_layers: true` (the default) writes **every** colorization into the
exported `.blend` as a separate colour attribute — `color_rgb`,
`color_semantic`, ... — all on the **one** mesh, which the material reads by name
through an `Attribute` node.

The exported file also embeds `scripts/blender_layer_switcher.py` as a text
datablock called `figure_panel.py`. In the GUI: **Scripting** tab -> **Run
Script** -> press **N** in the 3D viewport -> the **Figure** tab gives one button
per layer, an opacity slider wired to the `cloud_alpha` node, and show/hide
toggles for the graph overlays. That is the route to give the user; editing the
Attribute node's Name by hand works but is not discoverable for someone who does
not use Blender.

The script is embedded but **not** auto-run — auto-run would trip Blender's
script security prompt.

An empty `layer_name` on a Color Attribute node does **not** fall back to the
mesh's active colour attribute; it renders black. Tested. So per-layer switching
must go through the Attribute node's name, which is what the panel sets.

This is deliberately not one object per layer. Extra colour attributes cost one
float4 per point per layer and nothing else; the positions and, crucially, the
geometry-nodes sphere instancing are shared. Separate objects would duplicate
both and make a 4 M-point viewport crawl. The cost is file size — six layers
takes the ROI `.blend` from 122 MB to 486 MB. Trim with
`--set "data.colors=['rgb','semantic']"` at export time, or
`export.all_layers: false`.

Quaternions are `[w, x, y, z]` throughout.

## Iterating fast

Two things dominate, and they are not what you would guess.

**Loading dominates, not rendering.** For the 4 M-point MALIBU3D ROI: ~55 s to
decompress and parse the gzipped PLY, ~4 s to actually render at 1050x700 / 32
samples on the 4090.

This is handled automatically: the parsed cloud is cached to `.npz` under
`data/.figcache/` (gitignored) on first read, and reused afterwards, taking a
cold 63 s run down to a warm **7 s**. The cache key covers the size and mtime of
both the source and the palette file, so editing either rebuilds it. Configs
point `data.path` at the *source*; you never manage the cache by hand. Set
`data.cache: false` to force a re-parse, or delete `data/.figcache/`.

`scripts/inspect_pointcloud.py --cache <out.npz>` still exists for making a
standalone portable cache.

**Sphere radius must track point density.** Points that look invisible are
usually too small, not too few. `data.voxel` is a radius in scene units; it
needs to be about the mean point spacing. For a cloud of `N` points spread over
`A` m², spacing is `sqrt(A / N)`. If you subsample, scale the radius by
`sqrt(n_full / n_kept)` or the cloud goes transparent.

For quick looks, drop `render.n_samples` to 16-32 and halve
`render.resolution` via `--set`; keep the full values in the config.

## Rendering conventions## Rendering conventions

Renders use a transparent film (`film_transparent = True`) and `Standard` view
transform. Video frames are alpha-blended onto **white** before being written, so
figures drop straight into a LaTeX document. Videos are post-compressed with `ffmpeg`
(`libx264`, crf 28) into a `*_compressed.mp4` alongside the original.
