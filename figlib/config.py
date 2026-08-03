"""Loading of YAML figure configs, with `extends` inheritance.

A figure config is a YAML file describing one figure end to end: which point
cloud, where the camera sits, how the scene is lit, and (optionally) how the
camera moves for a video. Anything it does not set is inherited from the file
named in its `extends` key, recursively, down to `configs/base.yaml`.
"""
import ast
import os.path as osp

import yaml


def _deep_merge(base, override):
    """Recursively merge `override` into `base`; `override` wins."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path, _seen=None):
    """Load a YAML config, resolving its `extends` chain.

    `extends` is a path relative to the config file that declares it.
    """
    path = osp.abspath(path)
    _seen = _seen or []
    if path in _seen:
        chain = " -> ".join(osp.basename(p) for p in _seen + [path])
        raise ValueError(f"Circular `extends` in figure configs: {chain}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_rel = cfg.pop("extends", None)
    if parent_rel is None:
        return cfg

    parent_path = osp.normpath(osp.join(osp.dirname(path), parent_rel))
    parent = load_config(parent_path, _seen + [path])
    return _deep_merge(parent, cfg)


def apply_overrides(cfg, overrides):
    """Apply `a.b=value` strings onto a loaded config, in place.

    Values are parsed as Python literals where possible, so
    `--set render.resolution=[800,600]` and `--set sun.energy=2.5` both work.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # plain string

        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"Unknown config section {'.'.join(keys[:-1])!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"Unknown config key {dotted!r}")
        node[keys[-1]] = value
    return cfg


def require(cfg, section, keys, context):
    """Fail loudly and early when a config lacks keys a code path needs."""
    missing = [k for k in keys if cfg.get(section, {}).get(k) is None]
    if missing:
        raise ValueError(
            f"Config is missing {section}.{{{', '.join(missing)}}}, "
            f"which {context} requires. Add them to the figure's YAML file.")
