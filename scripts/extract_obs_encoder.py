"""
Extract the obs_encoder weights from a DiffusionAttentionHybridImagePolicy
training checkpoint and save them as a flat state_dict that can be loaded via
`policy.obs_encoder.load_state_dict(sd, strict=True)`.

The checkpoint is the output of `BaseWorkspace.save_checkpoint`, structured as

    {
      'cfg': OmegaConf,
      'state_dicts': {
          'model':     <full policy state_dict>,
          'ema_model': <EMA-smoothed copy>,
          'optimizer': ...,
          ...
      },
      'pickles': ...,
    }

Models are wrapped in DataParallel during training, so every key in `model` /
`ema_model` is prefixed `module.<...>`. The obs_encoder lives under
`module.obs_encoder.<rest>`. We strip that prefix so the saved file's keys begin
at `<rest>` (e.g. `obs_nets.overhead_camera.backbone.nets.0.weight`).

Two files are written:
    <target>           -- weights from `model`        (regular)
    <target_ema>       -- weights from `ema_model`    (EMA-smoothed)

where `<target_ema>` inserts `_ema` immediately before the `.pth` extension,
e.g. `foo/bar.pth` -> `foo/bar_ema.pth`.

Usage:
    python scripts/extract_obs_encoder.py \\
        --checkpoint path/to/some/run/checkpoints/step=N-interval.ckpt \\
        --out data/pretrained_models/my_encoder.pth
"""
import argparse
import os
import pathlib
import sys

import dill
import torch


PREFIX = "module.obs_encoder."


def add_ema_suffix(path: pathlib.Path) -> pathlib.Path:
    """`foo/bar.pth` -> `foo/bar_ema.pth`. Preserves directory and extension."""
    if path.suffix != ".pth":
        raise ValueError(f"--out must end in .pth (got {path})")
    return path.with_name(f"{path.stem}_ema{path.suffix}")


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v.detach().clone()
    return out


def extract(checkpoint_path: pathlib.Path, target_path: pathlib.Path) -> None:
    target_ema = add_ema_suffix(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(
        open(checkpoint_path, "rb"),
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )

    state_dicts = payload["state_dicts"]
    if "model" not in state_dicts or "ema_model" not in state_dicts:
        raise KeyError(
            f"checkpoint is missing 'model' or 'ema_model' under state_dicts; "
            f"found {list(state_dicts.keys())}"
        )

    sd_model = strip_prefix(state_dicts["model"], PREFIX)
    sd_ema = strip_prefix(state_dicts["ema_model"], PREFIX)

    if not sd_model or not sd_ema:
        raise RuntimeError(
            f"No keys with prefix {PREFIX!r} found. The checkpoint may not be a "
            f"DataParallel-wrapped policy; inspect raw keys to confirm."
        )
    if len(sd_model) != len(sd_ema):
        raise RuntimeError(
            f"model and ema_model disagree on key count: "
            f"{len(sd_model)} vs {len(sd_ema)}"
        )

    cfg = payload.get("cfg", None)
    if cfg is not None:
        try:
            print(f"  source policy: {cfg['policy']['_target_']}")
            print(f"  source n_obs_steps={cfg['n_obs_steps']}, "
                  f"horizon={cfg['horizon']}, "
                  f"crop_shape={cfg['policy']['crop_shape']}, "
                  f"obs_encoder_group_norm={cfg['policy']['obs_encoder_group_norm']}")
        except (KeyError, TypeError):
            pass

    print(f"  extracted {len(sd_model)} obs_encoder keys")

    torch.save(sd_model, target_path)
    torch.save(sd_ema, target_ema)
    print(f"Saved (regular):    {target_path}")
    print(f"Saved (ema-smoothed): {target_ema}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "-c", required=True, type=pathlib.Path,
                        help="path to the workspace checkpoint .ckpt")
    parser.add_argument("--out", "-o", required=True, type=pathlib.Path,
                        help="target .pth for the regular encoder weights "
                             "(EMA copy is saved alongside with `_ema` suffix)")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    extract(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
