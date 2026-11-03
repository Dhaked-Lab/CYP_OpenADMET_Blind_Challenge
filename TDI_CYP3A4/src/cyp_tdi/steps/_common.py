from __future__ import annotations
import numpy as np
import pandas as pd
from ..io_utils import read_json
from ..discovery import discover_inputs
from ..config import fracs as cfg_fracs
from ..resource_manager import checkpoint_spec

CK_MAP = {"chemeleon": "chemeleon_mp", "chemprop": "chemprop_medium"}


def raw_input_files(ctx):
    inp = discover_inputs(ctx.data_dir, ctx.iso_cfg)
    return list(inp["training"]) + [inp["blind"]]


def train_input_files(ctx):
    return list(discover_inputs(ctx.data_dir, ctx.iso_cfg)["training"])


def blind_input_file(ctx):
    return [discover_inputs(ctx.data_dir, ctx.iso_cfg)["blind"]]


def checkpoint_files(ctx, *names):
    return [checkpoint_spec(ctx.res, CK_MAP[n], ctx.root)[0] for n in names]


def load_clean(ctx):
    df = pd.read_csv(ctx.artifact("data_clean", "clean_data.csv"))
    return df, df["TDI_Label"].values.astype(int)


def load_folds(ctx):
    return np.load(ctx.artifact("scaffold_cv", "folds.npy"))


def load_groups(ctx):
    g = pd.read_csv(ctx.artifact("scaffold_cv", "scaffold_groups.csv"), keep_default_na=False)["group"].astype(str).values
    return g


def load_X(ctx):
    return np.load(ctx.artifact("features", "X_train.npy")), read_json(ctx.artifact("features", "feature_names.json"))


def load_S(ctx):
    return np.load(ctx.artifact("similarity", "sim_train_train.npy"))


def fracs(ctx):
    return cfg_fracs(ctx.cfg)


def convention(ctx):
    return ctx.cfg["selection"]["count_convention"]


def ckpt(ctx, which):
    return checkpoint_spec(ctx.res, CK_MAP[which], ctx.root)
