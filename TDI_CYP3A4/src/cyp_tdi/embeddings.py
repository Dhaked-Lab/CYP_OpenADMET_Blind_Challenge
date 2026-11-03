"""Frozen CheMeleon / ChemProp-medium embeddings with checkpoint verification and provenance-checked cache."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io_utils import sha256_strings, sha256_array, read_json, write_json, save_npy, library_versions
from .resource_manager import validate_checkpoint
from .deps import import_framework


class EmbeddingError(RuntimeError):
    pass


def _safe_torch_load(torch, path, dev, safe=True):
    """JB used weights_only=False. We try weights_only=True first (the CheMeleon checkpoint contains only
    tensors/OrderedDict). The caller has already verified the file's pinned sha256."""
    if safe:
        try:
            return torch.load(path, map_location=dev, weights_only=True)
        except Exception as e:
            raise EmbeddingError(f"safe torch.load(weights_only=True) failed for {path}: {e}. "
                                 "Set embeddings.safe_checkpoint_load=false ONLY if the sha256-pinned checkpoint "
                                 "requires unpickling custom classes.")
    return torch.load(path, map_location=dev, weights_only=False)


def chemeleon_embed(smiles, ckpt: Path, expected_sha, device, batch_size=64, safe=True):
    """[JB] extract_chemeleon.py: frozen BondMessagePassing, mean-pooled atom states (2048-d)."""
    validate_checkpoint(ckpt, expected_sha, "CheMeleon checkpoint")
    torch = import_framework("torch", "CheMeleon embeddings"); import_framework("chemprop", "CheMeleon embeddings")
    from chemprop import data, featurizers, nn
    from torch.utils.data import DataLoader
    state = _safe_torch_load(torch, ckpt, device, safe)
    mp = nn.BondMessagePassing(**state["hyper_parameters"])
    mp.load_state_dict(state["state_dict"]); mp.to(device).eval()
    agg = nn.MeanAggregation()
    dps = [data.MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles]
    dset = data.MoleculeDataset(dps, featurizers.SimpleMoleculeMolGraphFeaturizer())
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=data.collate_batch, num_workers=0)
    out = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch[0]; bmg.to(device)
            out.append(agg(mp(bmg), bmg.batch).detach().float().cpu().numpy())
    return np.vstack(out)


def chemprop_embed(smiles, ckpt: Path, expected_sha, device, batch_size=64):
    """[JB] tdi_tabicl_cp.extract: MPNN.load_from_file; H = agg(message_passing(bmg, V_d)) (pre-FFN)."""
    validate_checkpoint(ckpt, expected_sha, "ChemProp-medium checkpoint")
    torch = import_framework("torch", "ChemProp embeddings"); import_framework("chemprop", "ChemProp embeddings")
    from chemprop import data, featurizers, models
    model = models.MPNN.load_from_file(ckpt); model.eval(); model.to(device)
    pts = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    dset = data.MoleculeDataset(pts, featurizers.SimpleMoleculeMolGraphFeaturizer())
    loader = data.build_dataloader(dset, batch_size=batch_size, num_workers=0, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d = batch[0], batch[1]; bmg.to(device)
            out.append(model.agg(model.message_passing(bmg, V_d), bmg.batch).detach().cpu().numpy())
    return np.vstack(out)


def cache_key(model_name, ckpt_sha, smiles, round_decimals, versions):
    from .config import fingerprint
    return fingerprint({"model": model_name, "ckpt": ckpt_sha, "smiles": sha256_strings(smiles),
                        "round": round_decimals, "torch": versions.get("torch"), "chemprop": versions.get("chemprop")})[:24]


def embed_with_cache(model_name, smiles, ckpt, expected_sha, device, cfg, cache_dir: Path, logger):
    """Reuse a cached embedding ONLY if checkpoint hash, SMILES list, library versions and rounding match."""
    emb_cfg = cfg["embeddings"]
    ckpt_sha = validate_checkpoint(ckpt, expected_sha, f"{model_name} checkpoint")
    ver = library_versions()
    key = cache_key(model_name, ckpt_sha, smiles, emb_cfg["round_decimals"], ver)
    cdir = Path(cache_dir) / "embeddings"; cdir.mkdir(parents=True, exist_ok=True)
    npy, meta = cdir / f"{model_name}-{key}.npy", cdir / f"{model_name}-{key}.json"
    if npy.exists() and meta.exists():
        try:
            m = read_json(meta); arr = np.load(npy)
            if m.get("array_sha256") == sha256_array(arr) and m.get("key") == key and arr.shape[0] == len(smiles):
                logger.info(f"{model_name}: using validated cached embedding {npy.name}")
                return arr, {**m, "from_cache": True}
        except Exception:
            pass
        logger.warning(f"{model_name}: cache entry failed validation - recomputing")
    logger.info(f"{model_name}: extracting embeddings for {len(smiles)} molecules on {device}")
    if model_name == "chemeleon":
        arr = chemeleon_embed(smiles, ckpt, expected_sha, device, emb_cfg["batch_size"], emb_cfg["safe_checkpoint_load"])
    else:
        arr = chemprop_embed(smiles, ckpt, expected_sha, device, emb_cfg["batch_size"])
    if emb_cfg["round_decimals"] is not None:
        arr = np.round(arr, int(emb_cfg["round_decimals"]))
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise EmbeddingError(f"{model_name}: non-finite values in extracted embeddings")
    m = {"key": key, "model": model_name, "checkpoint_sha256": ckpt_sha, "n": int(arr.shape[0]), "dim": int(arr.shape[1]),
         "smiles_sha256": sha256_strings(smiles), "array_sha256": sha256_array(arr), "versions": ver,
         "device": str(device), "round_decimals": emb_cfg["round_decimals"], "from_cache": False}
    save_npy(npy, arr); write_json(m, meta)
    return arr, m
