# Pretrained checkpoints (external inputs)

Weights are not stored in the repository. Place both files in this directory; each is verified against the SHA-256 pinned in
`config/resources.yaml` before every load, and a missing or modified file stops the run.

| File | Source | SHA-256 |
|---|---|---|
| `chemeleon_mp.pt` | CheMeleon Foundation Model, Zenodo record 15460715, DOI 10.5281/zenodo.15460715 (v2; MIT license) | `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651` |
| `chemprop_medium.pt` | `github.com/jeremycheminf/openadmet_scripts`, `CYP_Challenge/checkpoints/chemprop_medium.pt` (third-party; check its license) | `b8137a0d202669baa9747f21eac4ef78b000afe68a3f100d25c849c41ad8a3d5` |

The Zenodo v2 file is 34.9 MB with MD5 `6a80b54fdb7de37ef0374d302f01e8ce`; the pinned CheMeleon file has this MD5.

The TabICL checkpoint is different: it is fetched once into the local Hugging Face cache (see README, "TabICL checkpoint").
