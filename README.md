# The Hold-Out Protocol Decides What a Post-Hoc Refiner Has Learned

Code for the paper **"The Hold-Out Protocol Decides What a Post-Hoc Refiner Has Learned: Personalization and Annotation Budgets in Human Mesh Recovery"** (Changmin Lee; under review).

A small residual MLP (34K parameters) refines the 14 evaluation joints predicted by a frozen human mesh recovery backbone (TokenHMR, HMR 2.0, SPIN) on 3DPW. The paper measures how much of the refiner's gain survives when the subject, not just the sequence, is held out; how many annotated sequences are needed and how to choose them; what the same labels buy when spent on the backbone's SMPL head instead; and how the numbers change on a split the refiner never saw (3DPW train+val as the label pool, test for evaluation).

This repository contains the refiner, the protocols and the scripts behind every table and figure. It does **not** contain 3DPW, SMPL, model weights or the backbones' per-frame predictions; see [Data preparation](#data-preparation).

## Repository layout

```
scripts/
  exp803b_data_efficiency_stats.py   # shared module: RefineMLP, standard Procrustes PA-MPJPE, training loop
  main_table_unified.py, run_fixed_tables.py, epochs_oracle_compare.py
  exp775_share_seeds.py, drive.py, summarize.py         # hold-out rules, size controls, per-subject shares
  exp774_personalization_controls.py                    # earlier controls; different-subject nearest selection
  exp773_fixed_vs_perquery.py                           # one fixed set vs per-target nearest selection
  exp790_subject_efficiency.py                          # data efficiency for a new subject
  exp770_head_finetune.py, exp771_adapt_budget.py, run771_seeds.py   # SMPL head adaptation (HMR 2.0)
  exp772_refiner_budget_strict.py                       # refiner under the same strict subject rule
  spin_ref_fixed.py, spin_budget_fixed.py               # SPIN numbers after the row-order fix
  spin_check.py ... spin_check5.py                      # the checks that found and verified that fix
  exp760_subject_disjoint.py, exp762_ablation_rerun.py, exp780_gated.py, diag_basis_angles.py
  exp700_4backbone_loo.py, exp720_analysis.py, exp751_cross_backbone_loo.py
  exp801_manifold_biomech.py, exp802_cross_backbone_consensus.py, run_fixed_unsupervised.py
  predictions/   # regenerate the backbones' per-frame 3DPW test predictions
  clean_split/   # build 3DPW train/val/test in one convention, run HMR 2.0, external-split experiment
figures/
  figs_r2.py, data_r2.json           # Figs. "efficiency", "fixedquery", "headrefiner"
results/tables/                      # every table of the paper and supplement as CSV
```

## Environment

```bash
uv sync                # Python 3.10-3.12, numpy, scipy, torch, matplotlib, smplx
```

The refiner experiments run on a single GPU in minutes to a few hours; a CPU works but is slow. Regenerating predictions (`scripts/predictions/`), the head-adaptation experiments (`exp770`, `exp771`) and `clean_split/infer_hmr2.py` need the corresponding backbone repositories and their own environments.

Scripts take parameters either as command-line flags or as a JSON string in the `CQ_PARAMS` environment variable (an artefact of the experiment runner we used; empty means defaults). Paths come from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `PRED_DIR` | per-frame predictions `tokenhmr_3dpw_cuda.pkl`, `4dhumans_3dpw_cuda.pkl`, `spin_3dpw_14j.pkl` | `data` |
| `TOKENHMR_3DPW_NPZ` | 3DPW test annotation file in TokenHMR's format (row order, sequence names, SMPL betas) | `data/tokenhmr/3dpw_test.npz` |
| `SPIN_ROOT` | SPIN repository; `data/dataset_extras/3dpw_test.npz` defines the SPIN prediction row order | `third_party/SPIN` |
| `HMR2_ROOT` | 4D-Humans repository (HMR 2.0); `hmr2_evaluation_data/3dpw_test.npz` | `third_party/4D-Humans` |
| `TOKENHMR_ROOT` | TokenHMR repository | `third_party/TokenHMR` |
| `THREEDPW_SEQ_DIR` | 3DPW `sequenceFiles/` (train, validation, test) | `data/3DPW/sequenceFiles` |
| `SMPL_DIR` | SMPL body models (clean split) | `data/body_models/smpl` |
| `RESULTS_DIR` | where scripts write their JSON output | `outputs` |
| `REPO_ROOT` | root under which `data/*.pkl` are looked up by `exp760`, `exp762`, `exp780`, `exp790` | repository root |

## Data preparation

Nothing here may be redistributed; obtain each item from its official source and accept its licence.

| Item | Source |
|---|---|
| 3DPW (images, `sequenceFiles`) | https://virtualhumans.mpi-inf.mpg.de/3DPW/ |
| SMPL / SMPL-H body models | https://smpl.is.tue.mpg.de/, https://mano.is.tue.mpg.de/ |
| TokenHMR code, checkpoint and evaluation data (incl. `3dpw_test.npz`) | https://github.com/saidwivedi/TokenHMR |
| HMR 2.0 (4D-Humans) code, checkpoint and `hmr2_evaluation_data` | https://github.com/shubham-goel/4D-Humans |
| SPIN code, checkpoint and `dataset_extras/3dpw_test.npz` | https://github.com/nkolot/SPIN |

**Per-frame predictions.** We do not redistribute the backbones' outputs. Regenerate them with the official checkpoints and each method's own 3DPW evaluation path:

```bash
# in the TokenHMR environment
python scripts/predictions/extract_tokenhmr_cuda.py --output $PRED_DIR/tokenhmr_3dpw_cuda.pkl
# in the 4D-Humans environment
python scripts/predictions/extract_4dhumans_cuda.py --output $PRED_DIR/4dhumans_3dpw_cuda.pkl
# in the SPIN environment, then convert H36M-17 to the 14 evaluation joints
python scripts/predictions/generate_spin_predictions_official.py --output $PRED_DIR/spin_3dpw_official.pkl
python scripts/predictions/exp700_prepare_spin_data.py      # reads/writes data/spin_3dpw_*.pkl
```

Each pickle is a list of dicts with `pred_joints_3d_eval` (14x3, pelvis-centred), `gt_joints_3d` (14x3) and the backbone's own `pa_mpjpe`. `run_fixed_tables.py` refuses to run unless our Procrustes reproduces the stored `pa_mpjpe` (49.1241 / 54.3258 / 57.6480 mm) to 0.01 mm.

## Reproducibility note: SPIN row order (key-based alignment)

The SPIN prediction file follows the row order of **SPIN's** `3dpw_test.npz` (it starts at `downtown_enterShop_00`), whereas TokenHMR and HMR 2.0 follow **TokenHMR's** `3dpw_test.npz` (it starts at `downtown_runForBus_01`). Sequence and subject labels are taken from the TokenHMR file. Attaching them to the SPIN rows by position, as an earlier version of our pipeline did, matches the same (sequence, frame) in 0 of 35,515 rows. Prediction and ground truth inside the SPIN file are still paired correctly, so the frame-level SPIN baseline (57.65 mm) was never affected; every SPIN result that uses sequence or subject labels was.

The fix aligns rows by key, not by position (`spin_order()` in `scripts/exp775_share_seeds.py`): each SPIN row is mapped to the TokenHMR row with the same (sequence, frame, rounded crop centre), the mapping is asserted to be one-to-one (0 failures, 0 ambiguities), and the SPIN arrays are permuted into TokenHMR order before any label is used. All SPIN numbers in the paper come from scripts that call `load_backbone(..., spin_reorder=True)`. Evidence that the alignment is right (`spin_check2.py`–`spin_check5.py`): after reordering, SPIN and HMR 2.0 ground truth for the same frame agree to 10.7 mm PA (146–176 mm before), the frame-difficulty correlation with HMR 2.0 rises from 0.03 to 0.62, and a rotation-invariant nearest-frame search recovers the matching frame in 342/600 queries (0/600 before). If you add a backbone, check its row order the same way.

## Reproducing the paper

| Paper item | Script(s) | Notes |
|---|---|---|
| Table "backbones" | `run_fixed_tables.py` (PA, verified against stored values), `exp700_4backbone_loo.py` (MPJPE) | |
| Table "main" — TokenHMR, HMR 2.0 | `run_fixed_tables.py` -> `main_table_unified.py` | 24-sequence LOO, seeds 42/123/456; `epochs_oracle_compare.py` checks 10 vs 30 epochs |
| Table "main" — SPIN | `spin_ref_fixed.py` | row-order fix applied |
| Tables "subjectloo", "controls", "persubject", "shareper" | `exp775_share_seeds.py` via `drive.py`, then `summarize.py` | `mkdir -p outputs/exp775 && CQ_PARAMS='{"jobs":"TokenHMR:b,TokenHMR:c,TokenHMR:d,4DHumans:b,4DHumans:c,4DHumans:d,SPIN:b,SPIN:c","n_seeds":5,"n_draws":10}' python scripts/drive.py` (run from the repository root) |
| Table "efficiency", Fig. "efficiency" | `exp803b_data_efficiency_stats.py` (TokenHMR, HMR 2.0; via `run_fixed_tables.py`), `spin_budget_fixed.py` (SPIN) | 20 random draws, Wilcoxon over 24 folds |
| Fig. "fixedquery" | `exp773_fixed_vs_perquery.py` (ViT backbones), `spin_budget_fixed.py` (SPIN) | |
| Table "diffsubj" | `exp774_personalization_controls.py` (ViT backbones), `spin_budget_fixed.py` (SPIN) | random column taken from Table "efficiency" |
| Table "subjeff" | `exp790_subject_efficiency.py` | |
| Table "cleansplit" | `clean_split/build_seq_npz.py` -> `clean_split/fix_scale.py` -> `clean_split/infer_hmr2.py` -> `clean_split/clean_split.py`, `clean_split/per_subject.py` | HMR 2.0 only; reproduces the official test npz (centre/scale identical, GT within 0.32 mm, 54.33 mm baseline) |
| Table "comparison" | refiner rows from the scripts above; unsupervised rows from `run_fixed_unsupervised.py` | SMPLify and ScoreHMR rows are quoted from the ScoreHMR paper |
| Fig. "headrefiner", Table "headfold" | `exp771_adapt_budget.py --k 1,2,3,4 --seed 0`, `run771_seeds.py` (seeds 1, 2), `exp772_refiner_budget_strict.py --seed {0,1,2}`, `summarize.py` | head adaptation needs the 4D-Humans environment and a feature cache |
| Supp. Table "ftcontrol" | `exp770_head_finetune.py` | |
| Supp. Tables "quartile", "baselines", "crossbb" | `exp720_analysis.py` (on `exp700_4backbone_loo.py` output), `exp751_cross_backbone_loo.py` | ViT backbones only |
| Supp. Table "gating" | `exp780_gated.py` | |
| Supp. Table "basisangles" | `diag_basis_angles.py` | `CQ_PARAMS='{"backbones":"TokenHMR,4DHumans"}'` |
| Supp. Table "unsupervised" | `run_fixed_unsupervised.py` (runs `exp801`, `exp802` with the standard Procrustes) | |
| Supp. S1.1 (random five sequences, 3 draws) | `exp762_ablation_rerun.py` | |
| Figures | `uv run python figures/figs_r2.py` | reads `figures/data_r2.json` (the numbers of the tables above) |

`results/tables/` holds every table of the paper (`main_*.csv`) and supplement (`supp_*.csv`) as printed.

### Notes on what was changed for release

Hard-coded machine paths (data files, sibling-script imports, output folders) were replaced by the environment variables above. No numerical code was changed. All PA-MPJPE values use the standard least-squares Procrustes with the optimal scale term; an older variant without that term, used in early drafts, is not part of this repository.

## Licence

Apache License 2.0 (`LICENSE`). Datasets, body models, backbone code and checkpoints are subject to their own licences and are not included.

## Citation

```bibtex
@unpublished{lee2026holdout_refiner,
  title  = {The Hold-Out Protocol Decides What a Post-Hoc Refiner Has Learned: Personalization and Annotation Budgets in Human Mesh Recovery},
  author = {Lee, Changmin},
  note   = {Under review},
  year   = {2026}
}
```
