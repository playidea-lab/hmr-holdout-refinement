#!/usr/bin/env python3
"""
exp802: Cross-Backbone Consensus Refinement — Feasibility

Key insight from error proxy analysis:
- Cross-backbone disagreement has Spearman r=0.258 with PA-MPJPE
- Q1 (low disagreement) = 47.3mm, Q4 (high disagreement) = 61.4mm
- This is 4x stronger signal than biomechanical metrics (r=0.06)

Method:
1. Use 2+ backbone predictions (4DHumans, TokenHMR)
2. For high-disagreement samples: move each prediction toward consensus
3. Correction constrained to backbone-specific PCA manifold (F2)
4. Joint-weighted (F3): proximal joints weighted higher
5. Only correct high-disagreement samples (F4 error gating)

Three consensus strategies tested:
A. Simple mean: refined = (pred_A + pred_B) / 2
B. Manifold-projected mean: project mean correction onto PCA manifold
C. Manifold-optimized consensus: optimize z to minimize disagreement + prior

Protocol: LOO (24-fold), no GT used at test time.

Usage:
    uv run python scripts/exp802_cross_backbone_consensus.py [run_dir]
"""

import os
import pickle
import json
import sys
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_pa_mpjpe_np(pred, gt):
    """PA-MPJPE in mm for single sample. pred, gt: (14, 3)."""
    mu_p = pred.mean(0)
    mu_g = gt.mean(0)
    p_c = pred - mu_p
    g_c = gt - mu_g
    s_p = np.sqrt((p_c ** 2).sum())
    s_g = np.sqrt((g_c ** 2).sum())
    p_c /= (s_p + 1e-8)
    g_c /= (s_g + 1e-8)
    H = p_c.T @ g_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    # 최적 스케일 trace(Sigma) 복원 — 누락 시 PA 가 체계적으로 과대 측정된다.
    scale = S[0] + S[1] + np.sign(d) * S[2]
    p_aligned = s_g * scale * (p_c @ R.T) + mu_g
    return np.sqrt(((p_aligned - gt) ** 2).sum(-1)).mean() * 1000


def compute_pa_mpjpe_batch(preds, gts):
    """PA-MPJPE for batch. preds, gts: (N, 14, 3)."""
    return np.array([compute_pa_mpjpe_np(preds[i], gts[i]) for i in range(len(preds))])


def load_backbone(pkl_path, mocap_path, n_limit=None):
    """Load backbone predictions and GT."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    preds = np.array([s['pred_joints_3d_eval'].reshape(14, 3) for s in data])
    gts = np.array([s['gt_joints_3d'].reshape(14, 3) for s in data])

    mocap = np.load(mocap_path, allow_pickle=True)
    imgnames = [s.decode() if isinstance(s, bytes) else s for s in mocap['imgname']]
    seq_names = []
    for name in imgnames:
        parts = Path(name).parts
        seq_names.append(parts[-2] if len(parts) >= 2 else "unknown")

    n = min(len(preds), len(seq_names))
    if n_limit:
        n = min(n, n_limit)
    return preds[:n], gts[:n], seq_names[:n]


def fit_pca(residuals, d=15):
    """Fit PCA on residuals, return (mean, W, sigma)."""
    mean = residuals.mean(0)
    centered = residuals - mean
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    W = Vt[:d]
    sigma = S[:d] / np.sqrt(len(centered) - 1)
    return mean, W, sigma


def project_to_manifold(correction, mean, W):
    """Project a correction vector onto the PCA manifold.
    correction: (42,), mean: (42,), W: (d, 42)
    Returns manifold-constrained correction (42,).
    """
    centered = correction - mean
    z = centered @ W.T  # (d,)
    recon = z @ W + mean  # (42,)
    return recon


def optimize_consensus(pred_A, pred_B, W, mean, sigma,
                       n_steps=30, lr=0.01,
                       lambda_prior=0.1, lambda_consensus=1.0):
    """Optimize z in manifold to minimize disagreement between backbones.

    The correction is applied to pred_A (the target backbone).
    Objective: move pred_A toward pred_B while staying on manifold.
    """
    pred_A_t = torch.tensor(pred_A.reshape(42), dtype=torch.float32)
    pred_B_t = torch.tensor(pred_B.reshape(42), dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32)
    sigma_t = torch.tensor(sigma, dtype=torch.float32)

    z = torch.zeros(W.shape[0], requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    best_loss = float('inf')
    best_z = z.data.clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Decode correction
        delta = z @ W_t + mean_t  # (42,)
        refined_A = pred_A_t + delta  # (42,)

        # Consensus loss: minimize distance to other backbone
        l_consensus = (refined_A - pred_B_t).pow(2).mean()

        # Manifold prior (Mahalanobis)
        l_prior = (z / (sigma_t + 1e-8)).pow(2).mean()

        total = lambda_consensus * l_consensus + lambda_prior * l_prior
        total.backward()
        optimizer.step()

        if total.item() < best_loss:
            best_loss = total.item()
            best_z = z.data.clone()

    with torch.no_grad():
        delta = best_z @ W_t + mean_t
        refined = (pred_A_t + delta).numpy()

    return refined.reshape(14, 3)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/exp802")
    run_dir.mkdir(parents=True, exist_ok=True)

    mocap_path = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")

    # Load both backbones
    print("Loading backbones...")
    # 원본은 상대경로라 저장소 루트에서만 돈다. 워크스페이스에서 실행하면 파일을 못 찾고
    # exit 1 로 죽는다(2026-09-21 실측). exp801 과 같은 절대경로 규약으로 맞춘다.
    preds_4dh, gts, seq_names = load_backbone(
        os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl"), mocap_path)
    preds_tkn, _, _ = load_backbone(
        os.path.join(os.environ.get("PRED_DIR", "data"), "tokenhmr_3dpw_cuda.pkl"), mocap_path)

    N = min(len(preds_4dh), len(preds_tkn))
    preds_4dh = preds_4dh[:N]
    preds_tkn = preds_tkn[:N]
    gts = gts[:N]
    seq_names = seq_names[:N]

    unique_seqs = sorted(set(seq_names))

    # Baseline PA-MPJPE
    pa_4dh = compute_pa_mpjpe_batch(preds_4dh, gts)
    pa_tkn = compute_pa_mpjpe_batch(preds_tkn, gts)
    print(f"Baseline 4DHumans: {pa_4dh.mean():.2f}mm")
    print(f"Baseline TokenHMR: {pa_tkn.mean():.2f}mm")

    # Disagreement measure
    disagreement = np.sqrt(((preds_4dh - preds_tkn) ** 2).sum(axis=(1, 2)))
    print(f"Mean disagreement: {disagreement.mean():.4f}")

    d = 15  # PCA dimensions

    # ================================================================
    # Strategy A: Simple Mean
    # ================================================================
    print(f"\n{'='*60}")
    print("Strategy A: Simple Mean (no manifold)")
    print(f"{'='*60}")

    mean_pred = (preds_4dh + preds_tkn) / 2
    pa_mean = compute_pa_mpjpe_batch(mean_pred, gts)
    print(f"  Mean ensemble: {pa_mean.mean():.2f}mm")
    print(f"  vs 4DHumans: {pa_4dh.mean() - pa_mean.mean():+.2f}mm")
    print(f"  vs TokenHMR: {pa_tkn.mean() - pa_mean.mean():+.2f}mm")

    # Strategy A with error gating
    for gate_pct in [100, 75, 50, 25]:
        if gate_pct == 100:
            refined = mean_pred.copy()
        else:
            threshold = np.percentile(disagreement, 100 - gate_pct)
            high_disagree = disagreement >= threshold
            refined = preds_4dh.copy()
            refined[high_disagree] = mean_pred[high_disagree]

        pa_gated = compute_pa_mpjpe_batch(refined, gts)
        n_refined = (gate_pct * N) // 100 if gate_pct < 100 else N
        print(f"  Gate {gate_pct:3d}% (refine {n_refined:5d}): "
              f"{pa_gated.mean():.2f}mm ({pa_4dh.mean() - pa_gated.mean():+.2f}mm)")

    # ================================================================
    # Strategy B: Manifold-Projected Mean (LOO)
    # ================================================================
    print(f"\n{'='*60}")
    print("Strategy B: Manifold-Projected Mean (LOO)")
    print(f"{'='*60}")

    all_pa_B = np.zeros(N)
    seq_results_B = []

    for test_seq in unique_seqs:
        train_mask = np.array([s != test_seq for s in seq_names])
        test_mask = ~train_mask
        n_test = test_mask.sum()

        # Fit PCA on 4DHumans train residuals
        train_res_4dh = (gts[train_mask] - preds_4dh[train_mask]).reshape(-1, 42)
        mean_pca, W, sigma = fit_pca(train_res_4dh, d=d)

        # For each test sample
        test_indices = np.where(test_mask)[0]
        for idx in test_indices:
            # Naive correction: move toward TokenHMR
            naive_correction = (preds_tkn[idx] - preds_4dh[idx]).reshape(42) * 0.5

            # Project onto manifold
            manifold_correction = project_to_manifold(naive_correction, mean_pca, W)
            refined = preds_4dh[idx] + manifold_correction.reshape(14, 3)
            all_pa_B[idx] = compute_pa_mpjpe_np(refined, gts[idx])

        seq_baseline = pa_4dh[test_mask].mean()
        seq_refined = all_pa_B[test_mask].mean()
        seq_results_B.append({
            'seq': test_seq, 'baseline': float(seq_baseline),
            'refined': float(seq_refined),
            'improvement_mm': float(seq_baseline - seq_refined),
        })

    print(f"  Refined: {all_pa_B.mean():.2f}mm ({pa_4dh.mean() - all_pa_B.mean():+.2f}mm)")
    n_worse = sum(1 for r in seq_results_B if r['improvement_mm'] < 0)
    print(f"  Worse seqs: {n_worse}/24")

    # With gating
    for gate_pct in [75, 50, 25]:
        threshold = np.percentile(disagreement, 100 - gate_pct)
        high_disagree = disagreement >= threshold
        gated = pa_4dh.copy()
        gated[high_disagree] = all_pa_B[high_disagree]
        print(f"  Gate {gate_pct:3d}%: {gated.mean():.2f}mm ({pa_4dh.mean() - gated.mean():+.2f}mm)")

    # ================================================================
    # Strategy C: Manifold-Optimized Consensus (LOO)
    # ================================================================
    print(f"\n{'='*60}")
    print("Strategy C: Manifold-Optimized Consensus (LOO)")
    print(f"{'='*60}")

    all_pa_C = np.zeros(N)
    seq_results_C = []

    for fold_idx, test_seq in enumerate(unique_seqs):
        train_mask = np.array([s != test_seq for s in seq_names])
        test_mask = ~train_mask
        n_test = test_mask.sum()

        # Fit PCA on 4DHumans train residuals
        train_res_4dh = (gts[train_mask] - preds_4dh[train_mask]).reshape(-1, 42)
        mean_pca, W, sigma = fit_pca(train_res_4dh, d=d)

        # Optimize each test sample
        test_indices = np.where(test_mask)[0]
        for i, idx in enumerate(test_indices):
            refined = optimize_consensus(
                preds_4dh[idx], preds_tkn[idx],
                W, mean_pca, sigma,
                n_steps=30, lr=0.01,
                lambda_prior=0.1, lambda_consensus=1.0,
            )
            all_pa_C[idx] = compute_pa_mpjpe_np(refined, gts[idx])

        seq_baseline = pa_4dh[test_mask].mean()
        seq_refined = all_pa_C[test_mask].mean()
        improvement = seq_baseline - seq_refined
        status = "✓" if improvement > 0 else "✗"

        seq_results_C.append({
            'seq': test_seq, 'n_samples': int(n_test),
            'baseline': float(seq_baseline), 'refined': float(seq_refined),
            'improvement_mm': float(improvement),
            'improvement_pct': float(improvement / seq_baseline * 100),
        })
        print(f"  {status} [{fold_idx+1:2d}/24] {test_seq:35s}: "
              f"{seq_baseline:.1f} → {seq_refined:.1f}mm ({improvement:+.1f}mm)")

    print(f"\n  Overall: {all_pa_C.mean():.2f}mm ({pa_4dh.mean() - all_pa_C.mean():+.2f}mm)")
    n_worse_C = sum(1 for r in seq_results_C if r['improvement_mm'] < 0)
    print(f"  Worse seqs: {n_worse_C}/24")

    # With gating
    print(f"\n  With error gating:")
    for gate_pct in [75, 50, 25]:
        threshold = np.percentile(disagreement, 100 - gate_pct)
        high_disagree = disagreement >= threshold
        gated = pa_4dh.copy()
        gated[high_disagree] = all_pa_C[high_disagree]
        n_refined = high_disagree.sum()
        print(f"    Gate {gate_pct:3d}% (refine {n_refined:5d}): "
              f"{gated.mean():.2f}mm ({pa_4dh.mean() - gated.mean():+.2f}mm)")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  4DHumans baseline:         {pa_4dh.mean():.2f}mm")
    print(f"  TokenHMR baseline:         {pa_tkn.mean():.2f}mm")
    print(f"  Strategy A (simple mean):  {pa_mean.mean():.2f}mm ({pa_4dh.mean()-pa_mean.mean():+.2f}mm)")
    print(f"  Strategy B (manifold proj):{all_pa_B.mean():.2f}mm ({pa_4dh.mean()-all_pa_B.mean():+.2f}mm)")
    print(f"  Strategy C (manifold opt): {all_pa_C.mean():.2f}mm ({pa_4dh.mean()-all_pa_C.mean():+.2f}mm)")
    print()
    print(f"  Reference:")
    print(f"    ScoreHMR:  54.3 → 51.1mm (-5.9%)")
    print(f"    Oracle:    54.3 → 41.1mm (-24.3%)")

    # Save
    results = {
        'baselines': {
            '4DHumans': float(pa_4dh.mean()),
            'TokenHMR': float(pa_tkn.mean()),
        },
        'strategy_A_simple_mean': float(pa_mean.mean()),
        'strategy_B_manifold_proj': float(all_pa_B.mean()),
        'strategy_C_manifold_opt': float(all_pa_C.mean()),
        'strategy_C_per_seq': seq_results_C,
        'strategy_C_n_worse': n_worse_C,
        'pca_dim': d,
        'method': 'cross_backbone_consensus',
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    print(f"\nSaved to: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
