#!/usr/bin/env python3
"""
exp801: Manifold-Constrained Biomechanical Refinement — Feasibility

Test: Can 3D biomechanical constraints alone guide optimization in the
learned 15D error manifold, WITHOUT 2D reprojection or 3D GT?

Biomechanical losses (no GT needed):
1. Bone length consistency (skeleton proportions)
2. Joint angle limits (anatomical range)
3. Left-right symmetry
4. Temporal smoothness (within sequence)

Protocol:
- LOO: train PCA on 23 sequences, test on 1
- For each test sample: optimize z in 15D manifold to minimize biomech loss
- Measure resulting PA-MPJPE

This answers: "Do biomechanical priors provide enough signal in the
constrained manifold to move toward GT?"

Usage:
    uv run python scripts/exp801_manifold_biomech.py [run_dir]
"""

import os
import pickle
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# Biomechanical Constants
# ============================================================

# 14 eval joints (standard HMR protocol):
# 0:R_Ankle, 1:R_Knee, 2:R_Hip, 3:L_Hip, 4:L_Knee, 5:L_Ankle,
# 6:R_Wrist, 7:R_Elbow, 8:R_Shoulder, 9:L_Shoulder, 10:L_Elbow, 11:L_Wrist,
# 12:Neck, 13:Head
JOINT_NAMES_14 = [
    'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle',
    'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'Neck', 'Head'
]

# Bone connections (parent, child) in 14-joint format
BONES_14 = [
    (2, 1),   # R_Hip -> R_Knee
    (1, 0),   # R_Knee -> R_Ankle
    (3, 4),   # L_Hip -> L_Knee
    (4, 5),   # L_Knee -> L_Ankle
    (8, 7),   # R_Shoulder -> R_Elbow
    (7, 6),   # R_Elbow -> R_Wrist
    (9, 10),  # L_Shoulder -> L_Elbow
    (10, 11), # L_Elbow -> L_Wrist
    (12, 13), # Neck -> Head
    (12, 8),  # Neck -> R_Shoulder
    (12, 9),  # Neck -> L_Shoulder
]

# Symmetric joint pairs (left, right)
SYMMETRIC_PAIRS_14 = [
    (0, 5),   # R_Ankle <-> L_Ankle
    (1, 4),   # R_Knee <-> L_Knee
    (2, 3),   # R_Hip <-> L_Hip
    (6, 11),  # R_Wrist <-> L_Wrist
    (7, 10),  # R_Elbow <-> L_Elbow
    (8, 9),   # R_Shoulder <-> L_Shoulder
]

# Symmetric bone pairs (indices into BONES_14)
SYMMETRIC_BONE_PAIRS = [
    (0, 2),   # R_Hip->R_Knee <-> L_Hip->L_Knee
    (1, 3),   # R_Knee->R_Ankle <-> L_Knee->L_Ankle
    (4, 6),   # R_Shoulder->R_Elbow <-> L_Shoulder->L_Elbow
    (5, 7),   # R_Elbow->R_Wrist <-> L_Elbow->L_Wrist
]

# Approximate bone length ratios (from anthropometric data, normalized to total height)
# Used as soft prior, not hard constraint
BONE_LENGTH_RATIOS = {
    (2, 1): 0.245,   # thigh
    (1, 0): 0.246,   # shin
    (3, 4): 0.245,
    (4, 5): 0.246,
    (8, 7): 0.186,   # upper arm
    (7, 6): 0.146,   # forearm
    (9, 10): 0.186,
    (10, 11): 0.146,
    (12, 13): 0.108,  # neck
}


def compute_bone_lengths(joints_3d):
    """Compute bone lengths for 14-joint skeleton. joints_3d: (14, 3)"""
    lengths = []
    for (p, c) in BONES_14:
        bone = joints_3d[c] - joints_3d[p]
        lengths.append(torch.norm(bone))
    return torch.stack(lengths)


def bone_symmetry_loss(joints_3d):
    """Penalize asymmetric bone lengths (left vs right)."""
    bone_lens = compute_bone_lengths(joints_3d)
    loss = 0.0
    for (li, ri) in SYMMETRIC_BONE_PAIRS:
        loss = loss + (bone_lens[li] - bone_lens[ri]).pow(2)
    return loss / len(SYMMETRIC_BONE_PAIRS)


def bone_proportion_loss(joints_3d):
    """Soft prior on bone proportions — penalize extreme ratios."""
    bone_lens = compute_bone_lengths(joints_3d)
    total_height = bone_lens.sum()

    loss = 0.0
    for i, (p, c) in enumerate(BONES_14):
        if (p, c) in BONE_LENGTH_RATIOS:
            expected_ratio = BONE_LENGTH_RATIOS[(p, c)]
            actual_ratio = bone_lens[i] / (total_height + 1e-8)
            loss = loss + (actual_ratio - expected_ratio).pow(2)
    return loss


def bone_length_variance_loss(joints_3d_batch):
    """Within a batch (temporal window), bone lengths should be constant.
    joints_3d_batch: (T, 14, 3)
    """
    if joints_3d_batch.shape[0] < 2:
        return torch.tensor(0.0)

    all_bone_lens = []
    for t in range(joints_3d_batch.shape[0]):
        bl = compute_bone_lengths(joints_3d_batch[t])
        all_bone_lens.append(bl)
    all_bone_lens = torch.stack(all_bone_lens)  # (T, n_bones)

    # Variance across time for each bone should be 0
    return all_bone_lens.var(dim=0).mean()


def joint_angle_loss(joints_3d):
    """Penalize joint angles outside anatomical range.
    Simple version: check knee and elbow angles (should be 0-180 deg, mostly >30 deg).
    """
    loss = 0.0

    # Knee angles (should not hyperextend)
    for hip, knee, ankle in [(2, 1, 0), (3, 4, 5)]:
        v1 = joints_3d[hip] - joints_3d[knee]
        v2 = joints_3d[ankle] - joints_3d[knee]
        cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        # cos_angle near -1 = straight, near 1 = fully bent
        # Penalize hyperextension (angle < 10 deg from straight)
        # cos > 0.98 means angle < ~11 deg from fully bent (unlikely)
        loss = loss + F.relu(cos_angle - 0.95).pow(2)

    # Elbow angles
    for shoulder, elbow, wrist in [(8, 7, 6), (9, 10, 11)]:
        v1 = joints_3d[shoulder] - joints_3d[elbow]
        v2 = joints_3d[wrist] - joints_3d[elbow]
        cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        loss = loss + F.relu(cos_angle - 0.95).pow(2)

    return loss / 4.0


def compute_pa_mpjpe_torch(pred, gt):
    """PA-MPJPE for single sample. pred, gt: (14, 3)."""
    mu_p = pred.mean(0)
    mu_g = gt.mean(0)
    p_c = pred - mu_p
    g_c = gt - mu_g

    s_p = torch.sqrt((p_c ** 2).sum())
    s_g = torch.sqrt((g_c ** 2).sum())
    p_c = p_c / (s_p + 1e-8)
    g_c = g_c / (s_g + 1e-8)

    H = p_c.T @ g_c
    U, S, Vh = torch.linalg.svd(H)
    d = torch.det(Vh.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d]))
    R = Vh.T @ D @ U.T

    # 최적 스케일 trace(Sigma) 복원 — 이 항이 빠져 있어 PA 가 백본별 +0.5mm 안팎으로
    # 과대 측정됐다. 복원하면 백본 pkl 의 공식 pa_mpjpe 와 소수 4자리까지 일치한다.
    scale = S[0] + S[1] + d * S[2]
    p_aligned = s_g * scale * (p_c @ R.T) + mu_g
    return torch.sqrt(((p_aligned - gt) ** 2).sum(-1)).mean() * 1000


def load_data(pkl_path, mocap_path):
    """Load backbone predictions and sequence info."""
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
    return preds[:n], gts[:n], seq_names[:n]


def refine_sample(pred_joints, z_init, W, mean, sigma,
                  n_steps=50, lr=0.01, lambda_prior=0.1,
                  lambda_symmetry=1.0, lambda_proportion=0.5,
                  lambda_angle=0.5):
    """Refine a single sample by optimizing z in PCA manifold.

    Returns refined joints and optimization history.
    """
    pred_t = torch.tensor(pred_joints, dtype=torch.float32)  # (14, 3)
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
        refined = pred_t + delta.reshape(14, 3)

        # Biomechanical losses
        l_sym = bone_symmetry_loss(refined)
        l_prop = bone_proportion_loss(refined)
        l_angle = joint_angle_loss(refined)

        # Prior: Mahalanobis distance in PCA space
        l_prior = (z / (sigma_t + 1e-8)).pow(2).mean()

        total = (lambda_symmetry * l_sym +
                 lambda_proportion * l_prop +
                 lambda_angle * l_angle +
                 lambda_prior * l_prior)

        total.backward()
        optimizer.step()

        if total.item() < best_loss:
            best_loss = total.item()
            best_z = z.data.clone()

    # Final refined joints
    with torch.no_grad():
        delta = best_z @ W_t + mean_t
        refined = pred_t + delta.reshape(14, 3)

    return refined.numpy()


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/exp801")
    run_dir.mkdir(parents=True, exist_ok=True)

    mocap_path = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")

    backbones = {
        '4DHumans': os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl"),
    }

    d = 15  # PCA dimensions
    all_results = {}

    for bb_name, pkl_path in backbones.items():
        print(f"\n{'='*60}")
        print(f"exp801: Manifold Biomechanical Refinement — {bb_name}")
        print(f"{'='*60}")

        preds, gts, seq_names = load_data(pkl_path, mocap_path)
        unique_seqs = sorted(set(seq_names))
        N = len(preds)

        # Compute baseline PA-MPJPE
        baseline_pa = np.array([
            compute_pa_mpjpe_torch(
                torch.tensor(preds[i]), torch.tensor(gts[i])
            ).item()
            for i in range(N)
        ])
        print(f"  Baseline PA-MPJPE: {baseline_pa.mean():.2f}mm")

        # LOO evaluation
        seq_results = []
        all_refined_pa = np.zeros(N)

        for fold_idx, test_seq in enumerate(unique_seqs):
            train_mask = np.array([s != test_seq for s in seq_names])
            test_mask = ~train_mask
            n_test = test_mask.sum()

            # Fit PCA on train residuals
            train_residuals = (gts[train_mask] - preds[train_mask]).reshape(-1, 42)
            mean = train_residuals.mean(0)
            centered = train_residuals - mean
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            W = Vt[:d]  # (d, 42)
            sigma = S[:d] / np.sqrt(len(centered) - 1)

            # Refine each test sample
            test_indices = np.where(test_mask)[0]
            refined_pa_list = []

            for i, idx in enumerate(test_indices):
                refined_joints = refine_sample(
                    preds[idx], None, W, mean, sigma,
                    n_steps=50, lr=0.01,
                    lambda_prior=0.1,
                    lambda_symmetry=1.0,
                    lambda_proportion=0.5,
                    lambda_angle=0.5,
                )
                pa = compute_pa_mpjpe_torch(
                    torch.tensor(refined_joints), torch.tensor(gts[idx])
                ).item()
                refined_pa_list.append(pa)
                all_refined_pa[idx] = pa

                if i % 500 == 0 and i > 0:
                    print(f"    [{test_seq}] {i}/{n_test} "
                          f"baseline={baseline_pa[test_indices[:i+1]].mean():.1f} "
                          f"refined={np.mean(refined_pa_list):.1f}")

            seq_baseline = baseline_pa[test_indices].mean()
            seq_refined = np.mean(refined_pa_list)
            improvement = seq_baseline - seq_refined

            seq_results.append({
                'seq': test_seq,
                'n_samples': int(n_test),
                'baseline': float(seq_baseline),
                'refined': float(seq_refined),
                'improvement_mm': float(improvement),
                'improvement_pct': float(improvement / seq_baseline * 100) if seq_baseline > 0 else 0,
            })

            status = "✓" if improvement > 0 else "✗"
            print(f"  {status} [{fold_idx+1:2d}/24] {test_seq:35s}: "
                  f"{seq_baseline:.1f} → {seq_refined:.1f}mm ({improvement:+.1f}mm)")

        # Summary
        mean_baseline = baseline_pa.mean()
        mean_refined = all_refined_pa.mean()
        mean_improvement = mean_baseline - mean_refined
        n_worse = sum(1 for r in seq_results if r['improvement_mm'] < 0)

        print(f"\n{'='*60}")
        print(f"RESULTS: {bb_name}")
        print(f"{'='*60}")
        print(f"  Baseline:    {mean_baseline:.2f}mm")
        print(f"  Refined:     {mean_refined:.2f}mm")
        print(f"  Improvement: {mean_improvement:+.2f}mm ({mean_improvement/mean_baseline*100:+.1f}%)")
        print(f"  Worse seqs:  {n_worse}/24")
        print()
        print(f"  Reference:")
        print(f"    ScoreHMR:  54.3 → 51.1mm (-5.9%)")
        print(f"    Oracle:    54.3 → 41.1mm (-24.3%)")

        all_results[bb_name] = {
            'baseline': float(mean_baseline),
            'refined': float(mean_refined),
            'improvement_mm': float(mean_improvement),
            'improvement_pct': float(mean_improvement / mean_baseline * 100),
            'n_worse_seqs': n_worse,
            'per_seq': seq_results,
            'pca_dim': d,
            'method': 'manifold_biomechanical',
        }

    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    print(f"\nSaved to: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
