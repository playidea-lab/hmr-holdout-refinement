#!/usr/bin/env python3
"""
exp700 Phase 0A: Convert SPIN data from H36M-17 to 14-joint format.

Uses SPIN's own H36M joint regressor for both pred and GT (self-consistent).
Different joint regressor than TokenHMR/4DHumans, but PA-MPJPE is invariant
to the choice of regressor as long as pred and GT use the same one.
"""

import pickle
from pathlib import Path

import numpy as np
import torch


# H36M-17 to 14-joint: drop spine(7), head(9), headtop(10)
H36M17_TO_14 = [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16]


def compute_similarity_transform_batch(S1, S2):
    """Procrustes alignment (batched)."""
    B = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = (X1**2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, s, Vh = torch.linalg.svd(K)
    V = Vh.permute(0, 2, 1)
    Z = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    R = torch.bmm(torch.bmm(V, Z), U.permute(0, 2, 1))
    trace = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    scale = (trace / var1).unsqueeze(-1).unsqueeze(-1)
    t = mu2 - scale * torch.bmm(R, mu1)
    S1_hat = scale * torch.bmm(R, S1) + t
    return S1_hat.permute(0, 2, 1)


def main():
    input_path = Path("data/spin_3dpw_official.pkl")
    output_path = Path("data/spin_3dpw_14j.pkl")

    print(f"Loading SPIN data from {input_path}...")
    with open(input_path, "rb") as f:
        spin = pickle.load(f)

    n_samples = len(spin["imgname"])
    print(f"Loaded {n_samples} samples")

    # Use SPIN's own H36M joints (self-consistent pred/GT)
    pred_17 = spin["pred_joints_h36m"]  # (N, 17, 3)
    gt_17 = spin["gt_joints_h36m"]  # (N, 17, 3)

    # Convert to 14-joint subset
    pred_14 = pred_17[:, H36M17_TO_14].astype(np.float32)
    gt_14 = gt_17[:, H36M17_TO_14].astype(np.float32)

    # Make pelvis-relative
    pred_14 = pred_14 - pred_14[:, 0:1]
    gt_14 = gt_14 - gt_14[:, 0:1]

    # Compute per-sample metrics
    print("Computing metrics...")
    batch_size = 512
    all_pa_mpjpe = []
    all_mpjpe = []

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        pred_t = torch.from_numpy(pred_14[i:end]).float()
        gt_t = torch.from_numpy(gt_14[i:end]).float()

        # MPJPE
        mpjpe = torch.sqrt(((pred_t - gt_t) ** 2).sum(dim=-1)).mean(dim=-1) * 1000

        # PA-MPJPE
        pred_aligned = compute_similarity_transform_batch(pred_t, gt_t)
        pa_mpjpe = (
            torch.sqrt(((pred_aligned - gt_t) ** 2).sum(dim=-1)).mean(dim=-1) * 1000
        )

        all_mpjpe.extend(mpjpe.numpy())
        all_pa_mpjpe.extend(pa_mpjpe.numpy())

    # Create output samples (same format as TokenHMR/4DHumans)
    samples = []
    for i in range(n_samples):
        samples.append({
            "pred_joints_3d_eval": pred_14[i],
            "gt_joints_3d": gt_14[i],
            "pa_mpjpe": np.float32(all_pa_mpjpe[i]),
            "mpjpe": np.float32(all_mpjpe[i]),
        })

    print(f"Saving {len(samples)} samples to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)

    print(f"\nSPIN 14-joint data statistics:")
    print(f"  N samples: {len(samples)}")
    print(f"  PA-MPJPE: {np.mean(all_pa_mpjpe):.2f} mm (expected ~57-60mm)")
    print(f"  MPJPE: {np.mean(all_mpjpe):.2f} mm")


if __name__ == "__main__":
    main()
