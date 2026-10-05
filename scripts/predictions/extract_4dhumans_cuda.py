#!/usr/bin/env python3
"""
Extract 4DHumans (HMR2.0) predictions for supervised refinement.

Uses official 4DHumans evaluation protocol:
- pred_keypoints_3d from model output
- Pelvis centering using pelvis_ind
- KEYPOINT_LIST for H36M 14 joints
"""

import os
import sys
from pathlib import Path

# Add 4D-Humans to path
sys.path.insert(0, os.environ.get("HMR2_ROOT", "third_party/4D-Humans"))

import torch
import numpy as np
import pickle
from tqdm import tqdm

from hmr2.configs import dataset_eval_config, CACHE_DIR_4DHUMANS
from hmr2.datasets import create_dataset
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to


def compute_similarity_transform(S1, S2):
    """Procrustes alignment (batched)."""
    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)

    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = (X1**2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    t = mu2 - scale * torch.matmul(R, mu1)
    S1_hat = scale * torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)


def compute_pa_mpjpe_batch(pred, gt):
    """Compute PA-MPJPE in mm (batched)."""
    pred_aligned = compute_similarity_transform(pred, gt)
    error = torch.sqrt(((pred_aligned - gt) ** 2).sum(dim=-1)).mean(dim=-1)
    return error * 1000


def extract_predictions(
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 32,
):
    """Extract 4DHumans predictions on 3DPW test set."""

    print("=" * 60)
    print("4DHumans (HMR2.0) Prediction Extraction")
    print("=" * 60)

    # Download models if needed
    print("\n[1/4] Loading model...")
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(checkpoint_path)

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load dataset config
    print("\n[2/4] Creating 3DPW-TEST dataset...")
    dataset_cfg = dataset_eval_config()['3DPW-TEST']
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"Dataset size: {len(dataset)}")

    # Get keypoint list and pelvis index
    keypoint_list = dataset_cfg.KEYPOINT_LIST
    pelvis_ind = model_cfg.EXTRA.PELVIS_IND
    print(f"Keypoint list (H36M 14): {keypoint_list}")
    print(f"Pelvis index: {pelvis_ind}")

    print(f"\n[3/4] Extracting predictions...")
    all_samples = []
    pa_mpjpe_list = []
    mpjpe_list = []

    for batch in tqdm(dataloader):
        batch = recursive_to(batch, device)
        batch_size_curr = batch['img'].shape[0]

        with torch.no_grad():
            output = model(batch)

            # Get predictions and GT
            pred_keypoints_3d = output['pred_keypoints_3d']  # (B, 44, 3)
            gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1]  # (B, 44, 3) remove confidence

            # Pelvis centering
            pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, [pelvis_ind]]
            gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, [pelvis_ind]]

            # Extract evaluation joints (H36M 14)
            pred_eval = pred_keypoints_3d[:, keypoint_list]  # (B, 14, 3)
            gt_eval = gt_keypoints_3d[:, keypoint_list]  # (B, 14, 3)

            # Compute metrics
            batch_pa_mpjpe = compute_pa_mpjpe_batch(pred_eval, gt_eval)
            batch_mpjpe = torch.sqrt(((pred_eval - gt_eval) ** 2).sum(dim=-1)).mean(dim=-1) * 1000

        # Store samples
        pred_eval_np = pred_eval.cpu().numpy()
        gt_eval_np = gt_eval.cpu().numpy()
        batch_pa_mpjpe_np = batch_pa_mpjpe.cpu().numpy()
        batch_mpjpe_np = batch_mpjpe.cpu().numpy()

        for j in range(batch_size_curr):
            pa_mpjpe = batch_pa_mpjpe_np[j]
            mpjpe = batch_mpjpe_np[j]
            pa_mpjpe_list.append(pa_mpjpe)
            mpjpe_list.append(mpjpe)

            sample = {
                'pred_joints_3d_eval': pred_eval_np[j],
                'gt_joints_3d': gt_eval_np[j],
                'pa_mpjpe': pa_mpjpe,
                'mpjpe': mpjpe,
            }
            all_samples.append(sample)

    print(f"\n[4/4] Saving predictions...")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Mean PA-MPJPE: {np.mean(pa_mpjpe_list):.2f} mm")
    print(f"  Mean MPJPE: {np.mean(mpjpe_list):.2f} mm")

    # Validate body proportions
    sample = all_samples[0]
    pred = sample['pred_joints_3d_eval']
    hip_width = np.linalg.norm(pred[0] - pred[3])
    shoulder_width = np.linalg.norm(pred[8] - pred[11])
    print(f"\n  Body proportions check:")
    print(f"    Hip width: {hip_width:.3f}m (expect ~0.15-0.25m)")
    print(f"    Shoulder width: {shoulder_width:.3f}m (expect ~0.30-0.45m)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output_path, np.mean(pa_mpjpe_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    extract_predictions(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
    )
