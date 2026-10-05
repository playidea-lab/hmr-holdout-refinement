#!/usr/bin/env python3
"""
Extract TokenHMR predictions for RL refinement on CUDA server.

Uses official TokenHMR evaluation protocol:
- J_regressor to compute 24 SMPL joints from vertices
- Pelvis = (L_Hip + R_Hip) / 2
- H36M 14 joints for evaluation: [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]

Output format:
- pred_joints_3d_eval (14, 3): H36M 14 joints for evaluation (pelvis-centered)
- gt_joints_3d (14, 3): GT H36M 14 joints (pelvis-centered)
- pa_mpjpe: PA-MPJPE for this sample
"""

import sys
import os
from pathlib import Path

# Add tokenhmr to path
tokenhmr_path = Path(os.environ.get("TOKENHMR_ROOT", "third_party/TokenHMR"))
sys.path.insert(0, str(tokenhmr_path))

import torch
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

# TokenHMR imports
from tokenhmr.lib.models import load_tokenhmr
from tokenhmr.lib.configs import dataset_eval_config
from tokenhmr.lib.datasets import create_dataset
from tokenhmr.lib.utils import recursive_to


def compute_similarity_transform(S1, S2):
    """Procrustes alignment (batched version using torch)."""
    if S1.dim() == 2:
        S1 = S1.unsqueeze(0)
        S2 = S2.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size = S1.shape[0]
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

    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))

    R = torch.bmm(torch.bmm(V, Z), U.permute(0, 2, 1))
    trace = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    scale = (trace / var1).unsqueeze(-1).unsqueeze(-1)
    t = mu2 - scale * torch.bmm(R, mu1)
    S1_hat = scale * torch.bmm(R, S1) + t
    S1_hat = S1_hat.permute(0, 2, 1)

    if squeeze:
        S1_hat = S1_hat.squeeze(0)

    return S1_hat


def compute_pa_mpjpe_batch(pred, gt):
    """Compute PA-MPJPE in mm (batched)."""
    pred_aligned = compute_similarity_transform(pred, gt)
    error = torch.sqrt(((pred_aligned - gt) ** 2).sum(dim=-1)).mean(dim=-1)
    return error * 1000


def extract_predictions(
    checkpoint_path: str,
    model_config: str,
    dataset_dir: str,
    output_path: str,
    batch_size: int = 32,
):
    """Extract TokenHMR predictions on 3DPW test set using official evaluation protocol."""

    print("=" * 60)
    print("TokenHMR Prediction Extraction for RL (CUDA Server)")
    print("=" * 60)

    print("\n[1/4] Loading TokenHMR model...")
    model, model_cfg = load_tokenhmr(
        checkpoint_path=checkpoint_path,
        model_cfg=model_config,
        dataset_dir=dataset_dir,
        is_demo=False,
        init_renderer=False,
    )

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    print("\n[2/4] Creating 3DPW-TEST dataset...")
    dataset_cfg = dataset_eval_config()['3DPW-TEST']
    if 'DATASET_FILE' in dataset_cfg:
        dataset_cfg['DATASET_FILE'] = os.path.join(dataset_dir, dataset_cfg['DATASET_FILE'])
    if 'IMG_DIR' in dataset_cfg:
        dataset_cfg['IMG_DIR'] = os.path.join(dataset_dir, dataset_cfg['IMG_DIR'])

    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Dataset size: {len(dataset)}")

    # Load J_regressor for computing 24 SMPL joints from vertices
    import smplx
    smpl = smplx.SMPL(model_path=model_cfg.SMPL.MODEL_PATH)
    J_regressor_24 = smpl.J_regressor.to(device).float()
    print(f"J_regressor_24 shape: {J_regressor_24.shape}")

    # H36M 14 joint indices from 24 SMPL joints
    h36m_14_from_24 = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]

    print(f"\n[3/4] Extracting predictions...")
    print(f"  Using J_regressor_24 to compute 24 joints from vertices")
    print(f"  Pelvis = (L_Hip + R_Hip) / 2")
    print(f"  Evaluation uses H36M 14 joints: {h36m_14_from_24}")

    all_samples = []
    pa_mpjpe_list = []
    mpjpe_list = []

    for batch in tqdm(dataloader):
        batch = recursive_to(batch, device)
        batch_size_curr = batch['img'].shape[0]

        with torch.no_grad():
            output = model(batch)

            pred_vertices = output['pred_vertices']
            gt_vertices = batch['vertices']

            # Compute 24 joints from vertices
            pred_joints_24 = torch.matmul(J_regressor_24, pred_vertices)
            gt_joints_24 = torch.matmul(J_regressor_24, gt_vertices)

            # Pelvis = (L_Hip + R_Hip) / 2
            pred_pelvis = (pred_joints_24[:, [1], :] + pred_joints_24[:, [2], :]) / 2.0
            gt_pelvis = (gt_joints_24[:, [1], :] + gt_joints_24[:, [2], :]) / 2.0

            # Pelvis-centered
            pred_joints_24_centered = pred_joints_24 - pred_pelvis
            gt_joints_24_centered = gt_joints_24 - gt_pelvis

            # Extract H36M 14 joints
            pred_eval = pred_joints_24_centered[:, h36m_14_from_24, :]
            gt_eval = gt_joints_24_centered[:, h36m_14_from_24, :]

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
        default=os.path.join(os.environ.get("TOKENHMR_ROOT", "third_party/TokenHMR"), "data/checkpoints/tokenhmr_model_latest.ckpt"),
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=os.path.join(os.environ.get("TOKENHMR_ROOT", "third_party/TokenHMR"), "data/checkpoints/model_config.yaml"),
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(os.environ.get("TOKENHMR_ROOT", "third_party/TokenHMR"), "data/datasets"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.environ.get("PRED_DIR", "data"), "tokenhmr_3dpw_cuda.pkl"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    extract_predictions(
        checkpoint_path=args.checkpoint,
        model_config=args.model_config,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )
