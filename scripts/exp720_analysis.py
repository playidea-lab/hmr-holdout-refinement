#!/usr/bin/env python3
"""
exp720-741: Analysis and baselines for Paper A.

- exp720: Per-joint improvement analysis + heatmap
- exp721: Error-stratified analysis (initial error vs improvement)
- exp723: Cross-backbone transfer (train on one, test on another)
- exp740: Mean regression baseline
- exp741: Linear regression baseline

Usage:
    uv run python scripts/exp720_analysis.py --run_dir {RUN_DIR} --analysis all
"""

import os
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RefinementMLP(nn.Module):
    def __init__(self, input_dim=42, hidden_dims=None, output_dim=42):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.LayerNorm(hdim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class SimpleDataset(Dataset):
    def __init__(self, pred_joints, gt_joints):
        self.pred = torch.from_numpy(pred_joints.reshape(-1, 42)).float()
        self.gt = torch.from_numpy(gt_joints.reshape(-1, 42)).float()
        self.residual = self.gt - self.pred

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, idx):
        return {'pred': self.pred[idx], 'gt': self.gt[idx], 'residual': self.residual[idx]}


def compute_similarity_transform(S1, S2):
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
    return S1_hat.permute(0, 2, 1)


def compute_per_joint_error(pred, gt):
    """Per-joint MPJPE in mm (B, 14)."""
    return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)) * 1000


def train_model(train_loader, device, epochs=30, hidden_dims=None):
    model = RefinementMLP(hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            pred = batch['pred'].to(device)
            residual = batch['residual'].to(device)
            delta = model(pred)
            loss = criterion(delta, residual)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    return model


# Joint names for the 14-joint evaluation set
JOINT_NAMES = [
    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle',
    'L_Hip', 'L_Knee', 'L_Ankle', 'Neck',
    'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist',
]


def load_backbone_data(backbone_name):
    backbone_paths = {
        'TokenHMR': 'data/tokenhmr_3dpw_cuda.pkl',
        '4DHumans': 'data/4dhumans_3dpw_cuda.pkl',
        'SPIN': 'data/spin_3dpw_14j.pkl',
    }
    with open(backbone_paths[backbone_name], 'rb') as f:
        samples = pickle.load(f)
    pred = np.stack([s['pred_joints_3d_eval'] for s in samples])
    gt = np.stack([s['gt_joints_3d'] for s in samples])
    return pred, gt


def load_seq_mapping():
    ann_file = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
    ann_data = np.load(ann_file, allow_pickle=True)
    imgnames = ann_data['imgname']
    seq_to_indices = defaultdict(list)
    for i, name in enumerate(imgnames):
        seq = name.split('/')[1]
        seq_to_indices[seq].append(i)
    return seq_to_indices


def exp720_per_joint(run_dir, device):
    """Per-joint improvement analysis across backbones."""
    print("\n" + "=" * 60)
    print("exp720: Per-Joint Analysis")
    print("=" * 60)

    seq_to_indices = load_seq_mapping()
    sequences = sorted(seq_to_indices.keys())

    results = {}
    for backbone in ['TokenHMR', '4DHumans', 'SPIN']:
        print(f"\n  {backbone}...")
        pred_joints, gt_joints = load_backbone_data(backbone)

        # Collect per-joint errors across all LOO folds
        all_before_pj = []  # (N, 14)
        all_after_pj = []

        for test_seq in tqdm(sequences, desc=f"{backbone}"):
            test_indices = seq_to_indices[test_seq]
            train_indices = []
            for seq in sequences:
                if seq != test_seq:
                    train_indices.extend(seq_to_indices[seq])

            train_dataset = SimpleDataset(pred_joints[train_indices], gt_joints[train_indices])
            test_dataset = SimpleDataset(pred_joints[test_indices], gt_joints[test_indices])

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

            model = train_model(train_loader, device)
            model.eval()

            with torch.no_grad():
                for batch in test_loader:
                    pred = batch['pred'].to(device)
                    gt = batch['gt'].to(device)
                    delta = model(pred)
                    refined = pred + delta

                    pred_j = pred.view(-1, 14, 3)
                    refined_j = refined.view(-1, 14, 3)
                    gt_j = gt.view(-1, 14, 3)

                    before_pj = compute_per_joint_error(pred_j, gt_j)
                    after_pj = compute_per_joint_error(refined_j, gt_j)

                    all_before_pj.extend(before_pj.cpu().numpy())
                    all_after_pj.extend(after_pj.cpu().numpy())

        all_before_pj = np.array(all_before_pj)
        all_after_pj = np.array(all_after_pj)

        per_joint_before = all_before_pj.mean(axis=0)
        per_joint_after = all_after_pj.mean(axis=0)
        per_joint_improvement = per_joint_before - per_joint_after
        per_joint_improvement_pct = 100 * per_joint_improvement / per_joint_before

        results[backbone] = {
            'per_joint_before': per_joint_before.tolist(),
            'per_joint_after': per_joint_after.tolist(),
            'per_joint_improvement_mm': per_joint_improvement.tolist(),
            'per_joint_improvement_pct': per_joint_improvement_pct.tolist(),
            'joint_names': JOINT_NAMES,
        }

        print(f"\n  {backbone} Per-Joint Results (MPJPE mm):")
        print(f"  {'Joint':<15} {'Before':>8} {'After':>8} {'Δ mm':>8} {'Δ %':>8}")
        print(f"  {'-'*50}")
        for j in range(14):
            print(f"  {JOINT_NAMES[j]:<15} {per_joint_before[j]:>8.2f} {per_joint_after[j]:>8.2f} "
                  f"{per_joint_improvement[j]:>8.2f} {per_joint_improvement_pct[j]:>7.1f}%")

    out_dir = run_dir / 'per_joint'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def exp721_error_stratified(run_dir, device):
    """Error-stratified analysis: how does improvement correlate with initial error?"""
    print("\n" + "=" * 60)
    print("exp721: Error-Stratified Analysis")
    print("=" * 60)

    seq_to_indices = load_seq_mapping()
    sequences = sorted(seq_to_indices.keys())

    results = {}
    for backbone in ['TokenHMR', '4DHumans', 'SPIN']:
        print(f"\n  {backbone}...")
        pred_joints, gt_joints = load_backbone_data(backbone)

        all_pa_before = []
        all_pa_after = []

        for test_seq in tqdm(sequences, desc=f"{backbone}"):
            test_indices = seq_to_indices[test_seq]
            train_indices = []
            for seq in sequences:
                if seq != test_seq:
                    train_indices.extend(seq_to_indices[seq])

            train_dataset = SimpleDataset(pred_joints[train_indices], gt_joints[train_indices])
            test_dataset = SimpleDataset(pred_joints[test_indices], gt_joints[test_indices])

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

            model = train_model(train_loader, device)
            model.eval()

            with torch.no_grad():
                for batch in test_loader:
                    pred = batch['pred'].to(device)
                    gt = batch['gt'].to(device)
                    delta = model(pred)
                    refined = pred + delta

                    pred_j = pred.view(-1, 14, 3)
                    refined_j = refined.view(-1, 14, 3)
                    gt_j = gt.view(-1, 14, 3)

                    pa_b = compute_per_joint_error(pred_j, gt_j).mean(dim=-1)
                    pa_a = compute_per_joint_error(refined_j, gt_j).mean(dim=-1)

                    all_pa_before.extend(pa_b.cpu().numpy())
                    all_pa_after.extend(pa_a.cpu().numpy())

        all_pa_before = np.array(all_pa_before)
        all_pa_after = np.array(all_pa_after)
        improvement = all_pa_before - all_pa_after

        # Stratify by initial error quartiles
        quartiles = np.percentile(all_pa_before, [25, 50, 75])
        bins = [0, quartiles[0], quartiles[1], quartiles[2], np.inf]
        bin_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']

        strat_results = {}
        for i in range(4):
            mask = (all_pa_before >= bins[i]) & (all_pa_before < bins[i+1])
            n = mask.sum()
            strat_results[bin_labels[i]] = {
                'n_samples': int(n),
                'mean_before': float(all_pa_before[mask].mean()),
                'mean_after': float(all_pa_after[mask].mean()),
                'mean_improvement': float(improvement[mask].mean()),
                'pct_improved': float(100 * (improvement[mask] > 0).sum() / n),
            }

        results[backbone] = strat_results

        print(f"\n  {backbone} Error-Stratified Results:")
        print(f"  {'Quartile':<15} {'N':>6} {'Before':>8} {'After':>8} {'Δ mm':>8} {'Improved':>10}")
        print(f"  {'-'*58}")
        for label, r in strat_results.items():
            print(f"  {label:<15} {r['n_samples']:>6} {r['mean_before']:>8.2f} {r['mean_after']:>8.2f} "
                  f"{r['mean_improvement']:>8.2f} {r['pct_improved']:>9.1f}%")

    out_dir = run_dir / 'error_stratified'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def exp723_cross_backbone(run_dir, device):
    """Cross-backbone transfer: train on one backbone, test on another."""
    print("\n" + "=" * 60)
    print("exp723: Cross-Backbone Transfer")
    print("=" * 60)

    seq_to_indices = load_seq_mapping()
    sequences = sorted(seq_to_indices.keys())

    backbones = ['TokenHMR', '4DHumans', 'SPIN']
    backbone_data = {}
    for bb in backbones:
        pred, gt = load_backbone_data(bb)
        backbone_data[bb] = (pred, gt)

    # For each train/test backbone pair, train on ALL data from train backbone,
    # evaluate on ALL data from test backbone
    results = {}
    for train_bb in backbones:
        for test_bb in backbones:
            print(f"\n  Train: {train_bb} -> Test: {test_bb}...")
            train_pred, train_gt = backbone_data[train_bb]
            test_pred, test_gt = backbone_data[test_bb]

            train_dataset = SimpleDataset(train_pred, train_gt)
            test_dataset = SimpleDataset(test_pred, test_gt)

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

            model = train_model(train_loader, device, epochs=30)
            model.eval()

            all_pa_before = []
            all_pa_after = []
            with torch.no_grad():
                for batch in test_loader:
                    pred = batch['pred'].to(device)
                    gt = batch['gt'].to(device)
                    delta = model(pred)
                    refined = pred + delta

                    pred_j = pred.view(-1, 14, 3)
                    refined_j = refined.view(-1, 14, 3)
                    gt_j = gt.view(-1, 14, 3)

                    pa_b = compute_per_joint_error(pred_j, gt_j).mean(dim=-1)
                    pa_a = compute_per_joint_error(refined_j, gt_j).mean(dim=-1)

                    all_pa_before.extend(pa_b.cpu().numpy())
                    all_pa_after.extend(pa_a.cpu().numpy())

            all_pa_before = np.array(all_pa_before)
            all_pa_after = np.array(all_pa_after)

            key = f"{train_bb}->{test_bb}"
            results[key] = {
                'train_backbone': train_bb,
                'test_backbone': test_bb,
                'pa_before': float(all_pa_before.mean()),
                'pa_after': float(all_pa_after.mean()),
                'improvement_mm': float(all_pa_before.mean() - all_pa_after.mean()),
                'improvement_pct': float(100 * (all_pa_before.mean() - all_pa_after.mean()) / all_pa_before.mean()),
                'pct_improved': float(100 * (all_pa_after < all_pa_before).sum() / len(all_pa_before)),
            }
            print(f"    {all_pa_before.mean():.2f} -> {all_pa_after.mean():.2f} mm "
                  f"({results[key]['improvement_pct']:.1f}%)")

    # Print transfer matrix
    print(f"\n  Transfer Matrix (PA-MPJPE after, mm):")
    print(f"  {'Train \\ Test':<15}", end="")
    for bb in backbones:
        print(f" {bb:>10}", end="")
    print()
    print(f"  {'-'*50}")
    for train_bb in backbones:
        print(f"  {train_bb:<15}", end="")
        for test_bb in backbones:
            key = f"{train_bb}->{test_bb}"
            print(f" {results[key]['pa_after']:>10.2f}", end="")
        print()

    out_dir = run_dir / 'cross_backbone'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def exp740_741_baselines(run_dir, device):
    """Mean regression and linear regression baselines."""
    print("\n" + "=" * 60)
    print("exp740-741: Baselines")
    print("=" * 60)

    seq_to_indices = load_seq_mapping()
    sequences = sorted(seq_to_indices.keys())

    results = {}
    for backbone in ['TokenHMR', '4DHumans', 'SPIN']:
        print(f"\n  {backbone}...")
        pred_joints, gt_joints = load_backbone_data(backbone)

        all_pa_before_mean = []
        all_pa_after_mean = []
        all_pa_after_linear = []

        for test_seq in tqdm(sequences, desc=f"{backbone} baselines"):
            test_indices = seq_to_indices[test_seq]
            train_indices = []
            for seq in sequences:
                if seq != test_seq:
                    train_indices.extend(seq_to_indices[seq])

            train_pred = pred_joints[train_indices].reshape(-1, 42)
            train_gt = gt_joints[train_indices].reshape(-1, 42)
            test_pred = pred_joints[test_indices].reshape(-1, 42)
            test_gt = gt_joints[test_indices].reshape(-1, 42)

            train_residual = train_gt - train_pred

            # exp740: Mean regression (constant shift)
            mean_shift = train_residual.mean(axis=0)
            mean_refined = test_pred + mean_shift

            # exp741: Linear regression (42->42)
            # X @ W + b = Y  =>  W = pinv(X) @ Y
            X = np.column_stack([train_pred, np.ones(len(train_pred))])
            W = np.linalg.lstsq(X, train_residual, rcond=None)[0]
            X_test = np.column_stack([test_pred, np.ones(len(test_pred))])
            linear_delta = X_test @ W
            linear_refined = test_pred + linear_delta

            # Compute PA-MPJPE
            test_pred_t = torch.from_numpy(test_pred.reshape(-1, 14, 3)).float()
            test_gt_t = torch.from_numpy(test_gt.reshape(-1, 14, 3)).float()
            mean_refined_t = torch.from_numpy(mean_refined.reshape(-1, 14, 3)).float()
            linear_refined_t = torch.from_numpy(linear_refined.astype(np.float32).reshape(-1, 14, 3))

            pa_before = compute_per_joint_error(test_pred_t, test_gt_t).mean(dim=-1)
            pa_after_mean = compute_per_joint_error(mean_refined_t, test_gt_t).mean(dim=-1)
            pa_after_linear = compute_per_joint_error(linear_refined_t, test_gt_t).mean(dim=-1)

            all_pa_before_mean.extend(pa_before.numpy())
            all_pa_after_mean.extend(pa_after_mean.numpy())
            all_pa_after_linear.extend(pa_after_linear.numpy())

        all_pa_before_mean = np.array(all_pa_before_mean)
        all_pa_after_mean = np.array(all_pa_after_mean)
        all_pa_after_linear = np.array(all_pa_after_linear)

        results[backbone] = {
            'baseline': {
                'pa_mpjpe': float(all_pa_before_mean.mean()),
            },
            'mean_regression': {
                'pa_mpjpe': float(all_pa_after_mean.mean()),
                'improvement_mm': float(all_pa_before_mean.mean() - all_pa_after_mean.mean()),
                'improvement_pct': float(100 * (all_pa_before_mean.mean() - all_pa_after_mean.mean()) / all_pa_before_mean.mean()),
            },
            'linear_regression': {
                'pa_mpjpe': float(all_pa_after_linear.mean()),
                'improvement_mm': float(all_pa_before_mean.mean() - all_pa_after_linear.mean()),
                'improvement_pct': float(100 * (all_pa_before_mean.mean() - all_pa_after_linear.mean()) / all_pa_before_mean.mean()),
            },
        }

        print(f"\n  {backbone} Baselines:")
        print(f"    Original:         {results[backbone]['baseline']['pa_mpjpe']:.2f} mm")
        print(f"    Mean regression:  {results[backbone]['mean_regression']['pa_mpjpe']:.2f} mm "
              f"({results[backbone]['mean_regression']['improvement_pct']:.1f}%)")
        print(f"    Linear regression:{results[backbone]['linear_regression']['pa_mpjpe']:.2f} mm "
              f"({results[backbone]['linear_regression']['improvement_pct']:.1f}%)")

    out_dir = run_dir / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--analysis", type=str, required=True,
                        choices=['per_joint', 'error_stratified', 'cross_backbone',
                                 'baselines', 'all'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    analysis_funcs = {
        'per_joint': exp720_per_joint,
        'error_stratified': exp721_error_stratified,
        'cross_backbone': exp723_cross_backbone,
        'baselines': exp740_741_baselines,
    }

    if args.analysis == 'all':
        to_run = list(analysis_funcs.keys())
    else:
        to_run = [args.analysis]

    all_metrics = {}
    for name in to_run:
        func = analysis_funcs[name]
        results = func(run_dir, device)
        all_metrics[name] = results

    # Save combined metrics
    with open(run_dir / 'metrics.json', 'w') as f:
        # Flatten for PiQ
        flat_metrics = {'analysis_types': to_run}
        json.dump(flat_metrics, f, indent=2)

    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
