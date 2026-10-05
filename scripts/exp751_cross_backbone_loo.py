#!/usr/bin/env python3
"""
exp751: Cross-Backbone LOO Transfer (Fair Comparison)

Addresses the unfair comparison in exp750 where cross-backbone transfer used
full data for train+test. This script uses LOO protocol for cross-backbone:

Protocol:
- For each test sequence s:
  - Train on backbone A's 23 sequences (excluding s)
  - Test on backbone B's sequence s
- This is fair: no sequence-level data leakage

Comparison:
- exp750 cross-bb (unfair): Train:Token→Test:4DH 37.44mm (-31.1%) — uses ALL data
- exp751 cross-bb (fair): LOO cross-backbone — true generalization test

Usage:
    uv run python scripts/exp751_cross_backbone_loo.py --run_dir {RUN_DIR}
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


class ConditionedRefinementMLP(nn.Module):
    """MLP that takes joint predictions + image features as input."""

    def __init__(self, joint_dim=42, feat_dim=384, hidden_dims=None, output_dim=42):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        input_dim = joint_dim + feat_dim
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

    def forward(self, joints, features):
        x = torch.cat([joints, features], dim=-1)
        return self.net(x)


class RefinementMLP(nn.Module):
    """Plain MLP (no DINOv3 conditioning) for comparison."""

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


class ConditionedDataset(Dataset):
    def __init__(self, pred_joints, gt_joints, features):
        self.pred = torch.from_numpy(pred_joints.reshape(-1, 42)).float()
        self.gt = torch.from_numpy(gt_joints.reshape(-1, 42)).float()
        self.features = torch.from_numpy(features).float()
        self.residual = self.gt - self.pred

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, idx):
        return {
            'pred': self.pred[idx],
            'gt': self.gt[idx],
            'residual': self.residual[idx],
            'features': self.features[idx],
        }


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


def compute_pa_mpjpe(pred, gt):
    pred_aligned = compute_similarity_transform(pred, gt)
    error = torch.sqrt(((pred_aligned - gt) ** 2).sum(dim=-1)).mean(dim=-1)
    return error * 1000


def evaluate_conditioned(model, dataloader, device):
    model.eval()
    pa_before_list, pa_after_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            pred = batch['pred'].to(device)
            gt = batch['gt'].to(device)
            features = batch['features'].to(device)
            delta = model(pred, features)
            refined = pred + delta
            pred_j = pred.view(-1, 14, 3)
            refined_j = refined.view(-1, 14, 3)
            gt_j = gt.view(-1, 14, 3)
            pa_b = compute_pa_mpjpe(pred_j, gt_j)
            pa_a = compute_pa_mpjpe(refined_j, gt_j)
            pa_before_list.extend(pa_b.cpu().numpy())
            pa_after_list.extend(pa_a.cpu().numpy())
    return np.array(pa_before_list), np.array(pa_after_list)


def train_conditioned(train_loader, device, feat_dim=384, epochs=30):
    model = ConditionedRefinementMLP(feat_dim=feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            pred = batch['pred'].to(device)
            features = batch['features'].to(device)
            residual = batch['residual'].to(device)
            delta = model(pred, features)
            loss = criterion(delta, residual)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("exp751: Cross-Backbone LOO Transfer (Fair Comparison)")
    print("=" * 60)

    # Load DINOv3 features (cached from exp750)
    feat_cache = Path('data/dinov3_3dpw_features.npy')
    if not feat_cache.exists():
        print(f"ERROR: DINOv3 features not found at {feat_cache}")
        print("Run exp750 first to extract features.")
        return
    features = np.load(feat_cache)
    feat_dim = features.shape[1]
    print(f"Loaded DINOv3 features: {features.shape}")

    # Load sequence info
    ann_file = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
    ann_data = np.load(ann_file, allow_pickle=True)
    imgnames = ann_data['imgname']

    seq_to_indices = defaultdict(list)
    for i, name in enumerate(imgnames):
        seq = name.split('/')[1]
        seq_to_indices[seq].append(i)
    sequences = sorted(seq_to_indices.keys())
    print(f"3DPW test: {len(imgnames)} samples, {len(sequences)} sequences")

    # Load all backbones
    backbones = {
        'TokenHMR': 'data/tokenhmr_3dpw_cuda.pkl',
        '4DHumans': 'data/4dhumans_3dpw_cuda.pkl',
        'SPIN': 'data/spin_3dpw_14j.pkl',
    }
    backbones = {k: v for k, v in backbones.items() if Path(v).exists()}
    bb_names = list(backbones.keys())

    # Load backbone data
    bb_data = {}
    for name, path in backbones.items():
        with open(path, 'rb') as f:
            samples = pickle.load(f)
        bb_data[name] = {
            'pred': np.stack([s['pred_joints_3d_eval'] for s in samples]),
            'gt': np.stack([s['gt_joints_3d'] for s in samples]),
        }
        n = min(len(bb_data[name]['pred']), len(features))
        bb_data[name]['pred'] = bb_data[name]['pred'][:n]
        bb_data[name]['gt'] = bb_data[name]['gt'][:n]

    # Cross-backbone LOO: Train on backbone A (22 seq), Test on backbone B (1 seq)
    all_transfer_results = []

    for train_bb in bb_names:
        for test_bb in bb_names:
            label = "SAME" if train_bb == test_bb else "TRANSFER"
            print(f"\n{'='*60}")
            print(f"[{label}] Train: {train_bb} → Test: {test_bb} (LOO)")
            print(f"{'='*60}")

            all_pa_before = []
            all_pa_after = []

            for test_seq in tqdm(sequences, desc=f"{train_bb}→{test_bb}"):
                test_indices = [i for i in seq_to_indices[test_seq]
                                if i < len(bb_data[test_bb]['pred'])]
                train_indices = []
                for seq, indices in seq_to_indices.items():
                    if seq != test_seq:
                        train_indices.extend(
                            [i for i in indices if i < len(bb_data[train_bb]['pred'])])

                if len(test_indices) == 0:
                    continue

                train_dataset = ConditionedDataset(
                    bb_data[train_bb]['pred'][train_indices],
                    bb_data[train_bb]['gt'][train_indices],
                    features[train_indices])
                test_dataset = ConditionedDataset(
                    bb_data[test_bb]['pred'][test_indices],
                    bb_data[test_bb]['gt'][test_indices],
                    features[test_indices])

                train_loader = DataLoader(train_dataset, batch_size=256,
                                          shuffle=True, num_workers=2)
                test_loader = DataLoader(test_dataset, batch_size=256,
                                         shuffle=False, num_workers=2)

                model = train_conditioned(train_loader, device,
                                          feat_dim=feat_dim, epochs=args.epochs)
                pa_before, pa_after = evaluate_conditioned(model, test_loader, device)

                all_pa_before.extend(pa_before)
                all_pa_after.extend(pa_after)

            all_pa_before = np.array(all_pa_before)
            all_pa_after = np.array(all_pa_after)

            result = {
                'train_bb': train_bb,
                'test_bb': test_bb,
                'type': label,
                'protocol': 'LOO',
                'pa_before': float(all_pa_before.mean()),
                'pa_after': float(all_pa_after.mean()),
                'improvement_mm': float(all_pa_before.mean() - all_pa_after.mean()),
                'improvement_pct': float(100 * (all_pa_before.mean() - all_pa_after.mean())
                                         / all_pa_before.mean()),
                'n_samples': len(all_pa_before),
                'pct_improved': float(100 * (all_pa_after < all_pa_before).sum()
                                      / len(all_pa_before)),
                'n_worse': int((all_pa_after > all_pa_before).sum()),
            }
            all_transfer_results.append(result)

            print(f"  {train_bb}→{test_bb}: {result['pa_before']:.2f} → {result['pa_after']:.2f}mm "
                  f"({result['improvement_pct']:.1f}%, {result['pct_improved']:.1f}% improved)")

    # Summary
    print(f"\n{'='*80}")
    print("CROSS-BACKBONE LOO TRANSFER SUMMARY")
    print(f"{'='*80}")
    print(f"{'Train→Test':<25} {'PA Before':>10} {'PA After':>10} {'Improv%':>10} {'Improved%':>10}")
    print("-" * 65)

    for r in all_transfer_results:
        label = f"{r['train_bb']}→{r['test_bb']}"
        print(f"{label:<25} {r['pa_before']:>10.2f} {r['pa_after']:>10.2f} "
              f"{r['improvement_pct']:>9.1f}% {r['pct_improved']:>9.1f}%")

    # Compare with exp750 (unfair) and exp700 (plain MLP LOO)
    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS RESULTS")
    print(f"{'='*80}")
    print("exp700 (plain MLP LOO):   TokenHMR 38.86mm (-20.9%), 4DHumans 41.13mm (-24.3%)")
    print("exp750 LOO (conditioned): TokenHMR 43.23mm (-12.0%), 4DHumans 46.78mm (-13.9%)")
    print("exp750 cross-bb (UNFAIR): Token→4DH 37.44mm (-31.1%)")

    # Save
    with open(run_dir / 'results.pkl', 'wb') as f:
        pickle.dump(all_transfer_results, f)

    metrics = {'protocol': 'cross_backbone_LOO'}
    for r in all_transfer_results:
        key = f"{r['train_bb']}_to_{r['test_bb']}"
        metrics[f'{key}_pa_before'] = r['pa_before']
        metrics[f'{key}_pa_after'] = r['pa_after']
        metrics[f'{key}_improvement_pct'] = r['improvement_pct']

    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
