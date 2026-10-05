#!/usr/bin/env python3
"""
exp700: 4-backbone LOO validation for Supervised Refinement.

Runs LOO for all 4 backbones: TokenHMR, 4DHumans, SPIN, ProHMR.
Saves per-sequence and summary results with metrics.json for PiQ.

Usage:
    uv run python scripts/exp700_4backbone_loo.py --run_dir {RUN_DIR}
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


def compute_pa_mpjpe(pred, gt):
    pred_aligned = compute_similarity_transform(pred, gt)
    error = torch.sqrt(((pred_aligned - gt) ** 2).sum(dim=-1)).mean(dim=-1)
    return error * 1000


def compute_mpjpe(pred, gt):
    error = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1)
    return error * 1000


def evaluate(model, dataloader, device):
    model.eval()
    pa_before_list, pa_after_list = [], []
    mpjpe_before_list, mpjpe_after_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            pred = batch['pred'].to(device)
            gt = batch['gt'].to(device)
            delta = model(pred)
            refined = pred + delta
            pred_j = pred.view(-1, 14, 3)
            refined_j = refined.view(-1, 14, 3)
            gt_j = gt.view(-1, 14, 3)
            pa_b = compute_pa_mpjpe(pred_j, gt_j)
            pa_a = compute_pa_mpjpe(refined_j, gt_j)
            mpjpe_b = compute_mpjpe(pred_j, gt_j)
            mpjpe_a = compute_mpjpe(refined_j, gt_j)
            pa_before_list.extend(pa_b.cpu().numpy())
            pa_after_list.extend(pa_a.cpu().numpy())
            mpjpe_before_list.extend(mpjpe_b.cpu().numpy())
            mpjpe_after_list.extend(mpjpe_a.cpu().numpy())
    return {
        'pa_before': np.array(pa_before_list),
        'pa_after': np.array(pa_after_list),
        'mpjpe_before': np.array(mpjpe_before_list),
        'mpjpe_after': np.array(mpjpe_after_list),
    }


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


def run_loo_for_backbone(backbone_name, data_path, seq_to_indices, device, output_dir, epochs=30):
    print(f"\n{'='*60}")
    print(f"LOO Validation: {backbone_name}")
    print(f"{'='*60}")

    with open(data_path, 'rb') as f:
        samples = pickle.load(f)

    pred_joints = np.stack([s['pred_joints_3d_eval'] for s in samples])
    gt_joints = np.stack([s['gt_joints_3d'] for s in samples])

    sequences = sorted(seq_to_indices.keys())
    results = []
    all_pa_before = []
    all_pa_after = []
    all_mpjpe_before = []
    all_mpjpe_after = []

    for test_seq in tqdm(sequences, desc=f"{backbone_name} LOO"):
        test_indices = seq_to_indices[test_seq]
        train_indices = []
        for seq, indices in seq_to_indices.items():
            if seq != test_seq:
                train_indices.extend(indices)

        train_dataset = SimpleDataset(pred_joints[train_indices], gt_joints[train_indices])
        test_dataset = SimpleDataset(pred_joints[test_indices], gt_joints[test_indices])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

        model = train_model(train_loader, device, epochs=epochs)
        metrics = evaluate(model, test_loader, device)

        pa_before = metrics['pa_before'].mean()
        pa_after = metrics['pa_after'].mean()
        mpjpe_before = metrics['mpjpe_before'].mean()
        mpjpe_after = metrics['mpjpe_after'].mean()

        # Per-sample metrics for detailed analysis
        all_pa_before.extend(metrics['pa_before'])
        all_pa_after.extend(metrics['pa_after'])
        all_mpjpe_before.extend(metrics['mpjpe_before'])
        all_mpjpe_after.extend(metrics['mpjpe_after'])

        # Check if any sequence got WORSE
        n_improved = (metrics['pa_after'] < metrics['pa_before']).sum()
        n_total = len(metrics['pa_before'])

        results.append({
            'sequence': test_seq,
            'n_samples': len(test_indices),
            'pa_before': float(pa_before),
            'pa_after': float(pa_after),
            'mpjpe_before': float(mpjpe_before),
            'mpjpe_after': float(mpjpe_after),
            'improvement_mm': float(pa_before - pa_after),
            'improvement_pct': float(100 * (pa_before - pa_after) / pa_before),
            'n_improved': int(n_improved),
            'pct_improved': float(100 * n_improved / n_total),
        })

    # Summary stats
    all_pa_before = np.array(all_pa_before)
    all_pa_after = np.array(all_pa_after)
    all_mpjpe_before = np.array(all_mpjpe_before)
    all_mpjpe_after = np.array(all_mpjpe_after)

    total_samples = sum(r['n_samples'] for r in results)
    weighted_pa_before = sum(r['pa_before'] * r['n_samples'] for r in results) / total_samples
    weighted_pa_after = sum(r['pa_after'] * r['n_samples'] for r in results) / total_samples
    weighted_mpjpe_before = sum(r['mpjpe_before'] * r['n_samples'] for r in results) / total_samples
    weighted_mpjpe_after = sum(r['mpjpe_after'] * r['n_samples'] for r in results) / total_samples

    # Count sequences that got worse
    n_seq_worse = sum(1 for r in results if r['improvement_mm'] < 0)

    summary = {
        'backbone': backbone_name,
        'n_sequences': len(sequences),
        'n_samples': total_samples,
        'weighted_pa_before': float(weighted_pa_before),
        'weighted_pa_after': float(weighted_pa_after),
        'weighted_mpjpe_before': float(weighted_mpjpe_before),
        'weighted_mpjpe_after': float(weighted_mpjpe_after),
        'pa_improvement_mm': float(weighted_pa_before - weighted_pa_after),
        'pa_improvement_pct': float(100 * (weighted_pa_before - weighted_pa_after) / weighted_pa_before),
        'mpjpe_improvement_mm': float(weighted_mpjpe_before - weighted_mpjpe_after),
        'mpjpe_improvement_pct': float(100 * (weighted_mpjpe_before - weighted_mpjpe_after) / weighted_mpjpe_before),
        'n_seq_worse': n_seq_worse,
        'pct_samples_improved': float(100 * (all_pa_after < all_pa_before).sum() / len(all_pa_before)),
    }

    print(f"\n{backbone_name} Results:")
    print(f"  PA-MPJPE: {weighted_pa_before:.2f} -> {weighted_pa_after:.2f} mm "
          f"({summary['pa_improvement_pct']:.1f}%)")
    print(f"  MPJPE: {weighted_mpjpe_before:.2f} -> {weighted_mpjpe_after:.2f} mm "
          f"({summary['mpjpe_improvement_pct']:.1f}%)")
    print(f"  Sequences worse: {n_seq_worse}/{len(sequences)}")
    print(f"  Samples improved: {summary['pct_samples_improved']:.1f}%")

    # Save per-backbone results
    backbone_dir = output_dir / backbone_name
    backbone_dir.mkdir(parents=True, exist_ok=True)
    with open(backbone_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'per_sequence': results,
            'summary': summary,
            'per_sample_pa_before': all_pa_before,
            'per_sample_pa_after': all_pa_after,
            'per_sample_mpjpe_before': all_mpjpe_before,
            'per_sample_mpjpe_after': all_mpjpe_after,
        }, f)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--backbones", nargs='+', default=None,
                        help="Which backbones to run (default: all available)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load sequence info
    ann_file = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
    ann_data = np.load(ann_file, allow_pickle=True)
    imgnames = ann_data['imgname']

    seq_to_indices = defaultdict(list)
    for i, name in enumerate(imgnames):
        seq = name.split('/')[1]
        seq_to_indices[seq].append(i)

    print(f"3DPW test: {len(imgnames)} samples, {len(seq_to_indices)} sequences")

    # Define backbones
    all_backbones = {
        'TokenHMR': 'data/tokenhmr_3dpw_cuda.pkl',
        '4DHumans': 'data/4dhumans_3dpw_cuda.pkl',
        'SPIN': 'data/spin_3dpw_14j.pkl',
        'ProHMR': 'data/prohmr_3dpw_matched.pkl',
    }

    if args.backbones:
        backbones = {k: v for k, v in all_backbones.items() if k in args.backbones}
    else:
        backbones = {k: v for k, v in all_backbones.items() if Path(v).exists()}

    print(f"Running LOO for backbones: {list(backbones.keys())}")

    # Run LOO for each backbone
    all_results = []
    for backbone_name, data_path in backbones.items():
        if not Path(data_path).exists():
            print(f"SKIP {backbone_name}: {data_path} not found")
            continue

        result = run_loo_for_backbone(
            backbone_name, data_path, seq_to_indices, device, run_dir, epochs=args.epochs
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("4-BACKBONE LOO SUMMARY")
    print("=" * 80)
    print(f"{'Backbone':<12} {'PA Before':>10} {'PA After':>10} {'Δ mm':>8} {'Δ %':>8} {'MPJPE Before':>12} {'MPJPE After':>12} {'Seq Worse':>10}")
    print("-" * 84)

    for r in all_results:
        print(f"{r['backbone']:<12} {r['weighted_pa_before']:>10.2f} {r['weighted_pa_after']:>10.2f} "
              f"{r['pa_improvement_mm']:>8.2f} {r['pa_improvement_pct']:>7.1f}% "
              f"{r['weighted_mpjpe_before']:>12.2f} {r['weighted_mpjpe_after']:>12.2f} "
              f"{r['n_seq_worse']:>5}/{r['n_sequences']}")

    # Save summary
    with open(run_dir / 'summary.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Save metrics.json for PiQ
    metrics = {}
    for r in all_results:
        bb = r['backbone']
        metrics[f"{bb}_pa_before"] = r['weighted_pa_before']
        metrics[f"{bb}_pa_after"] = r['weighted_pa_after']
        metrics[f"{bb}_pa_improvement_pct"] = r['pa_improvement_pct']
        metrics[f"{bb}_mpjpe_before"] = r['weighted_mpjpe_before']
        metrics[f"{bb}_mpjpe_after"] = r['weighted_mpjpe_after']
        metrics[f"{bb}_n_seq_worse"] = r['n_seq_worse']
        metrics[f"{bb}_pct_samples_improved"] = r['pct_samples_improved']

    metrics['n_backbones'] = len(all_results)
    metrics['avg_pa_improvement_pct'] = np.mean([r['pa_improvement_pct'] for r in all_results])

    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
