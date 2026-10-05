"""SPIN 기준치(seq-LOO)를 행 순서 교정 후 다시 잰다 — tab:main SPIN 행 대체용.

시드 42/123/456(tab:main 과 같은 시드), PA-MPJPE 와 MPJPE(0번 관절 기준 정렬),
시퀀스별 개선 여부, Wilcoxon(24 시퀀스 평균 짝), 시퀀스·배우 부트스트랩 95% 구간.
MPJPE 정의 점검: 미정제 baseline 이 논문 92.9 mm 를 재현하는지 먼저 출력한다.
RefineMLP 는 x + net(x) 를 내므로 목표는 GT 관절 그 자체다(exp775 와 같다).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import exp775_share_seeds as E  # noqa: E402

SEEDS = (42, 123, 456)
N_BOOT = 5000


def mpjpe_rows(pred, gt):
    """0번 관절 기준 정렬 MPJPE(mm), 행별."""
    p = pred.reshape(-1, 14, 3)
    g = gt.reshape(-1, 14, 3)
    p = p - p[:, :1]
    g = g - g[:, :1]
    return np.linalg.norm(p - g, axis=-1).mean(-1) * 1000.0


def main() -> int:
    names, center, seq, subj = E.load_rows()
    p, g = E.load_backbone("SPIN", names, center, spin_reorder=True)
    X = torch.as_tensor(p.reshape(len(p), -1), dtype=torch.float32, device=E.DEV)
    Y = torch.as_tensor(g.reshape(len(g), -1), dtype=torch.float32, device=E.DEV)
    base_pa = E.pa_torch(X, Y).cpu().numpy()
    base_mp = mpjpe_rows(p, g)
    print(f"baseline PA {base_pa.mean():.3f}  MPJPE(root0) {base_mp.mean():.3f}", flush=True)

    seqs = sorted(set(seq.tolist()))
    seq_idx = {s: np.nonzero(seq == s)[0] for s in seqs}
    all_rows = np.arange(len(base_pa))
    after_pa = np.zeros((len(SEEDS), len(base_pa)))
    after_mp = np.zeros_like(after_pa)
    for si, sd in enumerate(SEEDS):
        for s in seqs:
            te = seq_idx[s]
            tr = np.setdiff1d(all_rows, te)
            m = E.train(X, Y, tr, sd, 10)
            tt = torch.as_tensor(te, device=E.DEV)
            with torch.no_grad():
                out = m(X[tt])
            after_pa[si, te] = E.pa_torch(out, Y[tt]).cpu().numpy()
            after_mp[si, te] = mpjpe_rows(out.cpu().numpy(), g[te])
        print(f"seed {sd}: PA {after_pa[si].mean():.3f} ({E.pct(base_pa, after_pa[si]):+.2f}%)  "
              f"MPJPE {after_mp[si].mean():.3f} ({E.pct(base_mp, after_mp[si]):+.2f}%)", flush=True)

    a_pa, a_mp = after_pa.mean(0), after_mp.mean(0)
    pa_pcts = [E.pct(base_pa, after_pa[i]) for i in range(len(SEEDS))]
    mp_pcts = [E.pct(base_mp, after_mp[i]) for i in range(len(SEEDS))]
    print(f"@spin_pa_after={a_pa.mean():.3f} @spin_pa_pct={np.mean(pa_pcts):.3f} "
          f"@spin_pa_pct_sd={np.std(pa_pcts, ddof=1):.3f} @spin_pa_mm_sd={after_pa.mean(1).std(ddof=1):.3f}", flush=True)
    print(f"@spin_mp_before={base_mp.mean():.3f} @spin_mp_after={a_mp.mean():.3f} @spin_mp_pct={np.mean(mp_pcts):.3f}", flush=True)

    sb = np.array([base_pa[seq_idx[s]].mean() for s in seqs])
    sa = np.array([a_pa[seq_idx[s]].mean() for s in seqs])
    nw = int((sa <= sb).sum())
    w = stats.wilcoxon(sa, sb)
    print(f"@spin_noworse={nw} @spin_wilcoxon_W={w.statistic:.1f} @spin_wilcoxon_p={w.pvalue:.3e}", flush=True)

    rng = np.random.default_rng(0)
    sw = np.array([len(seq_idx[s]) for s in seqs])
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(seqs), len(seqs))
        b = (sb[pick] * sw[pick]).sum()
        a = (sa[pick] * sw[pick]).sum()
        boots.append(100 * (b - a) / b)
    print(f"@spin_seqboot_lo={np.percentile(boots, 2.5):.2f} @spin_seqboot_hi={np.percentile(boots, 97.5):.2f}", flush=True)
    subs = sorted(set(subj.tolist()))
    ub = np.array([base_pa[subj == u].sum() for u in subs])
    ua = np.array([a_pa[subj == u].sum() for u in subs])
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(subs), len(subs))
        boots.append(100 * (ub[pick].sum() - ua[pick].sum()) / ub[pick].sum())
    print(f"@spin_subjboot_lo={np.percentile(boots, 2.5):.2f} @spin_subjboot_hi={np.percentile(boots, 97.5):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
