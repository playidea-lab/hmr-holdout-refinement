"""exp775: 개인화 몫을 시드·추첨 분포로 다시 잰다 — CVIU 심사 M1/M2/M3 대응.

exp774 는 시드 1개, 크기 맞춤 추첨 3회였다. 여기서는
  (a) seq-LOO / 행 단위 subject-LOO / 시퀀스 통째 subject-LOO 를 시드 S 개로,
  (b) 크기 맞춘 seq-LOO(시퀀스 단위 표집, 행 단위 학습 프레임 수에 맞춤),
  (c) 크기 맞춘 seq-LOO(모든 시퀀스에서 프레임 균등 표집 — 다양성 보존, 같은 프레임 수),
  (d) 크기 맞춘 seq-LOO(시퀀스 단위 표집, 시퀀스 통째 프로토콜의 학습 프레임 수에 맞춤)
를 시드 S × 추첨 D 로 잰다. 몫 = 1 − (subject-LOO 개선율)/(대응 seq-LOO 개선율).
배우별·프레임 가중 집계 둘 다, (시드,추첨) 쌍에 대한 평균·부트스트랩 95% 구간·개별값 범위.

SPIN: spin_3dpw_14j.pkl 의 행 순서는 SPIN 저장소 3dpw_test.npz 순서다(spin_check4 로 확인).
TokenHMR npz 순서로 재배열해야 시퀀스·배우 라벨이 맞는다. exp774 까지는 재배열하지 않았다.

모델·손실·옵티마이저·스케줄·배치 크기는 exp803b 와 같다. 셔플만 DataLoader 대신
GPU randperm 으로 바꿔 빠르게 했다(같은 알고리즘, 다른 난수열). 체크포인트는 저장하지 않는다.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402

OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp775"))
SPIN_NPZ = os.path.join(os.environ.get("SPIN_ROOT", "third_party/SPIN"), "data/dataset_extras/3dpw_test.npz")
DEFAULTS = dict(n_seeds=5, n_draws=10, epochs=10, backbones="TokenHMR,4DHumans,SPIN",
                parts="a,b,c,d", n_boot=2000)
BETA_DECIMALS, N_BETAS = 3, 10
BATCH, LR, WD = 256, 1e-3, 1e-4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECS = [("b", "sequnit_rowcount"), ("c", "frameuniform_rowcount"), ("d", "sequnit_wholecount")]


# ---------------------------------------------------------------- 학습·평가
def pa_torch(pred, gt):
    """표준 Procrustes PA-MPJPE(mm), H.compute_pa_mpjpe 와 같은 규약을 배치로."""
    p = pred.reshape(-1, 14, 3).double()
    g = gt.reshape(-1, 14, 3).double()
    pc, gc = p - p.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    sp = pc.pow(2).sum((1, 2), keepdim=True).sqrt()
    sg = gc.pow(2).sum((1, 2), keepdim=True).sqrt()
    pc, gc = pc / (sp + 1e-8), gc / (sg + 1e-8)
    U, S, Vh = torch.linalg.svd(pc.transpose(1, 2) @ gc)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], 1))
    R = V @ D @ U.transpose(1, 2)
    scale = (S[:, 0] + S[:, 1] + d * S[:, 2]).view(-1, 1, 1)
    al = sg * scale * (pc @ R.transpose(1, 2)) + g.mean(1, keepdim=True)
    return ((al - g).pow(2).sum(-1).sqrt().mean(-1) * 1000.0).float()


def train(X, Y, idx, seed, epochs):
    """exp803b.train_mlp 와 같은 모델·손실·옵티마이저·스케줄. 셔플만 GPU randperm."""
    torch.manual_seed(seed)
    model = H.RefineMLP().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    idx = torch.as_tensor(idx, device=DEV)
    x, y = X[idx], Y[idx]
    n = len(idx)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            nn.functional.l1_loss(model(x[b]), y[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, X, Y, idx):
    idx = torch.as_tensor(idx, device=DEV)
    return pa_torch(model(X[idx]), Y[idx]).cpu().numpy()


def pct(base, after):
    return float(100.0 * (base.mean() - after.mean()) / base.mean())


# ---------------------------------------------------------------- 데이터
def load_rows():
    mocap = np.load(H.MOCAP, allow_pickle=True)
    names = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq = np.array([Path(n).parts[-2] for n in names])
    lut, subj = {}, []
    for b in mocap["betas"]:
        subj.append(lut.setdefault(tuple(np.round(b[:N_BETAS], BETA_DECIMALS)), len(lut)))
    return names, mocap["center"], seq, np.array(subj)


def spin_order(tk_names, tk_center):
    """SPIN pkl 행 r → TokenHMR npz 색인. (시퀀스, 프레임, 중심 반올림) 키로 맞춘다."""
    sp = np.load(SPIN_NPZ, allow_pickle=True)
    sp_names = [s.decode() if isinstance(s, bytes) else s for s in sp["imgname"]]

    def key(n, c):
        return n.split("/")[-2], n.split("/")[-1], tuple(np.round(c, 0))

    index = {key(n, c): i for i, (n, c) in enumerate(zip(tk_names, tk_center))}
    mapping = np.array([index[key(n, c)] for n, c in zip(sp_names, sp["center"])])
    assert len(set(mapping.tolist())) == len(mapping), "SPIN 재배열이 일대일이 아니다"
    return mapping


def load_backbone(bb, names, center, spin_reorder):
    p, g, _ = H.load_backbone(H.BACKBONES[bb], names)
    if bb == "SPIN" and spin_reorder:
        m = spin_order(names, center)
        p2, g2 = np.empty_like(p), np.empty_like(g)
        p2[m], g2[m] = p, g
        p, g = p2, g2
        print("  SPIN: SPIN npz 순서 → TokenHMR npz 순서로 재배열", flush=True)
    return p, g


# ---------------------------------------------------------------- 표집
def seq_unit_sample(pool_seqs, seq_idx, target, rng):
    """다른 시퀀스를 무작위 순서로 쌓아 target 프레임(마지막 시퀀스는 잘라냄). exp774 와 같다."""
    parts, total = [], 0
    for i in rng.permutation(len(pool_seqs)):
        idx = seq_idx[pool_seqs[i]]
        if total + len(idx) >= target:
            parts.append(rng.choice(idx, target - total, replace=False))
            return np.concatenate(parts)
        parts.append(idx)
        total += len(idx)
    return np.concatenate(parts)


def frame_uniform_sample(pool_rows, target, rng):
    """풀 전체에서 프레임 균등 비복원 표집 — 시퀀스 다양성 보존."""
    if target >= len(pool_rows):
        return pool_rows
    return np.sort(rng.choice(pool_rows, target, replace=False))


# ---------------------------------------------------------------- 요약
def boot_ci(flat, rng, n_boot):
    boots = np.array([rng.choice(flat, flat.size).mean() for _ in range(n_boot)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def summarize_share(sub_arr, match_arr, base_rows, rng, n_boot):
    """(시드 i, 추첨 j) 마다 몫 = 1 − pct(sub_i)/pct(match_ij). 시드는 짝지어 쓴다."""
    S, D = match_arr.shape[:2]
    sub_pct = np.array([pct(base_rows, sub_arr[i]) for i in range(S)])
    m_pct = np.array([[pct(base_rows, match_arr[i, j]) for j in range(D)] for i in range(S)])
    flat = (1.0 - sub_pct[:, None] / m_pct).ravel()
    return {"share_mean": float(flat.mean()), "share_boot95": boot_ci(flat, rng, n_boot),
            "share_indiv_range95": [float(np.percentile(flat, 2.5)), float(np.percentile(flat, 97.5))],
            "share_min_max": [float(flat.min()), float(flat.max())],
            "sub_pct_mean": float(sub_pct.mean()), "sub_pct_sd": float(sub_pct.std(ddof=1)),
            "match_pct_mean": float(m_pct.mean()), "match_pct_sd": float(m_pct.std(ddof=1))}


def summarize_raw(sub_arr, ref_arr, base_rows, rng, n_boot):
    """크기 맞춤 없는 몫: 시드 i 끼리 짝지어 1 − pct(sub_i)/pct(ref_i)."""
    S = sub_arr.shape[0]
    sp = np.array([pct(base_rows, sub_arr[i]) for i in range(S)])
    rp = np.array([pct(base_rows, ref_arr[i]) for i in range(S)])
    share = 1.0 - sp / rp
    return {"share_mean": float(share.mean()), "share_boot95": boot_ci(share, rng, n_boot),
            "share_min_max": [float(share.min()), float(share.max())],
            "sub_pct_mean": float(sp.mean()), "sub_pct_sd": float(sp.std(ddof=1)),
            "ref_pct_mean": float(rp.mean()), "ref_pct_sd": float(rp.std(ddof=1))}


# ---------------------------------------------------------------- 본체
def protocols_a(X, Y, base, seq, subj, S, E, bb, t0):
    n = len(base)
    seqs = sorted(set(seq.tolist()))
    seq_idx = {s: np.nonzero(seq == s)[0] for s in seqs}
    all_rows = np.arange(n)
    e_seq = np.full((S, n), np.nan)
    for i in range(S):
        for s in seqs:
            m = train(X, Y, np.setdiff1d(all_rows, seq_idx[s]), seed=i, epochs=E)
            e_seq[i, seq_idx[s]] = evaluate(m, X, Y, seq_idx[s])
        print(f"  [a] seq-LOO seed {i}: {pct(base, e_seq[i]):+.3f}%  @{bb}_seqloo_s{i}={pct(base, e_seq[i]):.4f}"
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
    e_row, e_whole = np.full((S, n), np.nan), np.full((S, n), np.nan)
    info, per = {}, {}
    for u in sorted(set(subj.tolist())):
        u_rows = np.nonzero(subj == u)[0]
        u_seqs = sorted(set(seq[u_rows].tolist()))
        tr_row = np.setdiff1d(all_rows, u_rows)
        tr_whole = np.nonzero(~np.isin(seq, u_seqs))[0]
        co = np.nonzero(np.isin(seq, u_seqs) & (subj != u))[0]
        co_by = {int(v): int((subj[co] == v).sum()) for v in sorted(set(subj[co].tolist()))}
        info[u] = dict(u_rows=u_rows, u_seqs=u_seqs, tr_row=tr_row, tr_whole=tr_whole)
        for i in range(S):
            e_row[i, u_rows] = evaluate(train(X, Y, tr_row, seed=i, epochs=E), X, Y, u_rows)
            if len(tr_whole):
                e_whole[i, u_rows] = evaluate(train(X, Y, tr_whole, seed=i, epochs=E), X, Y, u_rows)
        r = {"n_seq": len(u_seqs), "n_frames": int(len(u_rows)),
             "train_rowloo": int(len(tr_row)), "train_wholeseq": int(len(tr_whole)),
             "coactor_rows_kept_in_rowloo": int(len(co)), "coactor_rows_by_subject": co_by,
             "seqs": u_seqs,
             "seq_pct": [pct(base[u_rows], e_seq[i, u_rows]) for i in range(S)],
             "row_pct": [pct(base[u_rows], e_row[i, u_rows]) for i in range(S)],
             "whole_pct": ([pct(base[u_rows], e_whole[i, u_rows]) for i in range(S)]
                           if len(tr_whole) else None)}
        per[f"subject_{u}"] = r
        print(f"  [a] subject_{u}: seq {len(u_seqs):2d} frames {len(u_rows):6d}  train row {len(tr_row)} "
              f"whole {len(tr_whole)}  coactor kept {len(co)} {co_by}  seqLOO {np.mean(r['seq_pct']):+.2f}"
              f"±{np.std(r['seq_pct'], ddof=1):.2f}  row {np.mean(r['row_pct']):+.2f}±{np.std(r['row_pct'], ddof=1):.2f}"
              f"  whole {'-' if r['whole_pct'] is None else f'{np.mean(r[chr(119)+chr(104)+chr(111)+chr(108)+chr(101)+chr(95)+chr(112)+chr(99)+chr(116)]):+.2f}'}"
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return e_seq, e_row, e_whole, info, per, seqs, seq_idx


def matched_runs(tag, X, Y, info, seqs, seq_idx, n, S, D, E, t0):
    all_rows = np.arange(n)
    arr = np.full((S, D, n), np.nan)
    for u, iu in info.items():
        target = len(iu["tr_row"]) if tag in ("b", "c") else len(iu["tr_whole"])
        if target == 0:
            continue
        for s in iu["u_seqs"]:
            test = np.intersect1d(seq_idx[s], iu["u_rows"])
            pool_seqs = [q for q in seqs if q != s]
            pool_rows = np.setdiff1d(all_rows, seq_idx[s])
            for i in range(S):
                for j in range(D):
                    rng = np.random.default_rng([i, j, int(u), seqs.index(s), ord(tag)])
                    tr = (frame_uniform_sample(pool_rows, target, rng) if tag == "c"
                          else seq_unit_sample(pool_seqs, seq_idx, target, rng))
                    arr[i, j, test] = evaluate(train(X, Y, tr, seed=i, epochs=E), X, Y, test)
        print(f"  [{tag}] subject_{u} 완료 (target {target})  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return arr


def run_backbone(bb, names, center, seq, subj, prm, t0):
    p, g = load_backbone(bb, names, center, spin_reorder=bool(prm.get("spin_reorder", 1)))
    n = len(p)
    seq, subj = seq[:n], subj[:n]
    X = torch.tensor(p.reshape(n, -1), dtype=torch.float32, device=DEV)
    Y = torch.tensor(g.reshape(n, -1), dtype=torch.float32, device=DEV)
    base = pa_torch(X, Y).cpu().numpy()
    print(f"\n===== {bb}  n={n}  baseline {base.mean():.4f} mm  "
          f"(H 규약 500행 {H.compute_pa_mpjpe(p[:500], g[:500]).mean():.4f} vs torch {base[:500].mean():.4f})",
          flush=True)
    S, D, E = int(prm["n_seeds"]), int(prm["n_draws"]), int(prm["epochs"])
    parts = set(prm["parts"].split(","))
    e_seq, e_row, e_whole, info, per, seqs, seq_idx = protocols_a(X, Y, base, seq, subj, S, E, bb, t0)
    res = {"baseline_mm": float(base.mean()), "per_subject": per, "aggregate": {}}
    matched = {tag: matched_runs(tag, X, Y, info, seqs, seq_idx, n, S, D, E, t0)
               for tag, _ in SPECS if tag in parts}

    rng = np.random.default_rng(12345)
    nb = int(prm["n_boot"])
    groups = {f"subject_{u}": info[u]["u_rows"] for u in info}
    groups["aggregate"] = np.arange(n)
    for gname, rows in groups.items():
        out = {"raw_row": summarize_raw(e_row[:, rows], e_seq[:, rows], base[rows], rng, nb)}
        ok_whole = rows[~np.isnan(e_whole[0, rows])]
        if len(ok_whole):
            out["raw_whole"] = summarize_raw(e_whole[:, ok_whole], e_seq[:, ok_whole], base[ok_whole], rng, nb)
        for tag, _ in SPECS:
            if tag not in matched:
                continue
            sub, rr = (e_whole, ok_whole) if tag == "d" else (e_row, rows)
            if len(rr) == 0 or np.isnan(matched[tag][:, :, rr]).any():
                continue
            out[f"matched_{tag}"] = summarize_share(sub[:, rr], matched[tag][:, :, rr], base[rr], rng, nb)
        tgt = res["aggregate"] if gname == "aggregate" else res["per_subject"][gname]
        tgt["shares"] = out
        line = "  ".join(f"{k} {v['share_mean']:+.3f} [{v['share_boot95'][0]:+.3f},{v['share_boot95'][1]:+.3f}]"
                         for k, v in out.items())
        print(f"  [몫] {bb} {gname}: {line}", flush=True)
        if gname == "aggregate":
            for k, v in out.items():
                print(f"  @{bb}_share_{k}={v['share_mean']:.4f}", flush=True)
            print(f"  @{bb}_seq_pct={np.mean([pct(base, e_seq[i]) for i in range(S)]):.4f} "
                  f"@{bb}_row_pct={np.mean([pct(base, e_row[i]) for i in range(S)]):.4f}", flush=True)
    return res


def main() -> int:
    prm = {**DEFAULTS, **json.loads(os.environ.get("CQ_PARAMS") or "{}")}
    print(f"device={DEV} params={prm}", flush=True)
    names, center, seq, subj = load_rows()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = prm.get("tag", "full")
    out = {"params": prm}
    for bb in prm["backbones"].split(","):
        out[bb] = run_backbone(bb, names, center, seq, subj, prm, t0)
        (OUT / f"results_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n저장 {OUT}/results_{tag}.json  총 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
