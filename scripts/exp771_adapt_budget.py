"""라벨 예산을 SMPL 헤드 적응에 쓸 때의 곡선을 피험자 보류로 측정한다.

**왜 피험자 보류인가**: 논문 B §ftcontrol 은 헤드 적응을 점 하나(k=5, 시퀀스 LOO)로 재서
−28.3% 를 얻었는데, 같은 절에서 그 이득의 대부분이 개인화임을 측정했다(피험자 포함 fold
29.5% 대 미포함 10.4%, 낯선 하나는 −9.0%). 시퀀스 분할로 적응의 천장을 재면 개인화를
측정해놓고 일반화라 부르게 된다. 그래서 D' 의 모든 측정은 피험자 보류로 한다.

**피험자 정의**: 주석 npz 의 SMPL betas 를 소수 셋째 자리로 반올림한 값이 곧 배우 신원이다
(exp760_subject_disjoint 와 동일 규약 — 두 표가 같은 분할 위에 서야 비교가 된다).

**선택 규칙**: refiner 와 같다 — 예측 포즈 평균의 L2 거리. 다만 질의가 시퀀스 하나가 아니라
보류된 피험자 전체의 평균 예측이다. 후보는 그 피험자의 프레임이 하나도 없는 시퀀스뿐이다.

**적응 지점**: 지금은 SMPL 헤드만. 마지막 N 블록은 절단 지점별 캐시가 더 필요하고 디스크
여유가 80GB 남짓이라(캐시 한 점 17GB) 설계를 먼저 고정한 뒤 붙인다.

캐시(exp770 feats.npy 17GB fp16)를 재사용하므로 특징 추출은 없다.
"""

import argparse
import copy
import functools
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6+ 의 weights_only 기본값이 이 체크포인트를 막는다
_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

sys.path.insert(0, os.environ.get("HMR2_ROOT", "third_party/4D-Humans"))
os.chdir(os.environ.get("HMR2_ROOT", "third_party/4D-Humans"))          # hmr2 설정이 상대경로를 쓴다

CACHE = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp770"))
ANN = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
REFINER_PKL = os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl")
OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp771"))


def similarity_transform(S1, S2):
    """표준 Procrustes — 최소제곱 스케일 포함. exp770·exp760 과 동일."""
    B = S1.shape[0]
    S1, S2 = S1.permute(0, 2, 1), S2.permute(0, 2, 1)
    mu1, mu2 = S1.mean(2, keepdim=True), S2.mean(2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, _, Vh = torch.linalg.svd(K)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    R = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), Z), U.permute(0, 2, 1))
    tr = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    sc = (tr / var1).unsqueeze(-1).unsqueeze(-1)
    return (sc * torch.bmm(R, S1) + (mu2 - sc * torch.bmm(R, mu1))).permute(0, 2, 1)


def pa_mpjpe(p, g):
    return torch.sqrt(((similarity_transform(p, g) - g) ** 2).sum(-1)).mean(-1) * 1000


def build_folds():
    """시퀀스·피험자 색인과 후보 풀을 만든다. 학습 없이 구조만 본다."""
    meta = json.loads((CACHE / "meta.json").read_text())
    n = meta["n"]
    ann = np.load(ANN, allow_pickle=True)
    imgnames, betas = ann["imgname"][:n], ann["betas"][:n]

    lut, subj = {}, []
    for b in betas:                                   # betas 가 곧 배우 신원
        subj.append(lut.setdefault(tuple(np.round(b, 3)), len(lut)))
    subj = np.array(subj)

    seq_of = np.array([str(s).split("/")[1] for s in imgnames])
    seqs = sorted(set(seq_of.tolist()))
    seq_idx = {s: np.nonzero(seq_of == s)[0] for s in seqs}
    subjects = sorted(set(subj.tolist()))

    folds = {}
    for u in subjects:
        te = np.nonzero(subj == u)[0]
        # 보류 피험자의 프레임이 하나도 없는 시퀀스만 후보다
        cands = [s for s in seqs if not (subj[seq_idx[s]] == u).any()]
        folds[u] = {"test": te, "cands": cands}
    return meta, subj, seq_of, seq_idx, seqs, subjects, folds


def report_folds(folds, seq_idx, subj):
    print(f"피험자 {len(folds)}명", flush=True)
    pool_min = 10 ** 9
    for u, f in folds.items():
        n_cand_frames = sum(len(seq_idx[s]) for s in f["cands"])
        pool_min = min(pool_min, len(f["cands"]))
        print(f"  subject_{u}: 테스트 프레임 {len(f['test']):6d}  "
              f"후보 시퀀스 {len(f['cands']):3d}개 ({n_cand_frames:6d} 프레임)", flush=True)
    print(f"\n최소 후보 시퀀스 수 = {pool_min} → k 는 여기까지만 올릴 수 있다", flush=True)
    print(f"@pool_min={pool_min} @n_folds={len(folds)}", flush=True)
    return pool_min


def head_forward(model, feats):
    params, _, _ = model.smpl_head(feats)
    B = feats.shape[0]
    params["global_orient"] = params["global_orient"].reshape(B, -1, 3, 3)
    params["body_pose"] = params["body_pose"].reshape(B, -1, 3, 3)
    params["betas"] = params["betas"].reshape(B, -1)
    out = model.smpl(**{k: v.float() for k, v in params.items()}, pose2rot=False)
    return out.joints.reshape(B, -1, 3)


def evaluate(model, state, feats, gt44, idx, pelvis, kl, bs=32):
    model.smpl_head.load_state_dict(state)
    model.smpl_head.eval()
    vals = []
    with torch.no_grad():
        for b0 in range(0, len(idx), bs):
            bi = idx[b0:b0 + bs]
            f = torch.from_numpy(np.asarray(feats[bi])).float().cuda()
            g = torch.from_numpy(gt44[bi]).float().cuda()
            g3 = (g - g[:, [pelvis]])[:, kl]
            p = head_forward(model, f)
            p = (p - p[:, [pelvis]])[:, kl]
            vals += pa_mpjpe(p, g3).cpu().tolist()
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-only", action="store_true")
    ap.add_argument("--k", default="1,2,3,5")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta, subj, seq_of, seq_idx, seqs, subjects, folds = build_folds()
    pool_min = report_folds(folds, seq_idx, subj)
    if a.folds_only:
        return 0

    ks = [int(x) for x in a.k.split(",") if int(x) <= pool_min]
    print(f"\n예산 스윕 k={ks} (pool_min={pool_min} 으로 잘림)", flush=True)

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    pelvis, kl = meta["pelvis_ind"], meta["keypoint_list"]
    feats = np.load(CACHE / "feats.npy", mmap_mode="r")
    gt44 = np.load(CACHE / "gt44.npy")

    with open(REFINER_PKL, "rb") as f:
        samp = pickle.load(f)
    pred_eval = np.stack([x["pred_joints_3d_eval"] for x in samp])[:meta["n"]]
    seq_mean = {s: pred_eval[seq_idx[s]].reshape(len(seq_idx[s]), -1).mean(0) for s in seqs}

    from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT
    from hmr2.configs import CACHE_DIR_4DHUMANS
    download_models(CACHE_DIR_4DHUMANS)
    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.cuda().eval()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    init_head = copy.deepcopy(model.smpl_head.state_dict())

    # 미적응 baseline 은 k 와 무관하다. subject_1 은 테스트가 19,930 프레임이라
    # k 마다 다시 재면 그것만 8만 회 순전파가 된다 — fold 당 한 번만 재고 재사용한다.
    base_cache = {}
    for u, f in folds.items():
        base_cache[u] = evaluate(model, init_head, feats, gt44, f["test"], pelvis, kl)
        print(f"  baseline subject_{u}: {np.mean(base_cache[u]):.4f} "
              f"(n={len(f['test'])})", flush=True)

    results, t0 = {}, time.time()
    for k in ks:
        before_all, after_all, per_fold = [], [], {}
        for u, f in folds.items():
            te = f["test"]
            # 질의는 보류 피험자 전체의 평균 예측
            q = pred_eval[te].reshape(len(te), -1).mean(0)
            d = np.array([np.linalg.norm(q - seq_mean[s]) for s in f["cands"]])
            pick = [f["cands"][i] for i in np.argsort(d)[:k]]
            tr = np.concatenate([seq_idx[s] for s in pick])

            model.smpl_head.load_state_dict(init_head)
            model.smpl_head.train()
            opt = torch.optim.AdamW(model.smpl_head.parameters(), lr=a.lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
            for _ in range(a.epochs):
                perm = np.random.permutation(len(tr))
                for b0 in range(0, len(perm), 16):
                    bi = np.sort(tr[perm[b0:b0 + 16]])
                    fb = torch.from_numpy(np.asarray(feats[bi])).float().cuda()
                    gb = torch.from_numpy(gt44[bi]).float().cuda()
                    p3 = head_forward(model, fb)
                    p3 = p3 - p3[:, [pelvis]]
                    g3 = gb - gb[:, [pelvis]]
                    loss = torch.nn.functional.l1_loss(p3[:, kl], g3[:, kl])
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.smpl_head.parameters(), 1.0)
                    opt.step()
                sch.step()

            ft_head = copy.deepcopy(model.smpl_head.state_dict())
            after = evaluate(model, ft_head, feats, gt44, te, pelvis, kl)
            before = base_cache[u]
            per_fold[f"subject_{u}"] = {
                "n": len(te), "picked": pick,
                "before": float(np.mean(before)), "after": float(np.mean(after))}
            before_all += before
            after_all += after
            print(f"  k={k} subject_{u}: {np.mean(before):6.2f} -> {np.mean(after):6.2f} "
                  f"({100*(np.mean(before)-np.mean(after))/np.mean(before):+.2f}%)  "
                  f"picked={pick}  ({(time.time()-t0)/60:.1f} min)", flush=True)

        b, aft = float(np.mean(before_all)), float(np.mean(after_all))
        imp = 100 * (b - aft) / b
        # 프레임 가중은 사실상 2인 실험이다 — subject_0·1 이 전체 프레임의 93%.
        # 피험자 가중(fold 평균의 평균)을 함께 낸다. 둘 중 하나만 보면 오독한다.
        fb = float(np.mean([v["before"] for v in per_fold.values()]))
        fa = float(np.mean([v["after"] for v in per_fold.values()]))
        simp = 100 * (fb - fa) / fb
        results[str(k)] = {"before_pa": b, "after_pa": aft, "imp_pct": imp,
                           "subjw_before_pa": fb, "subjw_after_pa": fa,
                           "subjw_imp_pct": simp, "per_fold": per_fold}
        print(f"k={k}  프레임가중 {b:.4f} -> {aft:.4f} ({imp:+.2f}%)  |  "
              f"피험자가중 {fb:.4f} -> {fa:.4f} ({simp:+.2f}%)", flush=True)
        print(f"  @before_pa={b:.4f} @after_pa={aft:.4f} @imp_pct={imp:.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"pool_min": pool_min, "n_folds": len(folds), "epochs": a.epochs,
         "lr": a.lr, "k": results}, indent=2))
    print(f"\n저장: {OUT/'results.json'}  (총 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
