"""exp774: '개인화 몫'이 정말 개인화인지 가르는 대조실험 — B 논문 자체리뷰 M2/M4.

리뷰의 반론 세 가지를 각각 실험 하나로 가른다.
  (1) 학습 크기 교란: subject-LOO 는 큰 배우를 빼면 학습 프레임이 크게 준다. seq-LOO 를 같은
      학습 프레임 수로 줄여서(시퀀스 단위 무작위 표집, 3회) 다시 잰다.
  (2) 같은 시퀀스 누수: 행 단위 subject-LOO 는 그 배우가 나온 시퀀스의 다른 사람 행을 학습에
      남긴다. 그 배우가 나온 시퀀스를 통째로 빼는 판본을 함께 잰다.
  (3) 신원 대 유사 시퀀스: 최근접 선택을 '다른 배우만 나오는 시퀀스'로 제한했을 때와 비교하고,
      제한 없는 최근접이 같은 배우·같은 장면·형제 시퀀스(_00/_01)를 얼마나 고르는지 센다.
  (4) 배우별 표, (5) betas 신원 대 3DPW 원본 시퀀스 파일의 배우(성별·옷 템플릿) 대조.

모든 비교는 같은 평가 행 위에서 한다(배우 u 의 행). 모델·학습·표준 Procrustes 는
exp803b 모듈을 절대경로로 import 해서 한 글자도 바꾸지 않는다.
"""

import os
import glob
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402

import torch  # noqa: E402

OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp774"))
SEQ_PKL_GLOB = os.path.join(os.environ.get("THREEDPW_SEQ_DIR", "data/3DPW/sequenceFiles"), "test", "*.pkl")
N_SIZE_DRAWS = 3
N_RANDOM_SEL = 3
K_MAX = 5
EPOCHS = 10
SEED = 0
BETA_DECIMALS = 3
N_BETAS = 10


def pct(base, after):
    return float(100.0 * (base.mean() - after.mean()) / base.mean())


def fit_eval(preds, gts, train_idx, test_idx, dev):
    m = H.train_mlp(preds[train_idx], gts[train_idx], dev, EPOCHS, seed=SEED)
    return H.eval_mlp(m, preds[test_idx], gts[test_idx], dev)


def load_rows():
    mocap = np.load(H.MOCAP, allow_pickle=True)
    names = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq = np.array([Path(n).parts[-2] for n in names])
    lut, subj = {}, []
    for b in mocap["betas"]:
        subj.append(lut.setdefault(tuple(np.round(b[:N_BETAS], BETA_DECIMALS)), len(lut)))
    return names, seq, np.array(subj), lut


def size_matched_train(pool_seqs, seq_idx, target, rng):
    """다른 시퀀스를 무작위 순서로 쌓아 target 프레임에 맞춘다(마지막 시퀀스는 잘라낸다)."""
    order = rng.permutation(len(pool_seqs))
    parts, total = [], 0
    for i in order:
        idx = seq_idx[pool_seqs[i]]
        if total + len(idx) >= target:
            parts.append(rng.choice(idx, target - total, replace=False))
            total = target
            break
        parts.append(idx)
        total += len(idx)
    return np.concatenate(parts), total


def part_protocols(preds, gts, seq, subj, dev, bb, t0):
    """(1)(2)(4): 배우 u 의 행 위에서 네 프로토콜을 잰다."""
    n = len(preds)
    seqs = sorted(set(seq.tolist()))
    seq_idx = {s: np.nonzero(seq == s)[0] for s in seqs}
    base = H.compute_pa_mpjpe(preds, gts)
    all_rows = np.arange(n)

    e_seq = np.full(n, np.nan)
    for s in seqs:
        e_seq[seq_idx[s]] = fit_eval(preds, gts, np.setdiff1d(all_rows, seq_idx[s]),
                                     seq_idx[s], dev)
    print(f"  seq-LOO 완료 ({(time.time()-t0)/60:.1f} min)", flush=True)

    rng = np.random.default_rng(SEED)
    e_row, e_whole = np.full(n, np.nan), np.full(n, np.nan)
    e_sm = np.full((N_SIZE_DRAWS, n), np.nan)
    per_subject = {}
    for u in sorted(set(subj.tolist())):
        u_rows = np.nonzero(subj == u)[0]
        u_seqs = sorted(set(seq[u_rows].tolist()))
        # 행 단위 subject-LOO (논문 Table 3 방식)
        tr_row = np.setdiff1d(all_rows, u_rows)
        e_row[u_rows] = fit_eval(preds, gts, tr_row, u_rows, dev)
        # 시퀀스 통째 제외
        tr_whole = np.nonzero(~np.isin(seq, u_seqs))[0]
        if len(tr_whole):
            e_whole[u_rows] = fit_eval(preds, gts, tr_whole, u_rows, dev)
        # 크기 맞춘 seq-LOO: u 가 나온 시퀀스 s 마다, s 밖에서 len(tr_row) 프레임만
        target = len(tr_row)
        capped = []
        for s in u_seqs:
            test = np.intersect1d(seq_idx[s], u_rows)
            pool = [q for q in seqs if q != s]
            for d in range(N_SIZE_DRAWS):
                tr, got = size_matched_train(pool, seq_idx, target, rng)
                if got < target:
                    capped.append(s)
                e_sm[d, test] = fit_eval(preds, gts, tr, test, dev)
        r = {"n_seq": len(u_seqs), "n_frames": int(len(u_rows)),
             "train_frames_rowloo": int(len(tr_row)),
             "train_frames_wholeseq": int(len(tr_whole)),
             "baseline": float(base[u_rows].mean()),
             "seq_pct": pct(base[u_rows], e_seq[u_rows]),
             "sizematched_pct": pct(base[u_rows], np.nanmean(e_sm[:, u_rows], 0)),
             "sizematched_pct_draws": [pct(base[u_rows], e_sm[d, u_rows])
                                       for d in range(N_SIZE_DRAWS)],
             "subj_row_pct": pct(base[u_rows], e_row[u_rows]),
             "subj_whole_pct": (pct(base[u_rows], e_whole[u_rows])
                                if len(tr_whole) else None),
             "sizematch_capped_seqs": sorted(set(capped))}
        per_subject[f"subject_{u}"] = r
        print(f"  subject_{u}: seq {r['n_seq']:2d}  frames {r['n_frames']:6d}  "
              f"seqLOO {r['seq_pct']:+6.2f}  size-matched {r['sizematched_pct']:+6.2f}  "
              f"subj-row {r['subj_row_pct']:+6.2f}  subj-whole "
              f"{r['subj_whole_pct'] if r['subj_whole_pct'] is None else round(r['subj_whole_pct'], 2)}"
              f"  (train row {len(tr_row)} / whole {len(tr_whole)})  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    ok_whole = ~np.isnan(e_whole)
    agg = {
        "seq_pct": pct(base, e_seq),
        "sizematched_pct": pct(base, np.nanmean(e_sm, 0)),
        "sizematched_pct_draws": [pct(base, e_sm[d]) for d in range(N_SIZE_DRAWS)],
        "subj_row_pct": pct(base, e_row),
        "subj_whole_pct_on_covered_rows": pct(base[ok_whole], e_whole[ok_whole]),
        "subj_whole_covered_frac": float(ok_whole.mean()),
    }
    agg["share_raw"] = (agg["seq_pct"] - agg["subj_row_pct"]) / agg["seq_pct"]
    agg["share_sizematched"] = ((agg["sizematched_pct"] - agg["subj_row_pct"])
                                / agg["sizematched_pct"])
    print(f"  [{bb}] seqLOO {agg['seq_pct']:+.2f}  size-matched {agg['sizematched_pct']:+.2f} "
          f"{[round(x, 2) for x in agg['sizematched_pct_draws']]}  subj-row "
          f"{agg['subj_row_pct']:+.2f}  subj-whole {agg['subj_whole_pct_on_covered_rows']:+.2f} "
          f"(covered {agg['subj_whole_covered_frac']:.3f})", flush=True)
    print(f"  @{bb}_seq_pct={agg['seq_pct']:.4f} @{bb}_sizematched_pct={agg['sizematched_pct']:.4f} "
          f"@{bb}_subj_row_pct={agg['subj_row_pct']:.4f} "
          f"@{bb}_subj_whole_pct={agg['subj_whole_pct_on_covered_rows']:.4f} "
          f"@{bb}_share_raw={agg['share_raw']:.4f} "
          f"@{bb}_share_sizematched={agg['share_sizematched']:.4f}", flush=True)
    return {"aggregate": agg, "per_subject": per_subject}


def part_selection(preds, gts, seq, subj, dev, bb, t0):
    """(3): 최근접(제한 없음) / 최근접(다른 배우만) / 무작위, k=1..5, seq-LOO 대상."""
    seqs = sorted(set(seq.tolist()))
    seq_idx = {s: np.nonzero(seq == s)[0] for s in seqs}
    seq_subj = {s: set(subj[seq_idx[s]].tolist()) for s in seqs}
    mean_pred = {s: preds[seq_idx[s]].reshape(len(seq_idx[s]), -1).mean(0) for s in seqs}
    base = H.compute_pa_mpjpe(preds, gts)
    scene = {s: s.split("_")[0] for s in seqs}
    stem = {s: s.rsplit("_", 1)[0] for s in seqs}
    rng = np.random.default_rng(SEED + 1)

    res = {}
    overlap = {"same_subject": [], "same_scene": [], "sibling": []}
    for k in range(1, K_MAX + 1):
        near_e, restr_e, rand_e, base_e, restr_near_e, restr_base_e = [], [], [], [], [], []
        n_restr = 0
        for s in seqs:
            cands = [q for q in seqs if q != s]
            d = np.array([np.linalg.norm(mean_pred[s] - mean_pred[q]) for q in cands])
            order = [cands[i] for i in np.argsort(d)]
            near = order[:k]
            if k == K_MAX:
                for q in near:
                    overlap["same_subject"].append(bool(seq_subj[q] & seq_subj[s]))
                    overlap["same_scene"].append(scene[q] == scene[s])
                    overlap["sibling"].append(stem[q] == stem[s])
            te = seq_idx[s]
            en = fit_eval(preds, gts, np.concatenate([seq_idx[q] for q in near]), te, dev)
            near_e.append(en)
            base_e.append(base[te])
            er = []
            for _ in range(N_RANDOM_SEL):
                pick = rng.choice(cands, k, replace=False)
                er.append(fit_eval(preds, gts, np.concatenate([seq_idx[q] for q in pick]),
                                   te, dev))
            rand_e.append(np.mean(er, 0))
            other = [q for q in order if not (seq_subj[q] & seq_subj[s])]
            if len(other) >= k:
                n_restr += 1
                restr_e.append(fit_eval(preds, gts,
                                        np.concatenate([seq_idx[q] for q in other[:k]]),
                                        te, dev))
                restr_near_e.append(en)
                restr_base_e.append(base[te])
        B = np.concatenate(base_e)
        row = {"nearest_pct": pct(B, np.concatenate(near_e)),
               "random_pct": pct(B, np.concatenate(rand_e)),
               "n_targets_restricted": n_restr}
        if n_restr:
            RB = np.concatenate(restr_base_e)
            row["restricted_subset_nearest_pct"] = pct(RB, np.concatenate(restr_near_e))
            row["restricted_diffsubject_pct"] = pct(RB, np.concatenate(restr_e))
        res[f"k{k}"] = row
        print(f"  [{bb}] k={k}: nearest {row['nearest_pct']:+.2f}  random {row['random_pct']:+.2f}  "
              f"| 다른배우만 가능 대상 {n_restr}/24: nearest {row.get('restricted_subset_nearest_pct', float('nan')):+.2f}"
              f"  diff-subject {row.get('restricted_diffsubject_pct', float('nan')):+.2f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    res["nearest_k5_overlap"] = {key: float(np.mean(v)) for key, v in overlap.items()}
    print(f"  [{bb}] 최근접 k=5 고른 시퀀스 중 같은 배우 {res['nearest_k5_overlap']['same_subject']:.3f}  "
          f"같은 장면 {res['nearest_k5_overlap']['same_scene']:.3f}  "
          f"형제(_00/_01) {res['nearest_k5_overlap']['sibling']:.3f}", flush=True)
    return res


def part_identity(seq, subj, lut):
    """(5): 3DPW 원본 시퀀스 파일의 배우(betas·성별·옷 템플릿)와 betas 신원 대조."""
    inv = {v: k for k, v in lut.items()}
    table = []
    for f in sorted(glob.glob(SEQ_PKL_GLOB)):
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        name = Path(f).stem
        for i, b in enumerate(d["betas"]):
            key = tuple(np.round(np.asarray(b)[:N_BETAS], BETA_DECIMALS))
            tmpl = d.get("v_template_clothed")
            th = (hashlib.sha1(np.asarray(tmpl[i]).round(4).tobytes()).hexdigest()[:8]
                  if tmpl is not None else None)
            table.append({"seq": name, "actor": i, "gender": d["genders"][i],
                          "subject_id": lut.get(key), "template": th})
    by_subj = {}
    for r in table:
        by_subj.setdefault(r["subject_id"], []).append(r)
    summary = {}
    for u, rows in by_subj.items():
        summary[str(u)] = {"genders": sorted(set(r["gender"] for r in rows)),
                           "n_templates": len(set(r["template"] for r in rows)),
                           "n_actor_slots": len(rows),
                           "seqs": sorted(set(r["seq"] for r in rows))}
        print(f"  betas 신원 {u}: 성별 {summary[str(u)]['genders']}  옷 템플릿 "
              f"{summary[str(u)]['n_templates']}종  배우 슬롯 {len(rows)}", flush=True)
    unmatched = [r for r in table if r["subject_id"] is None]
    print(f"  원본 배우 슬롯 {len(table)}개 중 betas 신원에 매칭 안 된 것 {len(unmatched)}개", flush=True)
    # 한 시퀀스 안의 두 배우가 같은 신원으로 합쳐졌는지
    merged = [s for s in set(r["seq"] for r in table)
              if len([r for r in table if r["seq"] == s]) >= 2
              and len(set(r["subject_id"] for r in table if r["seq"] == s)) < 2]
    print(f"  한 시퀀스의 두 배우가 같은 신원으로 합쳐진 시퀀스: {merged}", flush=True)
    print(f"  npz 신원 수 {len(inv)}", flush=True)
    return {"slots": table, "by_subject": summary, "unmatched": len(unmatched),
            "merged_two_person_seqs": merged}


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, seq_all, subj_all, lut = load_rows()
    t0 = time.time()
    print(f"device={dev}  행 {len(names)}  시퀀스 {len(set(seq_all.tolist()))}  "
          f"betas 신원 {len(lut)}", flush=True)
    out = {"identity": part_identity(seq_all, subj_all, lut)}
    for bb, path in H.BACKBONES.items():
        preds, gts, sn = H.load_backbone(path, list(seq_all))
        n = len(preds)
        seq, subj = seq_all[:n], subj_all[:n]
        assert (sn == seq).all()
        print(f"\n===== {bb}  행 {n}", flush=True)
        out[bb] = {"protocols": part_protocols(preds, gts, seq, subj, dev, bb, t0),
                   "selection": part_selection(preds, gts, seq, subj, dev, bb, t0)}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n저장: {OUT/'results.json'}  (총 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
