"""exp775 결과 파일들을 한 표로 모은다(개선율 % 와 몫, 구간). 학습 없음."""

import os
import json
from pathlib import Path

import numpy as np

D = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp775"))
for bb in ("TokenHMR", "4DHumans", "SPIN"):
    files = {t: D / f"results_{bb}_{t}.json" for t in "bcd"}
    data = {t: json.loads(f.read_text())[bb] for t, f in files.items() if f.exists()}
    any_t = next(iter(data.values()))
    print(f"\n===== {bb}  baseline {any_t['baseline_mm']:.4f}", flush=True)
    groups = ["aggregate"] + [f"subject_{u}" for u in range(5)]
    for g in groups:
        src = any_t["aggregate"] if g == "aggregate" else any_t["per_subject"][g]
        sh = src["shares"]
        rr = sh["raw_row"]
        line = (f"  {g:10s} seqLOO {rr['ref_pct_mean']:6.2f}±{rr['ref_pct_sd']:.2f}  "
                f"row {rr['sub_pct_mean']:6.2f}±{rr['sub_pct_sd']:.2f}  ")
        if "raw_whole" in sh:
            line += f"whole {sh['raw_whole']['sub_pct_mean']:6.2f}±{sh['raw_whole']['sub_pct_sd']:.2f}  "
        line += (f"| 몫 raw_row {rr['share_mean']:+.3f} [{rr['share_boot95'][0]:+.3f},{rr['share_boot95'][1]:+.3f}]")
        if "raw_whole" in sh:
            w = sh["raw_whole"]
            line += f"  raw_whole {w['share_mean']:+.3f} [{w['share_boot95'][0]:+.3f},{w['share_boot95'][1]:+.3f}]"
        print(line, flush=True)
        for t, dd in data.items():
            s2 = (dd["aggregate"] if g == "aggregate" else dd["per_subject"][g])["shares"].get(f"matched_{t}")
            if s2 is None:
                continue
            print(f"      matched_{t}: seqLOO(matched) {s2['match_pct_mean']:6.2f}±{s2['match_pct_sd']:.2f}  "
                  f"vs sub {s2['sub_pct_mean']:6.2f}  몫 {s2['share_mean']:+.3f} "
                  f"boot95 [{s2['share_boot95'][0]:+.3f},{s2['share_boot95'][1]:+.3f}]  "
                  f"개별 95% [{s2['share_indiv_range95'][0]:+.3f},{s2['share_indiv_range95'][1]:+.3f}]", flush=True)
    ps = any_t["per_subject"]
    for g in ("subject_0", "subject_1", "subject_2", "subject_3"):
        print(f"  {g}: 행단위 LOO 가 학습에 남긴 공동출연자 행 {ps[g]['coactor_rows_kept_in_rowloo']} "
              f"{ps[g]['coactor_rows_by_subject']}  train row {ps[g]['train_rowloo']} whole {ps[g]['train_wholeseq']}",
              flush=True)
# 헤드 적응 시드 0/1/2 (k별 프레임가중 개선율, fold별)
print("\n===== head adaptation (HMR 2.0, strict subject-held-out)", flush=True)
runs = {0: Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp771/results.json")),
        1: D / "head_seed1/results.json", 2: D / "head_seed2/results.json"}
H = {s: json.loads(p.read_text()) for s, p in runs.items()}
R = json.loads(Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp772/results.json")).read_text())
print("exp772 키:", list(R)[:8], flush=True)
for k in ("1", "2", "3", "4"):
    imps = [H[s]["k"][k]["imp_pct"] for s in H]
    print(f"  k={k}: head {np.mean(imps):.2f}±{np.std(imps, ddof=1):.2f} {[round(x, 2) for x in imps]}", flush=True)
    for u in range(5):
        g = f"subject_{u}"
        af = [H[s]["k"][k]["per_fold"][g]["after"] for s in H]
        print(f"      {g}: head after {np.mean(af):.2f}±{np.std(af, ddof=1):.2f} "
              f"(before {H[0]['k'][k]['per_fold'][g]['before']:.2f})", flush=True)
print(json.dumps(R, indent=1)[:1500], flush=True)
