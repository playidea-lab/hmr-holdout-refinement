"""재배열 후 SPIN: GT 오프셋·난이도 상관·관절별 PA, 그리고 기존 결과 파일 요약 출력."""

import os
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402
import exp775_share_seeds as E  # noqa: E402
import spin_check2 as C2  # noqa: E402
import spin_check as C1  # noqa: E402


def main():
    names, center, seq, subj = E.load_rows()
    p_h2, g_h2 = E.load_backbone("4DHumans", names, center, False)
    p_tk, g_tk = E.load_backbone("TokenHMR", names, center, False)
    p_raw, g_raw = E.load_backbone("SPIN", names, center, False)
    p_sp, g_sp = E.load_backbone("SPIN", names, center, True)
    sub = np.arange(0, len(g_h2), 5)
    for nm, gs in (("재배열 전", g_raw), ("재배열 후", g_sp)):
        for mn, mp in (("as-is", C2.MAP_AS_IS), ("swapped", C2.MAP_SWAP)):
            d = C2.pa(gs[sub][:, mp], g_h2[sub][:, C2.LSP12])
            print(f"SPIN GT {nm} {mn:8s} vs HMR2 GT(12관절 이름맞춤) PA 거리 {d:7.2f} mm", flush=True)
    print(f"참고 TokenHMR vs HMR2 GT(12관절) {C2.pa(g_tk[sub][:, C2.LSP12], g_h2[sub][:, C2.LSP12]):.2f} mm",
          flush=True)
    e_h2 = H.compute_pa_mpjpe(p_h2, g_h2)
    e_tk = H.compute_pa_mpjpe(p_tk, g_tk)
    for nm, (pp, gg) in (("재배열 전", (p_raw, g_raw)), ("재배열 후", (p_sp, g_sp))):
        e = H.compute_pa_mpjpe(pp, gg)
        print(f"SPIN {nm}: baseline {e.mean():.4f}  프레임 난이도 상관 r(SPIN,HMR2) "
              f"{np.corrcoef(e, e_h2)[0, 1]:.3f}  r(SPIN,TokenHMR) {np.corrcoef(e, e_tk)[0, 1]:.3f}", flush=True)
    print(f"참고 r(TokenHMR,HMR2) {np.corrcoef(e_tk, e_h2)[0, 1]:.3f}", flush=True)
    pj = C1.per_joint_pa(p_sp, g_sp)
    print("SPIN 관절별 PA(재배열 후, 파일 순서 0=골반): " + " ".join(f"{x:.2f}" for x in pj.mean(0)), flush=True)
    for path in (os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp771/results.json"), os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp774/results.json")):
        try:
            d = json.load(open(path))
        except OSError as ex:
            print(f"{path}: 읽기 실패 {ex}", flush=True)
            continue
        print(f"\n== {path} 최상위 키 {list(d)[:12]}", flush=True)
        if "exp771" in path:
            print(json.dumps(d, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    main()
