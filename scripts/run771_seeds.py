"""exp771(헤드 적응, 엄격 subject-held-out)을 시드 1·2로 다시 돌린다 — 원본 코드 그대로.

원본 results.json(시드 0)을 덮어쓰지 않도록 OUT 만 exp775/head_seed{n} 으로 바꾼다.
"""

import os
import importlib.util
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent / "exp771_adapt_budget.py")
spec = importlib.util.spec_from_file_location("exp771", SRC)
m = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(SRC).parent))
spec.loader.exec_module(m)

for seed in (1, 2):
    m.OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), f"exp775/head_seed{seed}"))
    m.OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["exp771", "--k", "1,2,3,4", "--seed", str(seed)]
    print(f"\n######## seed {seed}  @head_seed={seed}", flush=True)
    m.main()
