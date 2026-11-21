"""Re-create submissions/<level>_submission.json from solutions/<level>_best.json."""
import json
from pathlib import Path

from solve import CHALLENGE, PROBLEM_ID
from submission_helper import create_submission

ROOT = Path(__file__).resolve().parent
(ROOT / "submissions").mkdir(exist_ok=True)
for level in ["easy", "medium", "hard"]:
    fn = ROOT / "solutions" / f"{level}_best.json"
    if not fn.exists():
        continue
    d = json.loads(fn.read_text())
    x = [float(v) for v in d["x"][: 2 * ((len(d["x"]) + 2) // 3 - 1)]] + [int(v) for v in d["x"][2 * ((len(d["x"]) + 2) // 3 - 1):]]
    out = ROOT / "submissions" / f"{level}_submission.json"
    create_submission(CHALLENGE, PROBLEM_ID[level], x, str(out), f"kttsp-{level}", f"f={d['f']:.4f}")
    print(f"{level}: f={d['f']:.4f} -> {out}")
