
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RES = ROOT / "results"

SOURCE = RES / "beam_v27_phase_diverse_lns_level_33.json"
TARGET = RES / "best_verified_33.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

def verify(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if len(route) != 33 or route[0] != 0:
            return False
        if len(set(route)) != 33:
            return False
        if len(transfers) != 32:
            return False
        if any(not 0 <= int(x) < P.n_tomatoes for x in route):
            return False

        day = 0.0
        special = 0

        for i, tr in enumerate(transfers):
            a = int(tr["from"])
            b = int(tr["to"])
            start = float(tr["start_time"])
            tof = float(tr["tof"])

            if a != route[i] or b != route[i + 1]:
                return False
            if start < day - 1e-6 or tof <= 0:
                return False
            if start + tof > P.max_time + 1e-6:
                return False

            dv = float(P.compute_transfer(a, b, start, tof))

            if not math.isfinite(dv):
                return False
            if abs(dv - float(tr["dv"])) > 0.1:
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                special += 1
            else:
                return False

            if status != tr["status"]:
                return False

            day = start + tof

        return (
            special == int(s["special_used"])
            and special <= P.n_exceptions
            and abs(day - float(s["current_time"])) < 1e-5
        )
    except Exception:
        return False


data = json.loads(SOURCE.read_text(encoding="utf-8"))

if isinstance(data, list):
    states = data
else:
    states = (
        data.get("frontier")
        or data.get("states")
        or data.get("solutions")
        or ([data] if "route" in data else [])
    )

states = [
    s.get("state", s) if isinstance(s, dict) else s
    for s in states
]

valid = [
    s for s in states
    if isinstance(s, dict) and verify(s)
]

assert valid, "Asnje rruge 33 nuk kaloi riverifikimin!"

best = min(
    valid,
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

payload = {
    "level": 33,
    "verified": True,
    "source": SOURCE.name,
    "frontier": [best]
}

if TARGET.exists():
    old = json.loads(TARGET.read_text(encoding="utf-8"))
    assert old["frontier"][0] == best, (
        "Ekziston rekord tjeter 33. Nuk u mbishkrua."
    )
else:
    temp = TARGET.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )
    temp.replace(TARGET)

print("\nREKORDI 33/49 U RIVERIFIKUA!")
print("Transferime:", len(best["transfers"]), "/32")
print("SPECIAL:", best["special_used"], "/5")
print("Dita:", round(best["current_time"], 6))
print(
    "Koha e mbetur:",
    round(P.max_time - best["current_time"], 6)
)
print("Rruga:", best["route"])
print("Skedari:", TARGET.name)
print("\nHAPI 152 PERFUNDOI ME SUKSES!")
