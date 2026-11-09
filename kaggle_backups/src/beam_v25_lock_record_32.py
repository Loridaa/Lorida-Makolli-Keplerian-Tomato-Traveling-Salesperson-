
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v24_lns5_level_32.json"
TARGET = RESULTS / "best_verified_32.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

def verify(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if len(route) != 32 or route[0] != 0:
            return False
        if len(set(route)) != 32:
            return False
        if len(transfers) != 31:
            return False

        day = 0.0
        used = 0

        for i, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if tr["from"] != route[i]:
                return False
            if tr["to"] != route[i + 1]:
                return False
            if depart < day - 1e-6 or tof <= 0:
                return False
            if depart + tof > P.max_time + 1e-6:
                return False

            dv = float(P.compute_transfer(
                tr["from"], tr["to"], depart, tof
            ))

            if not math.isfinite(dv):
                return False
            if abs(dv - float(tr["dv"])) > 0.1:
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                used += 1
            else:
                return False

            if status != tr["status"]:
                return False

            day = depart + tof

        return (
            used == s["special_used"]
            and used <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )
    except Exception:
        return False


data = json.loads(SOURCE.read_text(encoding="utf-8"))
states = data["frontier"]

valid = [s for s in states if verify(s)]

assert valid, "Rekordi 32 nuk kaloi riverifikimin."

best = min(
    valid,
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

# Ruajme bazen e re ne skedar te vecante.
payload = {
    "level": 32,
    "verified": True,
    "source": SOURCE.name,
    "frontier": [best]
}

if TARGET.exists():
    previous = json.loads(TARGET.read_text())
    assert previous["frontier"][0] == best, (
        "Ekziston nje rekord tjeter. Nuk u mbishkrua."
    )
else:
    temp = TARGET.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )
    temp.replace(TARGET)

print("REKORDI 32/49 U RIVERIFIKUA!")
print("Transferime:", len(best["transfers"]), "/31")
print("SPECIAL:", best["special_used"], "/5")
print("Dita:", round(best["current_time"], 6))
print("Koha e mbetur:",
      round(P.max_time - best["current_time"], 6))
print("Rruga:", best["route"])
print("Skedari:", TARGET.name)
print("HAPI 149 PERFUNDOI ME SUKSES!")
