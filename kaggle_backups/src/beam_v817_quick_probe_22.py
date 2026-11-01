
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v817_rescue_level_21.json"
CACHE = RESULTS / "beam_v817_quick_cache_21.json"
REPORT = RESULTS / "beam_v817_quick_report_22.json"

# Rrjet paraprak: 24 kombinime per domate
WAITS = [0, 1, 3, 8]
TOFS = [0.5, 1, 2, 5, 8, 13]

def load(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

beam = load(SOURCE)["beam"]

matching = [
    s for s in beam
    if s["special_used"] == 4
    and s["route"][-1] == 27
    and abs(s["current_time"] - 67.0) < 1e-6
]

assert len(matching) == 1, (
    f"Pritej 1 rruge, u gjeten {len(matching)}"
)

state = matching[0]

route = state["route"]
now = float(state["current_time"])
used = int(state["special_used"])
current = route[-1]

key = json.dumps([route, now, used])

cache = load(CACHE, {})

if key not in cache:
    cache[key] = {"targets": {}}

saved = cache[key]["targets"]

remaining = [
    t for t in range(problem.n_tomatoes)
    if t not in route
]

print("HAPI 76 - V8.17 QUICK PROBE")
print("=" * 55)
print("Niveli: 21 -> 22")
print("Koha fillestare:", now)
print("Domatja fundore:", current)
print("Speciale te perdorura:", used)
print("Domate per kontroll:", len(remaining))
print("Kombinime per domate:", len(WAITS)*len(TOFS))
print("=" * 55, flush=True)

for i, target in enumerate(remaining, 1):

    target_key = str(target)

    if target_key in saved:
        print(
            f"[{i}/{len(remaining)}] "
            f"Domatja {target}: nga CACHE",
            flush=True
        )
        continue

    best_normal = None
    best_special = None
    min_dv = None

    for wait in WAITS:
        start = now + wait

        for tof in TOFS:
            arrival = start + tof

            if arrival > problem.max_time:
                continue

            try:
                dv = float(
                    problem.compute_transfer(
                        current, target, start, tof
                    )
                )
            except Exception:
                continue

            if not math.isfinite(dv) or dv < 0:
                continue

            if min_dv is None or dv < min_dv:
                min_dv = dv

            move = {
                "target": target,
                "wait": wait,
                "tof": tof,
                "arrival": arrival,
                "dv": dv
            }

            if dv < problem.dv_threshold:
                move["status"] = "NORMAL"

                if best_normal is None or (
                    arrival, dv
                ) < (
                    best_normal["arrival"],
                    best_normal["dv"]
                ):
                    best_normal = move

            elif (
                dv < problem.exception_threshold
                and used < problem.n_exceptions
            ):
                move["status"] = "SPECIAL"

                if best_special is None or (
                    arrival, dv
                ) < (
                    best_special["arrival"],
                    best_special["dv"]
                ):
                    best_special = move

    saved[target_key] = {
        "best_normal": best_normal,
        "best_special": best_special,
        "min_dv": min_dv
    }

    # Ruhet pas cdo domateje
    save(CACHE, cache)

    status = (
        "NORMAL" if best_normal
        else "SPECIAL" if best_special
        else "ASNJE"
    )

    print(
        f"[{i}/{len(remaining)}] "
        f"Domatja {target}: {status}",
        flush=True
    )

normals = [
    v["best_normal"]
    for v in saved.values()
    if v["best_normal"] is not None
]

specials = [
    v["best_special"]
    for v in saved.values()
    if v["best_special"] is not None
]

print("\nREZULTATI V8.17 - RRUGA 67 DITE")
print("=" * 55)

print("Domate te kontrolluara:", len(saved))
print("Transferime NORMAL drejt:", len(normals),
      "domateve")
print("Transferime SPECIAL drejt:", len(specials),
      "domateve")

if normals:
    best = min(
        normals,
        key=lambda m: (m["arrival"], m["dv"])
    )

    print("\nTRANSFERIMI NORMAL ME I MIRE")
    print("Domatja:", best["target"])
    print("Arritja:", best["arrival"])
    print("DeltaV:", round(best["dv"], 3))
    print("Speciale te perdorura mbeten:", used)

if specials:
    best = min(
        specials,
        key=lambda m: (m["arrival"], m["dv"])
    )

    print("\nTRANSFERIMI SPECIAL ME I MIRE")
    print("Domatja:", best["target"])
    print("Arritja:", best["arrival"])
    print("DeltaV:", round(best["dv"], 3))
    print("Speciale pas transferimit:", used + 1)

if not normals:
    print(
        "\nNuk u gjet NORMAL ne rrjetin paraprak."
    )
    print(
        "Kjo NUK do te thote se eshte i pamundur."
    )

save(REPORT, {
    "source": SOURCE.name,
    "route": route,
    "initial_time": now,
    "special_used": used,
    "grid": {"waits": WAITS, "tofs": TOFS},
    "targets_checked": len(saved),
    "best_normals": normals,
    "best_specials": specials
})

print("\nHAPI 76 PERFUNDOI!")
