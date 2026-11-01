
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "beam_v10_snapshot_audit.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) < 2
            or len(route) > P.n_tomatoes
            or route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
            or any(
                type(t) is not int or t < 0 or t >= P.n_tomatoes
                for t in route
            )
        ):
            return False, "Struktura e rruges"

        day = 0.0
        specials = 0

        for i, tr in enumerate(transfers):
            start = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or start < day - 1e-6
                or tof <= 0
                or start + tof > P.max_time + 1e-6
            ):
                return False, f"Gabim kohe/lidhjeje ne hapin {i+1}"

            dv = float(P.compute_transfer(
                route[i], route[i + 1], start, tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) > 0.1
            ):
                return False, f"Gabim Delta-V ne hapin {i+1}"

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                specials += 1
            else:
                return False, f"Transferim i pavlefshem {i+1}"

            if tr["status"] != status:
                return False, f"Status i pasakte ne hapin {i+1}"

            day = start + tof

        if (
            abs(day - float(state["current_time"])) > 1e-5
            or specials != state["special_used"]
            or specials > P.n_exceptions
        ):
            return False, "Gjendja perfundimtare nuk perputhet"

        return True, "OK"

    except Exception as error:
        return False, str(error)

report = {
    "strategies": {},
    "highest_verified_level": 0
}

print("VERIFIKIMI I SNAPSHOT-EVE")
print("=" * 60, flush=True)

for name in ["A16", "B17", "C19"]:
    paths = sorted(
        RESULTS.glob(f"beam_v10_saved_{name}_level_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1])
    )

    if not paths:
        print(name, "- ende pa snapshot", flush=True)
        continue

    path = paths[-1]
    data = json.loads(path.read_text())
    level = data["level"]
    states = data["frontier"]

    verified = []
    invalid = []

    for state in states:
        ok, reason = verify(state)

        if ok:
            verified.append(state)
        else:
            invalid.append(reason)

    verified.sort(
        key=lambda s: (
            -len(s["route"]),
            s["special_used"],
            s["current_time"]
        )
    )

    result = {
        "snapshot": path.name,
        "level": level,
        "total_routes": len(states),
        "verified_routes": len(verified),
        "invalid_reasons": invalid[:10],
        "best_state": verified[0] if verified else None
    }

    report["strategies"][name] = result

    if verified:
        report["highest_verified_level"] = max(
            report["highest_verified_level"],
            level
        )

    print(
        f"{name}: Niveli {level}/49 | "
        f"Te verifikuara {len(verified)}/{len(states)}",
        flush=True
    )

    if verified:
        best = verified[0]

        print(
            f"  Dita {best['current_time']:.3f} | "
            f"SPECIAL {best['special_used']}/5 | "
            f"Fund {best['route'][-1]}",
            flush=True
        )

        print("  Rruga:", best["route"], flush=True)

        special_positions = [
            i + 1
            for i, tr in enumerate(best["transfers"])
            if tr["status"] == "SPECIAL"
        ]

        print(
            "  SPECIAL ne transferimet:",
            special_positions,
            flush=True
        )

OUTPUT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\nREZULTATI I SAKTE V10")
print("=" * 60)
print(
    "Niveli maksimal i verifikuar:",
    report["highest_verified_level"],
    "/49"
)
print("Skedari:", OUTPUT.name)
