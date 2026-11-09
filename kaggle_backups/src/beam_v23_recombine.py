
from pathlib import Path
import subprocess
import os
import sys

ROOT = Path("/kaggle/working/esa-spoc4")
SRC = ROOT / "src"
PYTHON = Path("/kaggle/working/esa-pykep-env/bin/python")
SCRIPT = SRC / "beam_v23_recombine.py"

print("HAPI 147 - RREGULLIMI AUTOMATIK")
print("=" * 55)

assert ROOT.exists(), "Projekti mungon."
assert PYTHON.exists(), "Mjedisi PyKEP mungon."

# Gjejme kodin e plote nga historia e Kaggle.
history = get_ipython().history_manager.input_hist_raw

source_found = False

for source in reversed(history):
    if (
        "beam_v23_recombine_checkpoint.json" not in source
        or "def try_bridge" not in source
        or "def build_campaign" not in source
        or "PERMBLEDHJA E HAPIT 147" not in source
    ):
        continue

    try:
        compile(source, str(SCRIPT), "exec")
    except SyntaxError:
        continue

    SCRIPT.write_text(source, encoding="utf-8")
    source_found = True
    break

if not source_found and SCRIPT.exists():
    try:
        compile(
            SCRIPT.read_text(encoding="utf-8"),
            str(SCRIPT),
            "exec"
        )
        source_found = True
    except SyntaxError:
        pass

if not source_found:
    print("KODI I HAPIT 147 NUK U GJET NE HISTORI.")
    print("Me dergo kete rezultat per ta rikthyer.")
else:
    print("OK - KODI I HAPIT 147 U GJET!")
    print("OK - Sintaksa eshte ne rregull!")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"

    print("\nEKZEKUTIMI ME PYKEP")
    print("=" * 55)

    try:
        result = subprocess.run(
            [str(PYTHON), "-u", str(SCRIPT)],
            cwd=str(SRC),
            env=env,
            capture_output=True,
            text=True,
            timeout=160
        )

        print(result.stdout[-14000:])

        if result.returncode != 0:
            print("GABIM:", result.stderr[-4000:])

    except subprocess.TimeoutExpired as error:
        if error.stdout:
            output = error.stdout
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            print(output[-5000:])
        print("Checkpoint-i mund te vazhdohet.")

    print("\nBACKUP NE GITHUB")
    sys.path.insert(0, str(SRC))

    try:
        from github_backup import backup_to_github
        backup_to_github()
    except Exception as error:
        print("Backup nuk u konfirmua:", error)