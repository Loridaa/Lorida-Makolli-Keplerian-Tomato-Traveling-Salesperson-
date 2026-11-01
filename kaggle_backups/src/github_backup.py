
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from urllib.parse import quote
import requests
import hashlib
import base64

ROOT = Path("/kaggle/working/esa-spoc4")
REPO = "Loridaa/Lorida-Makolli-Keplerian-Tomato-Traveling-Salesperson-"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}"

def backup_to_github():
    print("BACKUP AUTOMATIK NE GITHUB")
    print("=" * 55)

    token = UserSecretsClient().get_secret("GITHUB_TOKEN")

    if not token:
        raise RuntimeError("Mungon GITHUB_TOKEN ne Kaggle Secrets.")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    })

    # Lexojme listen e skedareve ekzistues ne GitHub
    response = session.get(
        f"{API}/git/trees/{BRANCH}",
        params={"recursive": 1},
        timeout=60
    )
    response.raise_for_status()

    tree_data = response.json()

    if tree_data.get("truncated"):
        raise RuntimeError("Lista e GitHub eshte e cunguar.")

    remote_files = {
        item["path"]: item["sha"]
        for item in tree_data["tree"]
        if item["type"] == "blob"
    }

    local_files = []

    for folder in ["src", "results"]:
        directory = ROOT / folder

        if directory.exists():
            local_files.extend(
                p for p in directory.rglob("*")
                if p.is_file()
                and p.suffix in (".py", ".json")
            )

    print("Repository:", REPO)
    print("Branch:", BRANCH)
    print("Skedare te gjetur:", len(local_files))

    uploaded = 0
    unchanged = 0
    failed = 0

    for path in sorted(local_files):
        relative = path.relative_to(ROOT).as_posix()
        remote = "kaggle_backups/" + relative

        content = path.read_bytes()

        # GitHub blob SHA
        header = f"blob {len(content)}\0".encode()
        digest = hashlib.sha1(header + content).hexdigest()

        old_sha = remote_files.get(remote)

        if old_sha == digest:
            unchanged += 1
            continue

        payload = {
            "message": f"Backup ESA: {relative}",
            "content": base64.b64encode(content).decode(),
            "branch": BRANCH
        }

        if old_sha:
            payload["sha"] = old_sha

        url = f"{API}/contents/{quote(remote, safe='/')}"

        try:
            result = session.put(
                url,
                json=payload,
                timeout=90
            )

            if result.status_code in (200, 201):
                uploaded += 1
                print("U RUAJT:", relative)
            else:
                failed += 1
                print("GABIM:", relative, result.status_code)

        except requests.RequestException as e:
            failed += 1
            print("GABIM:", relative, type(e).__name__)

    print("\nPERMBLEDHJA")
    print("=" * 55)
    print("Skedare te ngarkuar:", uploaded)
    print("Pa ndryshime:", unchanged)
    print("Gabime:", failed)

    if failed == 0:
        print("BACKUP PERFUNDOI ME SUKSES!")
    else:
        print("BACKUP I PJESSHEM - kontrollo gabimet.")

    print(f"https://github.com/{REPO}/tree/{BRANCH}/kaggle_backups")

    return failed == 0
