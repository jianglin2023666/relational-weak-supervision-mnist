"""Download data and figures from fixed PR commits without merging branches."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote

OUT = Path(__file__).resolve().parent / "comparison_inputs"
REPO = "jianglin2023666/relational-weak-supervision-mnist"
TASKS = [(2, 3, "mission2_fh"), (3, 2, "mission3_why"), (4, 5, "mission4_zdy")]


def get(url):
    with urlopen(Request(url, headers={"User-Agent": "Mission5-data-check"}), timeout=90) as response:
        return response.read()


def main():
    records = []
    for task, pr_number, prefix in TASKS:
        pr = json.loads(get(f"https://api.github.com/repos/{REPO}/pulls/{pr_number}"))
        source_repo, sha = pr["head"]["repo"]["full_name"], pr["head"]["sha"]
        tree = json.loads(get(f"https://api.github.com/repos/{source_repo}/git/trees/{sha}?recursive=1"))
        target = OUT / f"task{task}"
        files = [f for f in tree["tree"] if f["type"] == "blob" and f["path"].startswith(prefix + "/")
                 and Path(f["path"]).suffix.lower() not in {".pth", ".npz", ".pptx"}]

        def fetch(item):
            relative = Path(item["path"])
            if ".." in relative.parts:
                raise ValueError(relative)
            dest = target / relative
            url = f"https://raw.githubusercontent.com/{source_repo}/{sha}/{quote(item['path'])}"
            content = get(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return {"path": str(dest.relative_to(OUT)), "sha256": hashlib.sha256(content).hexdigest()}

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            downloaded = list(pool.map(fetch, files))
        records.append({"task": task, "pr": pr_number, "url": pr["html_url"], "sha": sha,
                        "state": pr["state"], "merged_at": pr["merged_at"], "files": downloaded})
        print(f"Task {task}: {len(downloaded)} files from PR #{pr_number}, commit {sha}", flush=True)
    (OUT / "download_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Downloaded data, reports and figures. Excluded weights, NPZ arrays and other members' PPTX files.")


if __name__ == "__main__":
    main()
