from pathlib import Path
import subprocess

def get_active_branch_name(wd="."):

    head_dir = Path(wd) / ".git" / "HEAD"
    with head_dir.open("r") as f: content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def get_git_commit_hash():
    try:
        # Run git command in the notebooks directory
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd="..",  # Move up from the 'notebooks' folder to the repo root
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit_hash
    except subprocess.CalledProcessError:
        return None  # Not a git repository or error occurred