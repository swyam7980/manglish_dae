"""Publish report/index.html to a gh-pages branch that GitHub Pages can serve.

The branch holds only index.html and an empty .nojekyll file. The commit is built with git plumbing and a temporary
index, so no working folder is touched and the current branch is never switched. Running it again with an unchanged
page does nothing; with a changed page it adds a commit on top of the previous one.

  python src/publish_pages.py            # updates the local gh-pages branch
  git push origin gh-pages               # then publish it
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = "gh-pages"


def git(repo, *args, env=None, stdin=None):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env, input=stdin)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout.decode(errors="replace").strip()


def blob(repo, data):
    return git(repo, "hash-object", "-w", "--no-filters", "--stdin", stdin=data)


def publish(repo, html_path, branch=BRANCH, message=None):
    """Create or update `branch` so its tree is index.html + .nojekyll. Returns the new commit, or None if unchanged."""
    repo = Path(repo)
    page = Path(html_path).read_bytes().replace(b"\r\n", b"\n")
    ref = f"refs/heads/{branch}"
    try:
        parent = git(repo, "rev-parse", "--verify", "-q", ref)
    except RuntimeError:
        parent = None
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        for name, data in (("index.html", page), (".nojekyll", b"")):
            git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob(repo, data)},{name}", env=env)
        tree = git(repo, "write-tree", env=env)
    if parent and git(repo, "rev-parse", f"{parent}^{{tree}}") == tree:
        return None
    try:
        source = git(repo, "rev-parse", "--short", "HEAD")
    except RuntimeError:
        source = "unknown"
    args = ["commit-tree", tree, "-m", message or f"Publish report page from {source}"]
    if parent:
        args += ["-p", parent]
    commit = git(repo, *args)
    git(repo, "update-ref", ref, commit)
    return commit


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(ROOT / "report" / "index.html"))
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--repo", default=str(ROOT))
    a = ap.parse_args(argv)
    commit = publish(a.repo, a.html, a.branch)
    if commit:
        print(f"{a.branch} is now {commit[:7]}. Publish it with: git push origin {a.branch}")
    else:
        print(f"{a.branch} already has this page; nothing to do.")
    return commit


if __name__ == "__main__":
    main()
