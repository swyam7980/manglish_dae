"""The gh-pages publisher builds a branch holding only the page, and is idempotent."""
import subprocess

import publish_pages


def run(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run(repo, "init", "-q", "-b", "main")
    run(repo, "config", "user.email", "t@example.com")
    run(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n")
    run(repo, "add", "README.md")
    run(repo, "commit", "-q", "-m", "init")
    return repo


def test_branch_holds_only_the_page_and_nojekyll(tmp_path):
    repo = make_repo(tmp_path)
    page = tmp_path / "index.html"
    page.write_bytes(b"<!doctype html>\r\n<title>x</title>\r\n")
    commit = publish_pages.publish(repo, page)
    assert commit
    assert run(repo, "ls-tree", "-r", "--name-only", "gh-pages").split() == [".nojekyll", "index.html"]
    assert run(repo, "show", "gh-pages:index.html") == "<!doctype html>\n<title>x</title>"      # CRLF normalised to LF
    assert run(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"                              # current branch untouched
    assert run(repo, "status", "--porcelain") == ""                                              # working folder untouched


def test_second_run_is_a_no_op_and_changes_add_a_commit(tmp_path):
    repo = make_repo(tmp_path)
    page = tmp_path / "index.html"
    page.write_text("<p>one</p>\n")
    first = publish_pages.publish(repo, page)
    assert publish_pages.publish(repo, page) is None                                             # same page, nothing new
    page.write_text("<p>two</p>\n")
    second = publish_pages.publish(repo, page)
    assert second and second != first
    assert run(repo, "rev-parse", f"{second}^") == first                                          # history is kept
    assert run(repo, "rev-list", "--count", "gh-pages") == "2"
