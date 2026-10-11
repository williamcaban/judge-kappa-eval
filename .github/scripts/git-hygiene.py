#!/usr/bin/env python3
"""git-hygiene: catches sloppy mistakes in commits and PR/issue text, for every repo.

Rules (stdlib only; exit 1 = blocked, findings on stderr, nothing secret is echoed):
  1. sign-off      every commit message carries the human's `Signed-off-by:` line, and no sign-off
                   names an AI or a bot. A PR body ends with the same line.
  2. session info  no Claude or Hermes session URL, session id or session trailer in commit
                   messages, staged files, or PR / issue / comment text. Those links are tied to a
                   private conversation; on a public repo they are a security exposure.

Modes:
  commit-msg FILE            check a commit message (git commit-msg hook)
  pre-commit                 check the staged diff for session URLs / ids
  pre-push REMOTE URL        stdin = git's ref lines; check the messages of every commit being pushed
  text [FILE|-] [--require-signoff]   check free text (PR title/body, issue, comment)
  range BASE HEAD            CI: every commit in BASE..HEAD (message + diff). Bot-authored commits
                             (Dependabot, CodeRabbit) need no human sign-off but get the session checks
  claude-hook                Claude Code PreToolUse hook: stdin = tool JSON; exit 2 blocks the call

Settings: GIT_HYGIENE_SIGNOFF overrides the required sign-off line.
CI only: GIT_HYGIENE_STRICT=1 makes git failures and internal errors fail the check (local hooks stay
fail-open); GIT_HYGIENE_ALLOW_BOT_COMMITS=1 lets `range` skip the human sign-off for bot-authored
commits, and the workflow sets it only when GitHub itself says the PR author is a bot.
Install on a machine: scripts/git-hygiene/install.sh. Hermes runs the same rules through
contrib-guard.py (config/git-hygiene.py is a copy of this file; `make sync-docs` refreshes it).
A hook can be skipped with --no-verify; do not.
"""
import json
import os
import re
import subprocess
import sys

SIGNOFF = os.environ.get("GIT_HYGIENE_SIGNOFF", "Signed-off-by: William Caban <william.caban@gmail.com>")
ZERO = "0" * 40
STRICT = os.environ.get("GIT_HYGIENE_STRICT") == "1"
ALLOW_BOT_COMMITS = os.environ.get("GIT_HYGIENE_ALLOW_BOT_COMMITS") == "1"

# Strong patterns: also checked in files and in anything a tool is about to run.
STRONG = [
    ("claude-session-url", re.compile(r"claude\.ai/(?:code/session_|chat/|share/)[A-Za-z0-9_-]{6,}")),
    ("claude-session-id", re.compile(r"\bsession_(?=[0-9A-Za-z]*\d)(?=[0-9A-Za-z]*[A-Z])[0-9A-Za-z]{20,}\b")),
]
# Message / text only: shapes that are legitimate in some files (memory frontmatter, logs).
TEXT_ONLY = [
    ("session-trailer", re.compile(r"(?im)^\s*(?:claude|hermes|agent)[- ]?session(?:[- ]?id)?\s*:")),
    ("hermes-session-id", re.compile(r"\b20\d{6}_\d{6}_[0-9a-f]{6,10}\b")),
    ("origin-session-id", re.compile(r"\boriginSessionId\b")),
]
AI_SIGNOFF = re.compile(r"(?im)^\s*signed-off-by:.*(claude|hermes|anthropic|\bbot\b|\[bot\]|noreply@anthropic)")


def git(*args, stdin=None):
    r = subprocess.run(["git", *args], input=stdin, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        if STRICT:
            raise RuntimeError(f"git {args[0]} failed with status {r.returncode}")
        return ""
    return r.stdout


class Report:
    def __init__(self):
        self.blocks = []

    def block(self, kind, where, msg):
        self.blocks.append(f"  [{kind}] {where}: {msg}")

    def finish(self, label):
        if not self.blocks:
            return 0
        print(f"git-hygiene: {label} BLOCKED, {len(self.blocks)} finding(s):", file=sys.stderr)
        for b in self.blocks[:30]:
            print(b, file=sys.stderr)
        print("Fix the content and retry (do not use --no-verify). Rules: scripts/git-hygiene/git-hygiene.py",
              file=sys.stderr)
        return 1


def session_hits(text, strong_only=False):
    """-> [(name, line_no)] for session information in text (the match itself is never shown)."""
    rules = STRONG if strong_only else STRONG + TEXT_ONLY
    out = []
    for name, rx in rules:
        for m in rx.finditer(text):
            out.append((name, text.count("\n", 0, m.start()) + 1))
    return out


def check_message(rep, msg, where, signoff=True, bot=False):
    body = "\n".join(l for l in msg.splitlines() if not l.startswith("#"))
    subject = body.strip().splitlines()[0] if body.strip() else ""
    if signoff and not subject.startswith("Merge "):
        if not re.search(r"(?m)^" + re.escape(SIGNOFF) + r"\s*$", body):
            rep.block("signoff", where, f"missing trailer: {SIGNOFF}")
    if AI_SIGNOFF.search(body) and not bot:
        rep.block("signoff", where, "Signed-off-by must name the human, not an AI or a bot")
    for name, ln in session_hits(body):
        rep.block(name, where, f"line {ln}: remove the session information (value not shown)")


def check_text(rep, text, where, require_signoff=False):
    for name, ln in session_hits(text):
        rep.block(name, where, f"line {ln}: remove the session information (value not shown)")
    if AI_SIGNOFF.search(text):
        rep.block("signoff", where, "Signed-off-by must name the human, not an AI or a bot")
    if require_signoff and not text.rstrip().endswith(SIGNOFF):
        rep.block("signoff", where, f"end the text with: {SIGNOFF}")


def check_diff(rep, diff, where):
    path, ln = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].removeprefix("b/") if raw[4:] != "/dev/null" else None
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            ln = int(m.group(1)) if m else 0
        elif raw.startswith("+") and not raw.startswith("+++"):
            if path:
                for name, _ in session_hits(raw[1:], strong_only=True):
                    rep.block(name, f"{path}:{ln}", "remove the session link or id (value not shown)")
            ln += 1


# ---------------------------------------------------------------- modes

def mode_commit_msg(args):
    rep = Report()
    check_message(rep, open(args[0], encoding="utf-8", errors="replace").read(), "commit message")
    return rep.finish("commit message")


def mode_pre_commit(_args):
    rep = Report()
    check_diff(rep, git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff"), "staged")
    return rep.finish("commit")


def mode_pre_push(_args):
    rep = Report()
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        _, lsha, rref, rsha = parts
        if lsha == ZERO or rref.startswith("refs/tags/"):
            continue
        rng = [lsha, "--not", "--remotes"] if rsha == ZERO else [f"{rsha}..{lsha}"]
        for sha in git("rev-list", "--no-merges", *rng).split():
            check_message(rep, git("log", "-1", "--format=%B", sha), f"commit {sha[:8]}")
    return rep.finish("push")


def mode_text(args):
    require = "--require-signoff" in args
    args = [a for a in args if a != "--require-signoff"]
    src = args[0] if args and args[0] != "-" else None
    text = open(src, encoding="utf-8", errors="replace").read() if src else sys.stdin.read()
    rep = Report()
    check_text(rep, text, "text", require)
    return rep.finish("PR/issue text")


def mode_range(args):
    base, head = args[0], args[1]
    rep = Report()
    for sha in git("rev-list", "--no-merges", f"{base}..{head}").split():
        # Commit metadata is spoofable; the exemption applies only when CI established that the PR
        # author is a bot (GIT_HYGIENE_ALLOW_BOT_COMMITS) and the commit also looks bot-authored.
        bot = ALLOW_BOT_COMMITS and "[bot]" in git("log", "-1", "--format=%an <%ae>", sha)
        check_message(rep, git("log", "-1", "--format=%B", sha), f"commit {sha[:8]}", signoff=not bot, bot=bot)
    check_diff(rep, git("diff", "-U0", "--no-color", "--no-ext-diff", f"{base}...{head}"), "diff")
    return rep.finish("pull request commits")


PR_WRITE = re.compile(r"\bgh\s+(?:-\S+\s+\S+\s+)*(pr|issue)\s+(create|edit|comment|review)\b")
BODY_FILE = re.compile(r"(?:--body-file|-F)[ =]+(['\"]?)([^\s'\"]+)\1")


def mode_claude_hook(_args):
    """PreToolUse hook for Claude Code (Bash tool). Exit 2 = block, stderr goes back to Claude."""
    try:
        data = json.load(sys.stdin)
        cmd = data.get("tool_input", {}).get("command", "")
        cwd = data.get("cwd") or os.getcwd()
    except Exception:
        return 0
    texts = [cmd]
    for m in BODY_FILE.finditer(cmd):  # PR/issue body read from a file
        p = m.group(2) if os.path.isabs(m.group(2)) else os.path.join(cwd, m.group(2))
        try:
            texts.append(open(p, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    rep = Report()
    for t in texts:
        for name, ln in session_hits(t, strong_only=True):
            rep.block(name, "command", "this would write a session link or id into a repo, PR or comment")
    pr = PR_WRITE.search(cmd)
    if pr and pr.group(1) == "pr" and pr.group(2) == "create":
        if not any(SIGNOFF in t for t in texts):
            rep.block("signoff", "gh pr create", f"the PR body must end with: {SIGNOFF}")
    if rep.blocks:
        rc = rep.finish("tool call")
        return 2 if rc else 0
    return 0


MODES = {"commit-msg": mode_commit_msg, "pre-commit": mode_pre_commit, "pre-push": mode_pre_push,
         "text": mode_text, "range": mode_range, "claude-hook": mode_claude_hook}


def main(argv):
    if len(argv) < 2 or argv[1] not in MODES:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        return MODES[argv[1]](argv[2:])
    except Exception as e:  # a bug here must be loud but must not freeze all work
        if STRICT:
            print(f"git-hygiene: INTERNAL ERROR, failing the check ({type(e).__name__}: {e})", file=sys.stderr)
            return 1
        print(f"git-hygiene: INTERNAL ERROR, check skipped ({type(e).__name__}: {e})", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
