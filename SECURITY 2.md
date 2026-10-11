# Security Policy

judge-kappa is a Python library and CLI that computes agreement statistics and runs LLM judges. It reads
YAML configs and result files, and can call model APIs with the operator's keys, so these are in scope:

- Code execution or unintended side effects triggered by a config or input file (unsafe deserialization,
  path traversal, command injection).
- Leaking API keys or evaluation data (logs, error messages, outputs, result files).
- Supply-chain problems: a malicious or typosquatted dependency, or a tampered release.
- Credentials committed to the repository.

**Report privately** with GitHub's private vulnerability reporting: the **Security** tab, then
**Report a vulnerability**. Do not open a public issue or PR. If that option is missing, open a public
issue that says only "security contact requested", with no details.

What to expect: an acknowledgement within 7 days and a first assessment within 14 days. This is a
volunteer project and these are targets, not guarantees. Only the latest release and `main` receive fixes.

## How the repository is protected

- Dependencies are locked (`uv.lock`); CI installs with `--locked`.
- `pip-audit` checks the locked set (all extras and the dev group) on every PR, every push to `main`
  and weekly, because advisories appear without any commit. A known vulnerability fails CI.
- Dependabot opens weekly PRs for GitHub Actions and Python dependencies, and security updates are on.
- GitHub Actions are pinned to full commit SHAs and run with read-only token permissions.
- Secret scanning and push protection are on.
- `main` requires a pull request, resolved review threads and passing checks (tests, dependency audit,
  hygiene); no force-push or deletion.
- A `hygiene` check requires a `Signed-off-by` line on every commit and rejects AI-session links or ids
  in commits, diffs and PR text.
- Releases use PyPI trusted publishing (OIDC, no stored token) behind a protected `pypi` environment, and
  only a version already published to TestPyPI can be released.
