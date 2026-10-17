"""Redacted offline release scan. Reports locations/rules, never matched values."""
import argparse
import hashlib
import json
import re
from pathlib import Path

RULES = {
    "hf-token": rb"\bhf_[A-Za-z0-9]{20,}\b",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{35,})\b",
    "openai-anthropic-key": rb"\bsk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{20,}\b",
    "cloud-access-id": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "google-api-key": rb"\bAIza[A-Za-z0-9_-]{30,}\b",
    "slack-token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "ssh-public-key": rb"(?:ssh-rsa|ssh-ed25519|ecdsa-sha2-nistp\d+) [A-Za-z0-9+/]{40,}={0,3}",
    "jwt": rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b",
    "credential-in-url": rb"https?://[^\s/:@]{2,}:[^\s/@]{8,}@",
    "bearer-literal": rb"(?i)\bBearer [A-Za-z0-9_.-]{20,}",
    "credential-assignment": rb"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|password|passwd|hf_token|modal_token_secret|vast_api_key)\s*[=:]\s*['\"][A-Za-z0-9_./+=-]{12,}['\"]",
}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}


def scan(root):
    findings = []
    files = 0
    byte_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP_DIRS.intersection(path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError("release scan does not follow symlinks")
        content = path.read_bytes()
        files += 1
        byte_count += len(content)
        for rule, pattern in RULES.items():
            for match in re.finditer(pattern, content):
                findings.append({"file": path.relative_to(root).as_posix(),
                                 "line": content.count(b"\n", 0, match.start()) + 1,
                                 "rule": rule,
                                 "fingerprint": hashlib.sha256(match.group()).hexdigest()[:12]})
    return {"scanner": "offline redacted credential regexes", "rules": sorted(RULES),
            "files_scanned": files, "bytes_scanned": byte_count,
            "findings": findings, "status": "pass" if not findings else "review_required",
            "limitations": "Pattern scan, not proof of absence; excludes git metadata and generated caches."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = scan(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["findings"]))


if __name__ == "__main__":
    main()
