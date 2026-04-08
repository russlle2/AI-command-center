"""
System agent — executes shell commands, installs applications, and manages
files on the local Windows/Linux machine.

⚠️  SECURITY NOTE
This agent runs arbitrary shell commands with the permissions of the user
that launched the AI Command Center process.  Only deploy it in a trusted
local environment.  Never expose the Gradio UI or any API port to the public
internet without authentication.

The agent supports:
  - Running shell commands (PowerShell on Windows, bash on Linux/macOS)
  - Installing Windows apps via winget / chocolatey / pip / npm / cargo
  - Downloading files with progress reporting
  - Reading/writing files
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Safe command prefix ────────────────────────────────────────────────────
_IS_WINDOWS = platform.system() == "Windows"
_SHELL = ["powershell", "-NoProfile", "-Command"] if _IS_WINDOWS else ["/bin/bash", "-c"]

# A basic denylist of commands that should never be allowed even in a trusted
# environment (destructive or irreversible operations).
_DENYLIST_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/", re.I),            # rm -rf /
    re.compile(r"\bformat\s+[a-z]:", re.I),           # format C:
    re.compile(r"\bdel\s+/[sf].*\*", re.I),           # del /s /f *
    re.compile(r"\brd\s+/[sq]\s+[a-z]:\\", re.I),     # rd /s /q C:\
    re.compile(r">\s*/dev/sd[a-z]", re.I),             # write to raw device
]


def _is_safe(command: str) -> bool:
    for pattern in _DENYLIST_PATTERNS:
        if pattern.search(command):
            return False
    return True


def _run_shell(command: str, timeout: int = 120) -> dict:
    """Execute a shell command and return stdout/stderr/returncode."""
    if not _is_safe(command):
        return {
            "stdout": "",
            "stderr": "⛔ Command blocked by safety denylist.",
            "returncode": -1,
        }

    logger.info("Executing shell command: %s", command[:200])
    try:
        proc = subprocess.run(
            _SHELL + [command],
            capture_output = True,
            text           = True,
            timeout        = timeout,
        )
        return {
            "stdout":     proc.stdout.strip(),
            "stderr":     proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "⚠️ Command timed out.", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def _install_package(package: str, manager: str = "auto") -> dict:
    """Install a software package using the appropriate package manager."""
    if manager == "auto":
        manager = _detect_manager(package)

    install_map = {
        "pip":    f"{sys.executable} -m pip install {package}",
        "npm":    f"npm install -g {package}",
        "winget": f"winget install --id {package} -e --silent",
        "choco":  f"choco install {package} -y",
        "apt":    f"sudo apt-get install -y {package}",
        "brew":   f"brew install {package}",
        "cargo":  f"cargo install {package}",
    }

    if manager not in install_map:
        return {"stdout": "", "stderr": f"Unknown package manager: {manager}", "returncode": -1}

    return _run_shell(install_map[manager], timeout=300)


def _detect_manager(package: str) -> str:
    """Heuristically choose the right package manager."""
    # If it looks like a PyPI package use pip
    if package.startswith("pip:") or "==" in package or "-" in package.lower():
        return "pip"
    if _IS_WINDOWS:
        return "winget"
    if sys.platform == "darwin":
        return "brew"
    return "apt"


def _download_file(url: str, dest: str | None = None) -> dict:
    """Download a file from a URL."""
    filename = url.split("/")[-1].split("?")[0] or f"download_{int(time.time())}"
    dest_path = Path(dest) if dest else Path.home() / "Downloads" / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s → %s", url, dest_path)
    try:
        urllib.request.urlretrieve(url, dest_path)
        return {"path": str(dest_path), "size": dest_path.stat().st_size}
    except Exception as exc:
        return {"error": str(exc)}


class SystemAgent:
    """Execute system-level tasks: shell commands, installs, file I/O."""

    def run(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Parse the prompt and dispatch to the right action.

        Supported natural language patterns:
          - "run <command>"
          - "install <package>"
          - "download <url> [to <path>]"
          - "read file <path>"
          - "write <content> to <path>"
          - "open <app>"
        """
        p = prompt.strip()

        # ── install ──────────────────────────────────────────────────────
        m = re.match(r"(?:install|download and install)\s+(.+)", p, re.I)
        if m:
            pkg = m.group(1).strip()
            result = _install_package(pkg)
            result["action"] = "install"
            result["package"] = pkg
            return result

        # ── download file ─────────────────────────────────────────────────
        m = re.match(r"download\s+(https?://\S+)(?:\s+(?:to|as|into)\s+(.+))?", p, re.I)
        if m:
            url  = m.group(1)
            dest = m.group(2)
            result = _download_file(url, dest)
            result["action"] = "download"
            return result

        # ── read file ─────────────────────────────────────────────────────
        m = re.match(r"(?:read|cat|show|open|print)\s+(?:file\s+)?(.+)", p, re.I)
        if m:
            path = Path(m.group(1).strip().strip("\"'"))
            if path.exists():
                return {"action": "read_file", "path": str(path),
                        "content": path.read_text(errors="replace")}
            return {"action": "read_file", "error": f"File not found: {path}"}

        # ── write file ────────────────────────────────────────────────────
        m = re.match(r"write\s+(.+?)\s+to\s+(.+)", p, re.I)
        if m:
            content, path_str = m.group(1), m.group(2).strip().strip("\"'")
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"action": "write_file", "path": str(path), "bytes": len(content)}

        # ── open application ─────────────────────────────────────────────
        m = re.match(r"(?:open|launch|start)\s+(.+)", p, re.I)
        if m:
            app = m.group(1).strip()
            cmd = f"Start-Process '{app}'" if _IS_WINDOWS else f"xdg-open '{app}'"
            result = _run_shell(cmd, timeout=10)
            result["action"] = "open"
            result["app"] = app
            return result

        # ── generic shell command ─────────────────────────────────────────
        m = re.match(r"(?:run|execute|shell|cmd|powershell)\s*:?\s*(.+)", p, re.I | re.S)
        if m:
            result = _run_shell(m.group(1).strip())
            result["action"] = "shell"
            return result

        # ── fallback: treat entire prompt as a shell command ──────────────
        result = _run_shell(p)
        result["action"] = "shell"
        return result
