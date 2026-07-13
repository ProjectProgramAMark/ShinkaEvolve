from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import time

import psutil
import pytest

from shinka.launch.local import monitor, submit


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _process_is_active(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_normal_local_evaluation_preserves_results_and_logs(tmp_path: Path) -> None:
    program = """
import json
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])
(results_dir / "correct.json").write_text(
    json.dumps({"correct": True}), encoding="utf-8"
)
(results_dir / "metrics.json").write_text(
    json.dumps({"score": 1.25}), encoding="utf-8"
)
print("normal stdout")
print("normal stderr", file=sys.stderr)
"""

    process = submit(
        str(tmp_path),
        [sys.executable, "-c", program, str(tmp_path)],
    )
    results = monitor(process, str(tmp_path), poll_interval=0.01)

    assert process.returncode == 0
    assert results["correct"] == {"correct": True}
    assert results["metrics"] == {"score": 1.25}
    assert "normal stdout" in results["stdout_log"]
    assert "normal stderr" in results["stderr_log"]
    assert all(file_handle.closed for file_handle in process.log_files)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_kills_spawned_grandchild(tmp_path: Path) -> None:
    grandchild_pid_path = tmp_path / "grandchild.pid"
    program = """
from pathlib import Path
import subprocess
import sys
import time

grandchild = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"]
)
Path(sys.argv[1]).write_text(str(grandchild.pid), encoding="utf-8")
time.sleep(60)
"""

    process = submit(
        str(tmp_path),
        [sys.executable, "-c", program, str(grandchild_pid_path)],
    )
    grandchild_pid: int | None = None

    try:
        _wait_until(grandchild_pid_path.exists)
        grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))
        assert os.getpgid(process.pid) == process.pid
        assert os.getpgid(grandchild_pid) == process.pid

        monitor(
            process,
            str(tmp_path),
            poll_interval=0.01,
            timeout="00:00:01",
        )

        _wait_until(lambda: not _process_is_active(grandchild_pid))
        assert process.returncode == -signal.SIGKILL
        assert not _process_is_active(grandchild_pid)
    finally:
        if process.poll() is None:
            process.kill()
        process.cleanup_logging()
        if grandchild_pid is not None and _process_is_active(grandchild_pid):
            os.kill(grandchild_pid, signal.SIGKILL)
