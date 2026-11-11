"""Launch one CPU controller with an explicit, repeatable macOS resource policy."""

import ctypes
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def worker(run: Path, label: str) -> None:
    # Record the actual inherited class before JAX or BLAS creates its threads.
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.qos_class_self.argtypes = []
    library.qos_class_self.restype = ctypes.c_uint
    qos = int(library.qos_class_self())
    if qos != 0x15:  # QOS_CLASS_DEFAULT, provided by Interactive launchd jobs.
        raise RuntimeError(f"Unexpected CPU controller QoS class: {qos}")
    receipt_path = run / "cpu-launch.json"
    history = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    history.append(
        dict(
            started_at=datetime.now(UTC).isoformat(),
            pid=os.getpid(),
            launch_method="launchd",
            process_type="Interactive",
            qos_class=qos,
            launchd_label=os.environ["XPC_SERVICE_NAME"],
            source_commit=json.loads((run / "plan.json").read_text())["source_commit"],
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            worker_sha256=hashlib.sha256((run / "cpu.sh").read_bytes()).hexdigest(),
        )
    )
    receipt_path.write_text(json.dumps(history, indent=2) + "\n")
    (run / "cpu.pid").write_text(str(os.getpid()) + "\n")
    os.execv("/bin/bash", ["bash", str(run / "cpu.sh"), str(run), label])


def launch(run: Path, python: str, label: str) -> None:
    job = "local.gpex.profiling." + hashlib.sha256(str(run).encode()).hexdigest()[:12]
    domain = f"gui/{os.getuid()}"
    target = f"{domain}/{job}"
    existing = subprocess.run(
        ["launchctl", "print", target], capture_output=True, text=True
    )
    if existing.returncode == 0:
        if "\n\tstate = running\n" in existing.stdout:
            raise RuntimeError(f"CPU controller already running: {target}")
        subprocess.run(["launchctl", "bootout", target], check=True)
    frozen = run / "cpu-launch.py"
    shutil.copyfile(__file__, frozen)
    plist = run / "cpu.plist"
    plist.write_bytes(
        plistlib.dumps(
            dict(
                Label=job,
                ProgramArguments=[python, str(frozen), "--worker", str(run), label],
                WorkingDirectory=str(run / "source"),
                RunAtLoad=True,
                KeepAlive=False,
                # Detach from the terminal without switching to utility/background QoS.
                # Never respawn a completed or failed campaign automatically.
                ProcessType="Interactive",
                StandardOutPath=str(run / "cpu.log"),
                StandardErrorPath=str(run / "cpu.log"),
            )
        )
    )
    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    (run / "cpu-launchd-label").write_text(job + "\n")
    print(f"CPU controller submitted: {target}\nCampaign: {run}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(Path(sys.argv[2]), sys.argv[3])
    else:
        launch(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3])
