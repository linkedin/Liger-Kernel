"""Observe benchmark threads without changing the benchmark process."""

import json
import sys
import time

from pathlib import Path

pid = int(sys.argv[1])
process = Path("/proc") / str(pid)
last = None
states = []
while process.exists():
    threads = []
    try:
        for task in (process / "task").iterdir():
            try:
                status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
            except (FileNotFoundError, ProcessLookupError):
                continue
            name = status["Name"].strip()
            if int(task.name) == pid or name in {"acl_thread", "release_thread", "pt_autograd_0"}:
                threads.append({"tid": int(task.name), "name": name, "cpus": status["Cpus_allowed_list"].strip()})
    except (FileNotFoundError, ProcessLookupError):
        break
    threads.sort(key=lambda row: row["tid"])
    if threads != last:
        states.append({"time_ns": time.time_ns(), "threads": threads})
        last = threads
    time.sleep(0.25)
print(json.dumps({"pid": pid, "states": states}, indent=2))
