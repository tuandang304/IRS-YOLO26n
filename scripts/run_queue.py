"""Run a queue of jobs (training, prediction, latency) on the local GPU.

    python scripts/run_queue.py configs/queues/f_p_704.json            # all jobs in the queue
    python scripts/run_queue.py configs/queues/r5_vm_a.json --only p_704_seed1

Each entry of a queue file is a job spec read by src/job_main.py, plus three keys used here:
  inputs   {"inputs/<name>": "<path>"} files copied into the job workspace (paths relative to the
           repository root, e.g. a teacher checkpoint produced by an earlier job)
  chunks   list of stop_after_epoch values for kind=train (null = run to the end)
  timeout  seconds per chunk (ignored here; kept for reference)

Each job keeps one workspace under runs/remote/<name>/ws (the layout expected by the scoring and
table scripts). Training resumes from out/<name>/weights/last.pt between chunks, as in the paper's
runs, which were trained in 15-epoch chunks on an NVIDIA L4. The DroneVehicle data are downloaded and
prepared once under $DV_DATA (default: data/dronevehicle).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "remote"
SRC_FILES = ["lowlight.py", "dv_prep.py", "dv_pairs.py", "job_main.py", "zdce.py", "evaluate.py", "dfine_job.py"]
STUB = "import sys\nsys.path.insert(0, 'src')\nimport job_main\njob_main.main('job.json')\n"


def run_chunk(spec, stop):
    name = spec["name"]
    ws = RUNS / name / "ws"
    (ws / "src").mkdir(parents=True, exist_ok=True)
    for f in SRC_FILES:
        shutil.copy2(ROOT / "src" / f, ws / "src" / f)
    for dst, src in spec.get("inputs", {}).items():
        (ws / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src if Path(src).is_absolute() else ROOT / src, ws / dst)
    job = {k: v for k, v in spec.items() if k not in ("inputs", "chunks", "timeout")}
    if spec["kind"] == "train":
        job["config"] = dict(job.get("config", {}), stop_after_epoch=stop)
    (ws / "job.json").write_text(json.dumps(job, indent=1))
    env = dict(os.environ)
    env.setdefault("DV_DATA", str(ROOT / "data" / "dronevehicle"))
    log = RUNS / name / f"chunk_{stop}_{time.strftime('%m%d-%H%M%S')}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([sys.executable, "-c", STUB], cwd=ws, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(json.dumps({"name": name, "chunk": stop, "secs": round(time.time() - t0), "rc": rc, "log": str(log)}),
          flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--only", nargs="*", help="run only the jobs with these names")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    for spec in json.loads(Path(a.queue).read_text()):
        if a.only and spec["name"] not in a.only:
            continue
        for stop in spec.get("chunks", [None]) if spec["kind"] == "train" else [None]:
            if not run_chunk(spec, stop):
                print(f"STOP: {spec['name']} chunk {stop} failed", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
