# Library Imports
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

def write_json(path, data):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)

def main(args):
    args.results_dir.mkdir(parents=True, exist_ok=True)
    active = []
    stopping = threading.Event()
    def stop(signum, frame):
        stopping.set()
        for process in active:
            if process.poll() is None:
                process.terminate()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    jobs = [(dataset, algorithm, fraction) for fraction in (.1, .5, 1.)
            for dataset in args.datasets for algorithm in ("cql_awr", "iql_awr", "td3_bc")]
    state_path = args.results_dir / "grid_state.json"
    state = {"status": "running", "workers": args.workers, "jobs": len(jobs), "seed": 0,
             "datasets": args.datasets, "profile": args.profile, "policies": 3 * len(jobs),
             "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "failed": []}
    with (args.results_dir / ".grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(state_path, state)

        def run_family(job):
            dataset, algorithm, fraction = job
            name = f"{dataset}_{algorithm}_c{int(fraction*100)}_seed0"
            if stopping.is_set():
                return name, -15
            directory = args.results_dir / name
            directory.mkdir(exist_ok=True)
            profile = "kitchen_v2" if args.profile == "submission" and dataset == "kitchen" else ("baseline" if args.profile == "submission" else args.profile)
            complete = directory / "complete.json"
            if complete.exists():
                config = json.loads((directory / "config.json").read_text())
                if config.get("profile", "baseline") != profile:
                    raise ValueError(f"Profile mismatch in {name}.")
                saved = json.loads(complete.read_text())
                if saved["total_steps_per_policy"] == 1000000 and set(saved["policies"]) == {"p10", "p50", "p100"}:
                    return name, 0
            command = [sys.executable, "-u", "train_decoupled.py", "--dataset", dataset, "--algorithm", algorithm,
                       "--critic-fraction", str(fraction), "--results-dir", str(args.results_dir),
                       "--checkpoint-dir", str(args.checkpoint_dir), "--data-dir", str(args.data_dir), "--profile", profile]
            print(f"Starting {name}", flush=True)
            before = time.monotonic()
            with (directory / "console.log").open("a") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                active.append(process)
                if stopping.is_set():
                    process.terminate()
                write_json(directory / "job_status.json", {"status": "running", "pid": process.pid,
                           "started_at": datetime.now(timezone.utc).isoformat()})
                code = process.wait()
            write_json(directory / "job_status.json", {"status": "complete" if code == 0 else "failed",
                       "returncode": code, "seconds": time.monotonic()-before,
                       "finished_at": datetime.now(timezone.utc).isoformat()})
            print(f"Finished {name} exit {code} seconds {round(time.monotonic()-before)}", flush=True)
            return name, code

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_family, job) for job in jobs]
                try:
                    for future in as_completed(futures):
                        name, code = future.result()
                        state["completed" if code == 0 else "failed"].append(name)
                        write_json(state_path, state)
                except BaseException:
                    stopping.set()
                    for future in futures:
                        future.cancel()
                    for process in active:
                        if process.poll() is None:
                            process.terminate()
                    raise
            if state["failed"]:
                raise RuntimeError(f"Failed families: {state['failed']}; inspect their console.log files and rerun to resume.")
            with (args.results_dir / "reporting.log").open("w") as handle:
                command = [sys.executable, "-u", "report_decoupled.py", "--results-dir", str(args.results_dir),
                           "--checkpoint-dir", str(args.checkpoint_dir), "--datasets", *args.datasets]
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            state["status"] = "complete"
        except BaseException:
            state["status"] = "interrupted" if any(process.poll() is None for process in active) else "failed"
            for process in active:
                if process.poll() is None:
                    process.terminate()
            raise
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            write_json(state_path, state)
    print("Full decoupled grid and reports complete.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the submitted 18 critic families and 54 extracted policies.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RL_WORKERS", 4)))
    parser.add_argument("--profile", choices=("baseline", "kitchen_v2", "submission"), default="submission")
    parser.add_argument("--datasets", nargs="+", choices=("kitchen", "hopper"))
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.datasets = args.datasets or (["kitchen"] if args.profile == "kitchen_v2" else ["hopper", "kitchen"])
    if args.profile == "kitchen_v2" and args.datasets != ["kitchen"]:
        parser.error("The kitchen_v2 profile runs Kitchen only.")
    folder = "kitchen_v2" if args.profile == "kitchen_v2" else "decoupled"
    args.results_dir = args.results_dir or Path("results") / folder
    args.checkpoint_dir = args.checkpoint_dir or Path("checkpoints") / folder
    if not 1 <= args.workers <= 8:
        parser.error("Use between 1 and 8 workers.")
    main(args)
