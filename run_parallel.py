import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
]

# makes sure no experiments are rerun if they are already run.
# gets used if a first run can't complete all experiments.
def result_is_complete(env_name, seed, expected_evaluations):
    result_path = ROOT / "results" / f"TD3_{env_name}_{seed}.npy"
    if not result_path.exists():
        return False

    try:
        evaluations = np.load(result_path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return False

    return evaluations.ndim == 1 and len(evaluations) == expected_evaluations

# runs 1 job (1 env, 1 seed)
def run_job(env_name, seed, max_timesteps, eval_freq):
    name = f"TD3_{env_name}_{seed}"
    run_dir = ROOT / ".parallel_runs" / name
    staged_results = run_dir / "results"
    staged_result = staged_results / f"{name}.npy"
    final_result = ROOT / "results" / f"{name}.npy"
    log_path = ROOT / "logs" / f"{name}.log"

    staged_results.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(ROOT / "main.py"),
        "--policy", "TD3",
        "--env", env_name,
        "--seed", str(seed),
        "--max_timesteps", str(max_timesteps),
        "--eval_freq", str(eval_freq),
    ]

    if env_name in {
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "Reacher-v5",
    }:
        command += ["--start_timesteps", "1000"]

    child_env = os.environ.copy()
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["MKL_NUM_THREADS"] = "1"
    child_env["OPENBLAS_NUM_THREADS"] = "1"

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\nStarting {name}\n")
        subprocess.run(
            command,
            cwd=run_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
        )

    expected_evaluations = 1 + max_timesteps // eval_freq
    evaluations = np.load(staged_result, allow_pickle=False)
    if evaluations.ndim != 1 or len(evaluations) != expected_evaluations:
        raise RuntimeError(
            f"{name} produced {len(evaluations)} evaluations; "
            f"expected {expected_evaluations}"
        )

    os.replace(staged_result, final_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_timesteps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    # this part about eval_freq and max_timesteps is only needed for result_is_complete func and could otherwise be removed
    if args.eval_freq < 1:
        parser.error("--eval_freq must be at least 1")
    if args.max_timesteps < args.eval_freq:
        parser.error("--max_timesteps must be at least --eval_freq")

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    expected_evaluations = 1 + args.max_timesteps // args.eval_freq
    jobs = [
        (env, seed)
        for seed in range(10)
        for env in ENVIRONMENTS
        if not result_is_complete(env, seed, expected_evaluations)
    ]

    total_jobs = len(ENVIRONMENTS) * 10
    print(f"Skipping {total_jobs - len(jobs)} complete runs; scheduling {len(jobs)}.")
    # handle failures such that all jobs at least start once
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_job, env, seed, args.max_timesteps, args.eval_freq
            ): (env, seed)
            for env, seed in jobs
        }

        for completed, future in enumerate(as_completed(futures), start=1):
            env, seed = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((env, seed))
                print(
                    f"FAILED {env} seed {seed}: {exc} "
                    f"(see logs/TD3_{env}_{seed}.log)",
                    file=sys.stderr,
                )
            else:
                print(f"Completed {completed}/{len(futures)}: {env} seed {seed}")

    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed; see their logs.")


if __name__ == "__main__":
    main()