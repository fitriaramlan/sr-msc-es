import os
import time
import argparse
import numpy as np
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Environment configuration to prevent thread oversubscription
# These settings ensure each worker process uses a single thread for numerical libraries
os.environ.setdefault("MPLBACKEND", "Agg")           # Non-interactive plotting
os.environ.setdefault("OMP_NUM_THREADS", "1")        # OpenMP
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # OpenBLAS
os.environ.setdefault("MKL_NUM_THREADS", "1")        # Intel MKL
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")    # NumExpr
os.environ.setdefault("JULIA_NUM_THREADS", "1")      # Julia (for PySR)
os.environ.setdefault("JAX_PLATFORMS", "cpu")       # JAX CPU only (prevents GPU/CUDA init in multiprocessing)

# This batch of datasets with small number of samples and features
DATASETS = [   
    # "1096_FacultySalaries.npy",
    # "192_vineyard.npy",
    # "228_elusage.npy",
    # "542_pollution.npy",
    # "523_analcatdata_neavote.npy",
    # "678_visualizing_environmental.npy",
    # "229_pwLinear.npy",
    # "561_cpu.npy",
    # "712_chscase_geyser1.npy",
    # "605_fri_c2_250_25.npy",
    # "519_vinnie.npy",
    # "556_analcatdata_apnea2.npy",
    # "557_analcatdata_apnea1.npy",
    # "522_pm10.npy",
    # "584_fri_c4_500_25.npy",
    # "637_fri_c1_500_50.npy",
    # "589_fri_c2_1000_25.npy",
    # "1028_SWD.npy",
    # "227_cpu_small.npy",
    # "505_tecator.npy",
]

# KDE bandwidth parameters to test
BANDWIDTHS = [0.3]

# Experimental parameters
NUM_RUNS = 30          # Number of repeated runs for each dataset
USE_CV = False        # True: use K-fold CV; False: single holdout split
N_SPLITS = 3          # Number of CV folds (ignored if USE_CV=False)
# Run range controls (inclusive start, inclusive end). Set END to None to use NUM_RUNS.
DEFAULT_START_RUN = 1
DEFAULT_END_RUN: int | None = 30

# Paths
DATA_DIR = Path("/home/fitria_w/hybrid_var/datasets")
OUT_ROOT = Path("/home/fitria_w/eurogp/results/")
# DATA_DIR = Path("../../datasets")
# OUT_ROOT = Path("../../results/outputs-IA")


def run_one_dataset(dataset_filename: str, start_run: int = 1, end_run: int | None = None) -> str:
    """Process a single dataset through KDE analysis and model evaluation"""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    # Configure JAX to use CPU only to avoid GPU/CUDA initialisation problems in multiprocessing
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from kde_analysis import kde_analysis
    from m1_evaluation import evaluate_models

    # Setup paths and logging
    input_file = DATA_DIR / dataset_filename
    dataset_name = Path(dataset_filename).stem
    output_dir = OUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "worker.log"

    start = time.time()
    with open(log_path, "a", buffering=1) as log:
        log.write(f"START: {dataset_name}\n")

    # KDE-based region separation (NaN handling: removes rows with any NaN values)
    kde_analysis(str(input_file), str(output_dir))

    # Infer feature names from data shape
    sample = np.load(str(input_file))
    n_features = sample.shape[1] - 1
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

    # Model training and evaluation
    evaluate_models(
        bandwidths=BANDWIDTHS,
        feature_names=feature_names,
        output_dir=str(output_dir),
        num_runs=NUM_RUNS,
        use_cv=USE_CV,
        n_splits=N_SPLITS,
        start_run=start_run,
        end_run=end_run,
    )

    # Format elapsed time
    elapsed = int(time.time() - start)
    h, r = divmod(elapsed, 3600)
    m, s = divmod(r, 60)
    msg = f"Completed '{dataset_name}' in {h}h {m}m {s}s"

    with open(log_path, "a") as log:
        log.write(msg + "\n")
        log.write(f"\n{'='*60}\n")
        log.write(f"DATASET: {dataset_name}\n")
        log.write(f"Original size: {sample.shape[0]} samples\n")
        log.write(f"Features: {n_features}\n")
        log.write(f"NaN handling method: drop rows (removes samples with any NaN values)\n")
        log.write(f"Cross-validation: {'Yes (' + str(N_SPLITS) + ' folds)' if USE_CV else 'No (single split)'}\n")
        log.write(f"Number of runs: {NUM_RUNS}\n")
        log.write(f"Bandwidths tested: {BANDWIDTHS}\n")
        log.write(f"Total time: {h}h {m}m {s}s\n")
        log.write(f"{'='*60}\n")

    return msg


def validate_and_list_datasets(names: list[str]) -> tuple[list[str], list[str]]:
    """Validate dataset file existence and return (existing, missing) lists"""
    existing, missing = [], []
    for name in names:
        p = DATA_DIR / name
        if p.exists():
            existing.append(name)
        else:
            missing.append(name)
    
    print(f"DATASETS list length: {len(names)}")
    print(f"Found on disk: {len(existing)} | Missing: {len(missing)}")
    for m in missing:
        print(f"File: {DATA_DIR / m} is missing")
    
    return existing, missing


def main(
    workers: int | None,
    start_run: int = DEFAULT_START_RUN,
    end_run: int | None = DEFAULT_END_RUN,
):
    """Run parallel dataset processing with multiprocessing"""
    mp.set_start_method("spawn", force=True)

    # Default to half of CPU cores (conservative for compute-intensive tasks)
    if not workers or workers <= 0:
        workers = max(1, (os.cpu_count() or 2) // 2)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Validate dataset files before processing
    existing, missing = validate_and_list_datasets(DATASETS)
    if not existing:
        print("No valid datasets found. Check paths/filenames.")
        return

    print(f"Starting parallel run with {workers} workers using SPAWN")
    start_all = time.time()

    # Submit all datasets as independent jobs
    ctx = mp.get_context("spawn")
    futures = {}
    completed, failed = [], []
    
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for ds in existing:
            futures[ex.submit(run_one_dataset, ds, start_run, end_run)] = ds

        # Collect results as they complete
        for fut in as_completed(futures):
            ds = futures[fut]
            try:
                print(fut.result(), flush=True)
                completed.append(ds)
            except Exception as e:
                print(f"[ERROR] Dataset failed: {ds} -> {e}", flush=True)
                # Log error to dataset-specific worker.log
                ddir = OUT_ROOT / Path(ds).stem
                ddir.mkdir(parents=True, exist_ok=True)
                with open(ddir / "worker.log", "a") as log:
                    import traceback
                    log.write(f"\nERROR:\n{traceback.format_exc()}\n")
                failed.append(ds)

    # Format total elapsed time
    total = int(time.time() - start_all)
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)

    # Print execution summary
    print("\nSUMMARY")
    print(f"Scheduled: {len(existing)} (from list of {len(DATASETS)})")
    print(f"Completed: {len(completed)}")
    print(f"Failed:    {len(failed)}")
    print(f"Missing:   {len(missing)}")
    if failed:
        print("Failed datasets:", failed)
    if missing:
        print("Missing files:", missing)

    print(f"\nAll scheduled datasets finished in {h}h {m}m {s}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run datasets in parallel.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes (default: ~half of CPU cores).",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=DEFAULT_START_RUN,
        help="First run index (1-based) to execute/resume.",
    )
    parser.add_argument(
        "--end-run",
        type=int,
        default=DEFAULT_END_RUN,
        help="Last run index (1-based, inclusive) to execute. Defaults to num_runs.",
    )
    args = parser.parse_args()
    if args.start_run < 1:
        raise ValueError("--start-run must be >= 1")
    if args.end_run is not None and args.end_run < args.start_run:
        raise ValueError("--end-run must be >= --start-run")
    main(args.workers, args.start_run, args.end_run)
