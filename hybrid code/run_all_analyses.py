import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a shell command and report success/failure."""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{description} failed (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"\n{description} failed: {e}")
        return False


def check_required_scripts(script_dir: Path, scripts: list[str]) -> bool:
    """Verify every required script exists before kicking off the run."""
    missing = [s for s in scripts if not (script_dir / s).exists()]
    if missing:
        print("\nThe following required scripts were not found:")
        for s in missing:
            print(f"  - {script_dir / s}")
        print(
            "\nMake sure all scripts are in the same directory as "
            "run_all_analyses.py before running."
        )
        return False
    return True


def main():
    script_dir = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(
        description=(
            "Run the analysis pipeline: calculate_hybrid_scores.py, then "
            "generate_summary_statistics.py, plus optional follow-up scripts."
        ),
        epilog="""
Examples:
  Run everything:
    python run_all_analyses.py . --run-all-analyses

  Run only the essentials (scores + summary statistics):
    python run_all_analyses.py .

  Run for a single dataset:
    python run_all_analyses.py . --run-all-analyses --dataset 1096_FacultySalaries

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(script_dir),
        help="Root directory containing dataset result folders",
    )

    # Hybrid score calculation options
    parser.add_argument("--systematic-exploration", action="store_true",
                        help="Use systematic lambda exploration")
    parser.add_argument("--lambda1-step", type=float, default=0.1,
                        help="Step size for lambda1")
    parser.add_argument("--lambda2-step", type=float, default=0.1,
                        help="Step size for lambda2")
    parser.add_argument("--lambda3-step", type=float, default=0.1,
                        help="Step size for lambda3")
    parser.add_argument("--num-points", type=int, default=None,
                        help="Number of evenly spaced points per lambda")
    parser.add_argument("--no-detailed-results", action="store_true",
                        help="Skip saving detailed CSV files")
    parser.add_argument("--clamp-percentile", type=float, default=70.0,
                        help="(Unused, kept for compatibility)")
    parser.add_argument("--absolute-max-cap", type=float, default=1000.0,
                        help="(Unused, kept for compatibility)")
    parser.add_argument("--constant-clamp-value", type=float, default=10.0,
                        help="Upper clamp value applied to Extrapolation_Divergence "
                             "and Sensitivity metrics (default 10.0).")

    # Follow-up analysis options
    parser.add_argument("--skip-summary-stats", action="store_true",
                        help="Skip generating summary statistics")
    parser.add_argument("--compare-correlations", action="store_true",
                        help="Compare correlations against baseline")
    parser.add_argument("--run-all-analyses", action="store_true",
                        help="Run every available follow-up analysis")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Restrict processing to a single dataset")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Parallel workers for dataset processing")

    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()

    print("=" * 80)
    print("Running All Analyses")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Script directory:  {script_dir}\n")

    required = ["calculate_hybrid_scores.py", "generate_summary_statistics.py"]
    run_correlations = args.compare_correlations or args.run_all_analyses
    if run_correlations:
        required.append("compare_correlations_baseline.py")

    if not check_required_scripts(script_dir, required):
        sys.exit(1)

    # Step 1: hybrid scores
    cmd1 = [sys.executable, str(script_dir / "calculate_hybrid_scores.py"), str(results_dir)]

    if args.dataset:
        cmd1.extend(["--dataset", args.dataset])

    if args.systematic_exploration:
        cmd1.append("--systematic-exploration")
        if args.num_points is not None:
            cmd1.extend(["--num-points", str(args.num_points)])
        else:
            cmd1.extend(["--lambda1-step", str(args.lambda1_step)])
            cmd1.extend(["--lambda2-step", str(args.lambda2_step)])
            cmd1.extend(["--lambda3-step", str(args.lambda3_step)])

    if args.no_detailed_results:
        cmd1.append("--no-detailed-results")

    if args.num_workers is not None:
        cmd1.extend(["--num-workers", str(args.num_workers)])

    cmd1.extend(["--clamp-percentile", str(args.clamp_percentile)])
    cmd1.extend(["--absolute-max-cap", str(args.absolute_max_cap)])
    cmd1.extend(["--constant-clamp-value", str(args.constant_clamp_value)])

    success1 = run_command(cmd1, "Step 1: Calculate Hybrid Scores")
    if not success1:
        print("\n" + "=" * 80)
        print("Step 1 failed. Stopping.")
        print("=" * 80)
        sys.exit(1)

    # Step 2: summary statistics
    if not args.skip_summary_stats:
        cmd2 = [sys.executable, str(script_dir / "generate_summary_statistics.py"), str(results_dir)]
        if args.dataset:
            cmd2.extend(["--dataset", args.dataset])

        success2 = run_command(cmd2, "Step 2: Generate Summary Statistics")
        if not success2:
            print("\nSummary statistics step failed, continuing anyway.")
    else:
        print("\nSkipping summary statistics generation.")

    # Step 3: correlation comparison (optional)
    if run_correlations:
        cmd3 = [
            sys.executable,
            str(script_dir / "compare_correlations_baseline.py"),
            str(results_dir),
            "--create-plots",
        ]
        if args.dataset:
            cmd3.extend(["--dataset", args.dataset])
        success3 = run_command(cmd3, "Step 3: Compare Correlations Against Baseline")
        if not success3:
            print("\nCorrelation comparison failed, continuing anyway.")

    print("\n" + "=" * 80)
    print("All Analyses Complete")
    print("=" * 80)
    print(f"\nResults available in:")
    print(f"  hybrid_results/")

    if not args.skip_summary_stats:
        print(f"    - clamp_value_summary.csv")
        print(f"    - normalisation_summary.csv, *_SVP.csv, *_MVP.csv")
        print(f"    - metric_distributions_after_normalisation.csv, *_SVP.csv, *_MVP.csv")
        print(f"    - best_hybrid_overall.csv, best_hybrid_overall_SVP.csv, best_hybrid_overall_MVP.csv")
        print(f"    - best_hybrid_interpolation.csv, *_SVP.csv, *_MVP.csv")
        print(f"    - best_hybrid_extrapolation.csv, *_SVP.csv, *_MVP.csv")
        print(f"    - hybrid_comparison_results.csv, *_SVP.csv, *_MVP.csv")

    if run_correlations:
        print(f"  correlation_comparisons/")
    print()


if __name__ == "__main__":
    main()
