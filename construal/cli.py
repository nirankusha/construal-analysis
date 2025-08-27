import argparse
from .pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to selected.parquet")
    ap.add_argument("--out", dest="out_dir", required=True, help="Directory for artifacts")
    ap.add_argument("--steps", nargs="*", help="Steps to run, e.g., 1 2 3 4 5 6 7")
    ap.add_argument("--all", action="store_true", help="Run all steps")
    args = ap.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
