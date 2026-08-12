"""CLI script for dataset generation at scale."""

import argparse
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rf_sim.dataset_generator import DatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large-scale windowed RF sensing dataset.")
    parser.add_argument("--num-scenarios", type=int, default=200, help="Number of scenarios to simulate (default 200).")
    parser.add_argument("--out", type=str, default="data/dataset.parquet", help="Output Parquet path (default data/dataset.parquet).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default 42).")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train split fraction (default 0.7).")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation split fraction (default 0.15).")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test split fraction (default 0.15).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    os.makedirs(out_dir, exist_ok=True)

    print(f"--- Starting Dataset Generation ({args.num_scenarios} scenarios, seed={args.seed}) ---")
    generator = DatasetGenerator(base_seed=args.seed)
    df = generator.generate_dataset(
        num_scenarios=args.num_scenarios,
        duration_per_scenario_s=5.0,
        sample_rate_hz=50.0,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    # Save full dataset
    df.to_parquet(out_path, index=False)
    print(f"Saved full dataset ({len(df)} rows) to {out_path}")

    # Save individual split files
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"  - Train split: {len(train_df)} rows ({train_path})")
    print(f"  - Val split:   {len(val_df)} rows ({val_path})")
    print(f"  - Test split:  {len(test_df)} rows ({test_path})")
    print("--- Dataset Generation Complete ---")


if __name__ == "__main__":
    main()
