"""CLI script for training and evaluating classical ML models across all 4 RF sensing tasks."""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rf_sim.ml.data_loading import load_task_data
from rf_sim.ml.train import train_and_evaluate_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML models for all 4 RF sensing tasks.")
    parser.add_argument("--train-path", type=str, default="data/train.parquet", help="Train split Parquet path.")
    parser.add_argument("--val-path", type=str, default="data/val.parquet", help="Val split Parquet path.")
    parser.add_argument("--test-path", type=str, default="data/test.parquet", help="Test split Parquet path.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save trained model artifacts.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation reports.")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    tasks = ["task_1_presence", "task_2_movement", "task_3_direction", "task_4_zone"]
    summary_rows = []

    print("=================================================================")
    print("      RF SENSING CLASSICAL ML PIPELINE - TRAINING ALL TASKS      ")
    print("=================================================================\n")

    for task_name in tasks:
        print(f"--- Training Task: {task_name} ---")
        task_data = load_task_data(
            task_name=task_name,
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
        )

        res = train_and_evaluate_task(
            task_name=task_name,
            task_data=task_data,
            model_dir=args.model_dir,
        )

        winner = res["winning_model_name"]
        test_acc = res["test_metrics"]["accuracy"]
        test_f1 = res["test_metrics"]["f1"]
        test_prec = res["test_metrics"]["precision"]
        test_rec = res["test_metrics"]["recall"]

        print(f"  -> Winner: {winner}", flush=True)
        print(f"  -> Validation Macro-F1s: " + ", ".join([f"{m}: {res['val_results'][m]['f1']:.4f}" for m in res['val_results']]), flush=True)
        print(f"  -> FINAL TEST METRICS (Evaluated ONCE on {len(task_data.y_test)} test samples):", flush=True)
        print(f"     Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | Macro-F1: {test_f1:.4f}\n", flush=True)

        # Collect summary rows per model type
        for model_name, val_m in res["val_results"].items():
            is_winner = (model_name == winner)
            summary_rows.append({
                "task": task_name,
                "model": model_name,
                "is_winner": is_winner,
                "val_accuracy": val_m["accuracy"],
                "val_precision": val_m["precision"],
                "val_recall": val_m["recall"],
                "val_f1_macro": val_m["f1"],
                "test_accuracy": test_acc if is_winner else None,
                "test_precision": test_prec if is_winner else None,
                "test_recall": test_rec if is_winner else None,
                "test_f1_macro": test_f1 if is_winner else None,
                "best_params": str(val_m["best_params"]),
            })

    results_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.results_dir, "task_results.csv")
    md_path = os.path.join(args.results_dir, "task_results.md")

    results_df.to_csv(csv_path, index=False)
    try:
        results_df.to_markdown(md_path, index=False)
    except (ImportError, ModuleNotFoundError):
        with open(md_path, "w") as f:
            f.write(f"| {' | '.join(results_df.columns)} |\n")
            f.write(f"| {' | '.join(['---'] * len(results_df.columns))} |\n")
            for _, row in results_df.iterrows():
                f.write(f"| {' | '.join(str(val) for val in row.values)} |\n")

    print(f"Saved results summary table to {csv_path} and {md_path}", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
