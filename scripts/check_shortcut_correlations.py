"""Script to analyze and check for spurious shortcut correlations in generated dataset."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, ttest_ind


def main() -> None:
    data_path = "data/dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generate_dataset.py first.")
        return

    df = pd.read_parquet(data_path)

    print("==========================================================")
    print("      SPURIOUS SHORTCUT CORRELATION VERIFICATION REPORT   ")
    print("==========================================================")

    # 1. TX-RX Distance vs Person Presence
    dist_present = df[df["person_present"] == True]["tx_rx_distance"].values
    dist_absent = df[df["person_present"] == False]["tx_rx_distance"].values

    mean_pres_dist = float(np.mean(dist_present))
    mean_abs_dist = float(np.mean(dist_absent))
    mean_diff = abs(mean_pres_dist - mean_abs_dist)

    ks_stat, p_val_dist = ks_2samp(dist_present, dist_absent)
    t_stat, p_val_ttest = ttest_ind(dist_present, dist_absent, equal_var=False)

    print(f"\n1. TX-RX Distance vs. Person Presence:")
    print(f"   - Mean Distance (Present): {mean_pres_dist:.3f} m")
    print(f"   - Mean Distance (Absent):  {mean_abs_dist:.3f} m")
    print(f"   - Absolute Mean Diff:      {mean_diff:.3f} m ({mean_diff/np.mean(df['tx_rx_distance'])*100:.2f}% of mean)")
    print(f"   - KS Test p-value:         {p_val_dist:.4f}")
    print(f"   - Welch's t-test p-value:  {p_val_ttest:.4f}")

    if p_val_dist > 0.05:
        print("   -> [STATUS: PASSED] No statistically significant distance bias between present/absent.")
    else:
        print("   -> [WARNING] Statistically significant distance correlation detected!")

    # 2. Wall Material vs Person Presence
    contingency_table = pd.crosstab(df["person_present"], df["wall_material_primary"])
    chi2_stat, p_val_chi2, dof, _ = chi2_contingency(contingency_table)

    print(f"\n2. Primary Wall Material vs. Person Presence:")
    print("   Contingency Table:")
    print(contingency_table)
    print(f"   - Chi-Square Test p-value: {p_val_chi2:.4f}")

    if p_val_chi2 > 0.05:
        print("   -> [STATUS: PASSED] Wall material is independent of person presence label.")
    else:
        print("   -> [WARNING] Wall material correlation detected!")

    print("\n==========================================================")


if __name__ == "__main__":
    main()
