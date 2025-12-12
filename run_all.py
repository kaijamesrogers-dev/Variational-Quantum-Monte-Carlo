"""
run_all.py

Runs all Computational Physics project task files sequentially.
All figures are:
  - shown on screen (plt.show())

This file is for reproducibility and marking purposes.
"""

import runpy
import os
import sys
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TASK_FILES = [
    "task_2_1.py",
    "task_2_2.py",
    "task_3_1.py",
    "task_4_1.py",
]

SAVE_FIGS = True
FIG_DPI = 300

# ------------------------------------------------------------
# Helper: save all currently open figures
# ------------------------------------------------------------

def save_all_figures(prefix):
    fig_nums = plt.get_fignums()
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        filename = f"{prefix}_fig{i:02d}.png"
        fig.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  saved: {filename}")

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

def main():
    print("\n=== Running all project tasks ===\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    for task in TASK_FILES:
        print(f"\n--- Running {task} ---")

        if not os.path.exists(task):
            print(f"ERROR: {task} not found")
            continue

        # Clear old figures
        plt.close("all")

        # Run the script as if called directly
        runpy.run_path(task, run_name="__main__")

        # Save figures AFTER they are created
        if SAVE_FIGS:
            prefix = os.path.splitext(task)[0]
            save_all_figures(prefix)

        print(f"--- Finished {task} ---")

    print("\n=== All tasks completed ===\n")
    print("Figures have been shown and saved in this directory.")

if __name__ == "__main__":
    main()
