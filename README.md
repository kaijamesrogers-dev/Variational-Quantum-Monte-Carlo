# Computational Physics Project – Reproducibility Instructions

This folder contains the Python code used to generate all figures and numerical
results presented in my Computational Physics lab report.

A single driver script (`run_all.py`) is provided so that all figures can be
reproduced easily for marking.

---

## Files

- `task_2_1.py`  
- `task_2_2.py`  
- `task_3_1.py`  
- `task_4_1.py`  
  Core project scripts. Each corresponds to a task in the project specification
  and generates figures shown in the report.

- `run_all.py`  
  Runs all task files sequentially and shows all plots.

- `README.md`  
  This file.

---

## How to reproduce all figures

From a terminal in this directory, run:

```bash
python run_all.py

#Code may take a while to run, especially task_4.1.py
#ChatGPT was used to assist in writing this README.md file, and the run_all.py driver script.
