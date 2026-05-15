#!/bin/bash
# Master runner: supply (GPU) → eval (GPT API) → analysis
# Usage: nohup bash run_all.sh > ~/run_history_shock.log 2>&1 &

set -e

echo "=== History-Shock Pipeline ==="
echo "Start: $(date)"
echo ""

# Step 1: Supply generation (vLLM on GPU)
echo "--- Step 1: Supply (vLLM) ---"
python supply_history_shock.py --full

# Step 2: GPT evaluation (absolute + pairwise)
echo ""
echo "--- Step 2: GPT Evaluation ---"
python eval_gpt.py --all --resume

# Step 3: Analysis
echo ""
echo "--- Step 3: Analysis ---"
python analyze.py

echo ""
echo "=== Pipeline Complete ==="
echo "End: $(date)"
echo "Results: $DATA_DIR"
