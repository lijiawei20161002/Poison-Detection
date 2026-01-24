#!/bin/bash
cd /mnt/nw/home/j.li/Poison-Detection
python3 experiments/compute_qwen_minimal.py 2>&1 | tee qwen_diagonal_run.log
