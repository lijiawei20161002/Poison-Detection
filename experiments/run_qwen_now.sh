#!/bin/bash
cd /mnt/nw/home/j.li/Poison-Detection
python3 experiments/complete_tiny_experiments.py --model qwen-small 2>&1 | tee qwen_final_$(date +%Y%m%d_%H%M%S).log
