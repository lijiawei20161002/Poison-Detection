#!/bin/bash
cd /mnt/nw/home/j.li/Poison-Detection
/usr/bin/python3 experiments/complete_tiny_experiments.py --model both 2>&1 | tee completion_log_$(date +%Y%m%d_%H%M%S).log
