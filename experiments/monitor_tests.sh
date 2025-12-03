#!/bin/bash
# Monitor transformation testing progress

LOG_FILE="experiments/results/polarity_all_transforms_run.log"
RESULT_DIR="experiments/results"

echo "================================================"
echo "TRANSFORMATION TEST PROGRESS MONITOR"
echo "================================================"
echo

# Check if tests are running
if pgrep -f "test_all_transforms.py" > /dev/null; then
    echo "✅ Tests are currently RUNNING"
else
    echo "❌ No tests currently running"
fi

echo
echo "Latest Log Entries:"
echo "-------------------"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "Log file not found: $LOG_FILE"
fi

echo
echo "================================================"
echo "Completed Test Files:"
echo "================================================"

# Count completed polarity tests
COMPLETED=$(find "$RESULT_DIR" -name "polarity_*.json" -type f 2>/dev/null | wc -l)
echo "Found $COMPLETED completed test result files"

if [ $COMPLETED -gt 0 ]; then
    echo
    echo "Recent results:"
    find "$RESULT_DIR" -name "polarity_*.json" -type f -exec ls -lth {} \; | head -10
fi

echo
echo "================================================"
echo "To check real-time progress, run:"
echo "  tail -f $LOG_FILE"
echo "================================================"
