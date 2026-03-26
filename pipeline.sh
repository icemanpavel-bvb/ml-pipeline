#!/bin/bash

echo "=== Step 1: Data creation ==="
python3 data_creation.py

echo "=== Step 2: Data quality check ==="
python3 data_quality_check.py

# Проверяем наличие проблем в данных
if [ -f "quality_report.json" ]; then
    HAS_ISSUES=$(python3 -c "
import json
with open('quality_report.json', 'r') as f:
    report = json.load(f)
    if report['summary']['total_missing'] > 0 or report['summary']['total_zscore_anomalies'] > 0 or report['summary']['total_iqr_anomalies'] > 0:
        print('yes')
    else:
        print('no')
")
    
    if [ "$HAS_ISSUES" = "yes" ]; then
        echo "=== Step 3: Data fixing (issues detected) ==="
        python3 data_fix.py
    else
        echo "=== Step 3: Data fixing skipped (no issues) ==="
    fi
else
    echo "=== Step 3: Data fixing skipped (no report) ==="
fi

echo "=== Step 4: Preprocessing ==="
python3 model_preprocessing.py

echo "=== Step 5: Model training ==="
python3 model_preparation.py

echo "=== Step 6: Testing ==="
python3 model_testing.py

echo "=== Pipeline finished ==="
