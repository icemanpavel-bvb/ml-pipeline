#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   ML PIPELINE${NC}"
echo -e "${BLUE}==========================================${NC}"

# ============================================
# 1. НАСТРОЙКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ
# ============================================
echo -e "${BLUE}=== Step 0: Environment setup ===${NC}"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Активируем виртуальное окружение
source venv/bin/activate

# ============================================
# 2. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ============================================
echo -e "${BLUE}=== Step 1: Installing dependencies ===${NC}"

# Проверяем установку пакетов
if ! python -c "import pandas, numpy, sklearn, scipy, joblib" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing packages...${NC}"
    pip install -q pandas numpy scikit-learn scipy joblib
    echo -e "${GREEN}✓ Packages installed${NC}"
else
    echo -e "${GREEN}✓ All packages already installed${NC}"
fi

echo ""

# ============================================
# 3. ЗАПУСК ОСНОВНОГО ПАЙПЛАЙНА
# ============================================
echo "=== Step 2: Data creation ==="
python3 data_creation.py

echo "=== Step 3: Data quality check ==="
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
        echo "=== Step 4: Data fixing (issues detected) ==="
        python3 data_fix.py
    else
        echo "=== Step 4: Data fixing skipped (no issues) ==="
    fi
else
    echo "=== Step 4: Data fixing skipped (no report) ==="
fi

echo "=== Step 5: Preprocessing ==="
python3 model_preprocessing.py

echo "=== Step 6: Model training ==="
python3 model_preparation.py

echo "=== Step 7: Testing ==="
python3 model_testing.py

echo "=== Pipeline finished ==="

# Деактивируем виртуальное окружение
deactivate
