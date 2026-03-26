#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   ML PIPELINE SETUP AND RUN${NC}"
echo -e "${BLUE}==========================================${NC}"

# Проверяем наличие виртуального окружения
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Виртуальное окружение не найдено. Создаем...${NC}"
    
    # Проверяем наличие python3-venv
    if ! dpkg -l | grep -q python3-venv; then
        echo -e "${YELLOW}📦 Устанавливаем python3-venv...${NC}"
        sudo apt update
        sudo apt install python3-venv python3-full -y
    fi
    
    # Создаем виртуальное окружение
    python3 -m venv venv
    echo -e "${GREEN}✓ Виртуальное окружение создано${NC}"
else
    echo -e "${GREEN}✓ Виртуальное окружение уже существует${NC}"
fi

# Активируем виртуальное окружение
echo -e "${BLUE}🔧 Активируем виртуальное окружение...${NC}"
source venv/bin/activate

# Проверяем установку пакетов
echo -e "${BLUE}📦 Проверяем установленные пакеты...${NC}"
if ! python -c "import pandas, numpy, sklearn, scipy, joblib" 2>/dev/null; then
    echo -e "${YELLOW}📦 Устанавливаем необходимые пакеты...${NC}"
    pip install pandas numpy scikit-learn scipy joblib
    echo -e "${GREEN}✓ Пакеты установлены${NC}"
else
    echo -e "${GREEN}✓ Все пакеты уже установлены${NC}"
fi

# Запускаем pipeline
echo -e "${BLUE}🚀 Запускаем ML Pipeline...${NC}"
echo -e "${BLUE}==========================================${NC}"
./pipeline.sh

# Сохраняем код возврата
PIPELINE_EXIT_CODE=$?

# Деактивируем виртуальное окружение
deactivate

echo -e "${BLUE}==========================================${NC}"
if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Pipeline выполнен успешно!${NC}"
else
    echo -e "${RED}❌ Pipeline завершился с ошибкой (код: $PIPELINE_EXIT_CODE)${NC}"
fi

exit $PIPELINE_EXIT_CODE
