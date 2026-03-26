import os
import pandas as pd
import numpy as np
import json
from scipy import stats
from datetime import datetime

def check_missing_values(df, filename, report):
    """Проверка пропущенных значений"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    for col in missing[missing > 0].index:
        report['missing'].append({
            'file': filename,
            'column': col,
            'count': int(missing[col]),
            'percentage': float(missing_pct[col]),
            'indices': df[df[col].isnull()].index.tolist()
        })
    
    return len(missing[missing > 0]) > 0

def detect_anomalies_zscore(df, column, filename, report, threshold=3):
    """Обнаружение аномалий методом Z-оценки"""
    if column not in df.columns or df[column].nunique() < 3:
        return []
    
    # Получаем только не-null значения
    valid_data = df[column].dropna()
    if len(valid_data) < 3:
        return []
    
    z_scores = np.abs(stats.zscore(valid_data))
    anomaly_mask = z_scores > threshold
    anomaly_indices = valid_data.index[anomaly_mask].tolist()
    
    if anomaly_indices:
        for idx in anomaly_indices:
            # Находим z-score для этого индекса
            idx_in_valid = valid_data.index.get_loc(idx)
            z_score_value = float(z_scores[idx_in_valid])
            
            report['anomalies_zscore'].append({
                'file': filename,
                'column': column,
                'index': int(idx),
                'value': float(df.loc[idx, column]),
                'z_score': z_score_value,
                'threshold': threshold
            })
    
    return anomaly_indices



def detect_anomalies_iqr(df, column, filename, report, multiplier=1.5):
    """Обнаружение аномалий методом межквартильного размаха (IQR)"""
    if column not in df.columns:
        return []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    anomaly_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    
    if anomaly_indices:
        for idx in anomaly_indices:
            report['anomalies_iqr'].append({
                'file': filename,
                'column': column,
                'index': int(idx),
                'value': float(df.loc[idx, column]),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'multiplier': multiplier
            })
    
    return anomaly_indices

def check_data_statistics(df, filename, report):
    """Сбор статистики по данным"""
    stats_data = {
        'file': filename,
        'rows': len(df),
        'columns': list(df.columns),
        'statistics': {}
    }
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            stats_data['statistics'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q1': float(df[col].quantile(0.25)),
                'q3': float(df[col].quantile(0.75))
            }
    
    report['statistics'].append(stats_data)

def main():
    print("=" * 60)
    print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("=" * 60)
    
    # Структура отчета
    report = {
        'timestamp': datetime.now().isoformat(),
        'missing': [],
        'anomalies_zscore': [],
        'anomalies_iqr': [],
        'statistics': [],
        'summary': {
            'total_files': 0,
            'files_with_missing': 0,
            'total_missing': 0,
            'files_with_zscore_anomalies': 0,
            'total_zscore_anomalies': 0,
            'files_with_iqr_anomalies': 0,
            'total_iqr_anomalies': 0
        }
    }
    
    # Проверка train данных
    print("\n📁 ПРОВЕРКА TRAIN ДАННЫХ:")
    print("-" * 40)
    
    train_files = [f for f in os.listdir("train") if f.endswith(".csv")]
    for f in sorted(train_files):
        print(f"\n📄 Файл: {f}")
        df = pd.read_csv(f"train/{f}")
        
        report['summary']['total_files'] += 1
        
        # Сбор статистики
        check_data_statistics(df, f, report)
        
        # Проверка пропусков
        has_missing = check_missing_values(df, f, report)
        if has_missing:
            report['summary']['files_with_missing'] += 1
        
        # Проверка аномалий в колонках y и x
        for col in ['y', 'x']:
            if col in df.columns:
                zscore_anomalies = detect_anomalies_zscore(df, col, f, report)
                if zscore_anomalies:
                    report['summary']['files_with_zscore_anomalies'] += 1
                    report['summary']['total_zscore_anomalies'] += len(zscore_anomalies)
                
                iqr_anomalies = detect_anomalies_iqr(df, col, f, report)
                if iqr_anomalies:
                    report['summary']['files_with_iqr_anomalies'] += 1
                    report['summary']['total_iqr_anomalies'] += len(iqr_anomalies)
        
        # Краткая информация
        print(f"  📊 Строк: {len(df)}")
        print(f"  📋 Колонки: {', '.join(df.columns)}")
    
    # Проверка test данных
    print("\n\n📁 ПРОВЕРКА TEST ДАННЫХ:")
    print("-" * 40)
    
    test_files = [f for f in os.listdir("test") if f.endswith(".csv")]
    for f in sorted(test_files):
        print(f"\n📄 Файл: {f}")
        df = pd.read_csv(f"test/{f}")
        
        report['summary']['total_files'] += 1
        
        # Сбор статистики
        check_data_statistics(df, f, report)
        
        # Проверка пропусков
        has_missing = check_missing_values(df, f, report)
        if has_missing:
            report['summary']['files_with_missing'] += 1
            report['summary']['total_missing'] += df.isnull().sum().sum()
        
        # Проверка аномалий в колонках y и x
        for col in ['y', 'x']:
            if col in df.columns:
                zscore_anomalies = detect_anomalies_zscore(df, col, f, report)
                if zscore_anomalies:
                    report['summary']['files_with_zscore_anomalies'] += 1
                    report['summary']['total_zscore_anomalies'] += len(zscore_anomalies)
                
                iqr_anomalies = detect_anomalies_iqr(df, col, f, report)
                if iqr_anomalies:
                    report['summary']['files_with_iqr_anomalies'] += 1
                    report['summary']['total_iqr_anomalies'] += len(iqr_anomalies)
        
        # Краткая информация
        print(f"  📊 Строк: {len(df)}")
        print(f"  📋 Колонки: {', '.join(df.columns)}")
    
    # Сохраняем отчет
    report['summary']['total_missing'] = len(report['missing'])
    
    with open('quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Итоговый отчет
    print("\n\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    print(f"\n📊 Общая статистика:")
    print(f"  - Всего проверено файлов: {report['summary']['total_files']}")
    print(f"  - Файлы с пропусками: {report['summary']['files_with_missing']}")
    print(f"  - Всего пропусков: {report['summary']['total_missing']}")
    print(f"  - Файлы с Z-score аномалиями: {report['summary']['files_with_zscore_anomalies']}")
    print(f"  - Всего Z-score аномалий: {report['summary']['total_zscore_anomalies']}")
    print(f"  - Файлы с IQR аномалиями: {report['summary']['files_with_iqr_anomalies']}")
    print(f"  - Всего IQR аномалий: {report['summary']['total_iqr_anomalies']}")
    
    print(f"\n✅ Отчет сохранен в 'quality_report.json'")
    print("=" * 60)

if __name__ == "__main__":
    main()
