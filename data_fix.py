import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

def fix_missing_values(df, missing_entries):
    """Исправление пропущенных значений на основе отчета"""
    fixed_count = 0
    
    for missing in missing_entries:
        file_name = missing['file']
        column = missing['column']
        indices = missing['indices']
        
        # Для каждого пропуска используем интерполяцию
        for idx in indices:
            if idx > 0 and idx < len(df) - 1:
                # Интерполяция между соседними значениями
                df.loc[idx, column] = (df.loc[idx-1, column] + df.loc[idx+1, column]) / 2
            elif idx == 0:
                # Заполняем первое значение следующим
                df.loc[idx, column] = df.loc[idx+1, column]
            elif idx == len(df) - 1:
                # Заполняем последнее значение предыдущим
                df.loc[idx, column] = df.loc[idx-1, column]
            fixed_count += 1
    
    return df, fixed_count

def fix_anomalies_zscore(df, anomalies, method='clip'):
    """Исправление Z-score аномалий на основе отчета"""
    fixed_count = 0
    
    for anomaly in anomalies:
        file_name = anomaly['file']
        column = anomaly['column']
        idx = anomaly['index']
        value = anomaly['value']
        
        if method == 'clip':
            # Обрезаем до нормального диапазона
            mean_val = df[column].mean()
            std_val = df[column].std()
            threshold = anomaly['threshold']
            upper_bound = mean_val + threshold * std_val
            lower_bound = mean_val - threshold * std_val
            
            if value > upper_bound:
                df.loc[idx, column] = upper_bound
            elif value < lower_bound:
                df.loc[idx, column] = lower_bound
            fixed_count += 1
        
        elif method == 'replace':
            # Заменяем на медиану
            df.loc[idx, column] = df[column].median()
            fixed_count += 1
        
        elif method == 'remove':
            # Отмечаем для удаления (будет удалено позже)
            df.loc[idx, 'to_remove'] = True
            fixed_count += 1
    
    return df, fixed_count

def fix_anomalies_iqr(df, anomalies, method='clip'):
    """Исправление IQR аномалий на основе отчета"""
    fixed_count = 0
    
    for anomaly in anomalies:
        column = anomaly['column']
        idx = anomaly['index']
        value = anomaly['value']
        lower_bound = anomaly['lower_bound']
        upper_bound = anomaly['upper_bound']
        
        if method == 'clip':
            if value > upper_bound:
                df.loc[idx, column] = upper_bound
            elif value < lower_bound:
                df.loc[idx, column] = lower_bound
            fixed_count += 1
        
        elif method == 'replace':
            df.loc[idx, column] = df[column].median()
            fixed_count += 1
        
        elif method == 'remove':
            df.loc[idx, 'to_remove'] = True
            fixed_count += 1
    
    return df, fixed_count

def smooth_anomalies(df, anomalies, window=5):
    """Сглаживание аномальных точек"""
    fixed_count = 0
    
    # Группируем аномалии по файлам и колонкам
    for anomaly in anomalies:
        column = anomaly['column']
        idx = anomaly['index']
        
        # Создаем скользящее окно
        left_idx = max(0, idx - window//2)
        right_idx = min(len(df) - 1, idx + window//2)
        
        # Сглаживаем значение
        smoothed_value = df[column].iloc[left_idx:right_idx+1].mean()
        df.loc[idx, column] = smoothed_value
        fixed_count += 1
    
    return df, fixed_count

def load_report():
    """Загрузка отчета о качестве данных"""
    try:
        with open('quality_report.json', 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report
    except FileNotFoundError:
        print("❌ Ошибка: Файл quality_report.json не найден!")
        print("   Сначала запустите data_quality_check.py")
        return None

def main():
    print("=" * 60)
    print("ИСПРАВЛЕНИЕ ДАННЫХ НА ОСНОВЕ ОТЧЕТА")
    print("=" * 60)
    
    # Загружаем отчет
    report = load_report()
    if report is None:
        return
    
    # Настройки исправления
    fix_config = {
        'fix_missing': True,
        'fix_zscore': True,
        'fix_iqr': True,
        'smooth_anomalies': True,
        'anomaly_method': 'clip',  # clip, replace, remove
        'smooth_window': 5
    }
    
    print("\n📋 Конфигурация исправления:")
    for key, value in fix_config.items():
        print(f"  - {key}: {value}")
    
    # Группируем проблемы по файлам
    files_to_fix = set()
    
    if fix_config['fix_missing']:
        for missing in report['missing']:
            files_to_fix.add(missing['file'])
    
    if fix_config['fix_zscore']:
        for anomaly in report['anomalies_zscore']:
            files_to_fix.add(anomaly['file'])
    
    if fix_config['fix_iqr']:
        for anomaly in report['anomalies_iqr']:
            files_to_fix.add(anomaly['file'])
    
    print(f"\n📁 Файлы для исправления: {len(files_to_fix)}")
    
    total_fixes = {
        'missing': 0,
        'zscore': 0,
        'iqr': 0,
        'smooth': 0,
        'removed_rows': 0
    }
    
    # Обрабатываем каждый файл
    for file_name in sorted(files_to_fix):
        print(f"\n📄 Исправление файла: {file_name}")
        
        # Определяем директорию
        if file_name in [f for f in os.listdir("train") if f.endswith(".csv")]:
            filepath = f"train/{file_name}"
        else:
            filepath = f"test/{file_name}"
        
        # Загружаем данные
        df = pd.read_csv(filepath)
        
        # Добавляем временную колонку для пометок на удаление
        df['to_remove'] = False
        
        modifications = []
        
        # 1. Исправляем пропуски
        if fix_config['fix_missing']:
            missing_for_file = [m for m in report['missing'] if m['file'] == file_name]
            if missing_for_file:
                df, fixed = fix_missing_values(df, missing_for_file)
                total_fixes['missing'] += fixed
                modifications.append(f"исправлено пропусков: {fixed}")
        
        # 2. Исправляем Z-score аномалии
        if fix_config['fix_zscore']:
            zscore_for_file = [a for a in report['anomalies_zscore'] if a['file'] == file_name]
            if zscore_for_file:
                df, fixed = fix_anomalies_zscore(df, zscore_for_file, method=fix_config['anomaly_method'])
                total_fixes['zscore'] += fixed
                modifications.append(f"исправлено Z-score аномалий: {fixed}")
        
        # 3. Исправляем IQR аномалии
        if fix_config['fix_iqr']:
            iqr_for_file = [a for a in report['anomalies_iqr'] if a['file'] == file_name]
            if iqr_for_file:
                df, fixed = fix_anomalies_iqr(df, iqr_for_file, method=fix_config['anomaly_method'])
                total_fixes['iqr'] += fixed
                modifications.append(f"исправлено IQR аномалий: {fixed}")
        
        # 4. Сглаживание аномалий
        if fix_config['smooth_anomalies']:
            all_anomalies = [a for a in report['anomalies_zscore'] if a['file'] == file_name]
            all_anomalies.extend([a for a in report['anomalies_iqr'] if a['file'] == file_name])
            if all_anomalies:
                df, smoothed = smooth_anomalies(df, all_anomalies, window=fix_config['smooth_window'])
                total_fixes['smooth'] += smoothed
                modifications.append(f"сглажено аномалий: {smoothed}")
        
        # 5. Удаляем помеченные строки
        rows_before = len(df)
        df = df[~df['to_remove']]
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            total_fixes['removed_rows'] += rows_removed
            modifications.append(f"удалено строк: {rows_removed}")
        
        # Удаляем временную колонку
        if 'to_remove' in df.columns:
            df = df.drop(columns=['to_remove'])
        
        # Сохраняем исправленный файл
        df.to_csv(filepath, index=False)
        
        if modifications:
            print(f"  ✅ Выполнено:")
            for mod in modifications:
                print(f"     - {mod}")
        else:
            print(f"  ✅ Изменений не требуется")
    
    # Сохраняем историю исправлений
    fix_history = {
        'timestamp': datetime.now().isoformat(),
        'config': fix_config,
        'fixes': total_fixes,
        'based_on_report': report['timestamp']
    }
    
    with open('fix_history.json', 'w', encoding='utf-8') as f:
        json.dump(fix_history, f, ensure_ascii=False, indent=2)
    
    # Итоговый отчет
    print("\n\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ ПО ИСПРАВЛЕНИЯМ")
    print("=" * 60)
    
    print(f"\n📊 Статистика исправлений:")
    print(f"  - Исправлено пропусков: {total_fixes['missing']}")
    print(f"  - Исправлено Z-score аномалий: {total_fixes['zscore']}")
    print(f"  - Исправлено IQR аномалий: {total_fixes['iqr']}")
    print(f"  - Сглажено аномалий: {total_fixes['smooth']}")
    print(f"  - Удалено строк: {total_fixes['removed_rows']}")
    
    print(f"\n✅ История исправлений сохранена в 'fix_history.json'")
    print("=" * 60)

if __name__ == "__main__":
    main()
