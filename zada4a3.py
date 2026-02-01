import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_parse_data(file_path):
    """
    Завантаження даних з текстового файлу
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    # Розбір структури (вертикальний формат)
    data = []
    i = 4  # Починаємо після заголовків
    
    while i < len(lines):
        line = lines[i]
        if any(line.startswith(prefix) for prefix in ['Video', 'User', 'Entry']):
            record_name = line
            values = []
            j = i + 1
            while j < len(lines) and not any(lines[j].startswith(prefix) for prefix in ['Video', 'User', 'Entry', '─']):
                val = lines[j]
                if val and not val.startswith('─'):
                    values.append(val)
                    j += 1
                else:
                    break
            
            if values:
                data.append([record_name] + values)
                i = j
            else:
                i += 1
        else:
            i += 1
    
    num_cols = len(data[0]) - 1
    headers = ['Record', 'Time_s', 'Positive_count', 'Negative_count'][:len(data[0])]
    
    df = pd.DataFrame(data, columns=headers[:len(data[0])])
    
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def normalize_minmax(df):
    """
    Min-Max нормалізація: приведення до діапазону [0, 1]
    Формула: (x - min) / (max - min)
    """
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    normalization_params = {}
    
    print("\n" + "="*80)
    print("📊 MIN-MAX НОРМАЛІЗАЦІЯ [0, 1]")
    print("="*80)
    
    for col in numeric_cols:
        # Пропускаємо колонки з N/A
        if df[col].notna().sum() == 0:
            continue
            
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val != min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            normalization_params[col] = {'min': min_val, 'max': max_val}
            print(f"\n✓ {col}:")
            print(f"  Оригінал: [{min_val:.2f}, {max_val:.2f}]")
            print(f"  Нормалізовано: [0.00, 1.00]")
        else:
            print(f"\n⚠️  {col}: всі значення однакові ({min_val}), нормалізація неможлива")
    
    return df_normalized, normalization_params


def normalize_zscore(df):
    """
    Z-score (стандартизація): приведення до N(0, 1)
    Формула: (x - mean) / std
    """
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    normalization_params = {}
    
    print("\n" + "="*80)
    print("📊 Z-SCORE СТАНДАРТИЗАЦІЯ")
    print("="*80)
    
    for col in numeric_cols:
        if df[col].notna().sum() == 0:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val != 0:
            df_normalized[col] = (df[col] - mean_val) / std_val
            normalization_params[col] = {'mean': mean_val, 'std': std_val}
            print(f"\n✓ {col}:")
            print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
            print(f"  Після: Mean ≈ 0, Std ≈ 1")
        else:
            print(f"\n⚠️  {col}: std = 0, стандартизація неможлива")
    
    return df_normalized, normalization_params


def normalize_robust(df):
    """
    Robust scaling: використовує медіану та IQR
    Формула: (x - median) / IQR
    Менш чутлива до викидів
    """
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    normalization_params = {}
    
    print("\n" + "="*80)
    print("📊 ROBUST SCALING (МЕДІАНА + IQR)")
    print("="*80)
    
    for col in numeric_cols:
        if df[col].notna().sum() == 0:
            continue
            
        median_val = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        if iqr != 0:
            df_normalized[col] = (df[col] - median_val) / iqr
            normalization_params[col] = {'median': median_val, 'Q1': q1, 'Q3': q3, 'IQR': iqr}
            print(f"\n✓ {col}:")
            print(f"  Median: {median_val:.2f}, IQR: {iqr:.2f}")
            print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}")
        else:
            print(f"\n⚠️  {col}: IQR = 0, нормалізація неможлива")
    
    return df_normalized, normalization_params


def normalize_decimal_scaling(df):
    """
    Decimal scaling: ділення на 10^d, де d - кількість цифр
    Формула: x / 10^d
    """
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    normalization_params = {}
    
    print("\n" + "="*80)
    print("📊 DECIMAL SCALING")
    print("="*80)
    
    for col in numeric_cols:
        if df[col].notna().sum() == 0:
            continue
            
        max_abs = df[col].abs().max()
        if max_abs > 0:
            d = int(np.ceil(np.log10(max_abs)))
            divisor = 10 ** d
            df_normalized[col] = df[col] / divisor
            normalization_params[col] = {'d': d, 'divisor': divisor}
            print(f"\n✓ {col}:")
            print(f"  Max |value|: {max_abs:.2f}")
            print(f"  Divisor: 10^{d} = {divisor}")
            print(f"  Діапазон після: [{df_normalized[col].min():.4f}, {df_normalized[col].max():.4f}]")
    
    return df_normalized, normalization_params


def save_results(df_original, df_normalized, output_filename, method, params):
    """
    Збереження результатів нормалізації
    """
    output_path = output_filename
    
    # Зберігаємо в текстовому форматі
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Нормалізовані дані\n")
        f.write(f"Метод: {method}\n")
        f.write("=" * 60 + "\n\n")
        
        # Параметри нормалізації
        f.write("Параметри нормалізації:\n")
        f.write("-" * 60 + "\n")
        for col, param in params.items():
            f.write(f"\n{col}:\n")
            for key, value in param.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("НОРМАЛІЗОВАНІ ДАНІ\n")
        f.write("=" * 60 + "\n\n")
        
        # Назви колонок
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            f.write(f"\t{col}\n")
        
        # Дані
        for idx, row in df_normalized.iterrows():
            f.write(f"{row['Record']}\n")
            for col in numeric_cols:
                if pd.notna(row[col]):
                    f.write(f"\t{row[col]:.6f}\n")
                else:
                    f.write(f"\tN/A\n")
    
    print(f"\n💾 Результати збережено: {output_path}")
    
    # CSV файл
    csv_path = output_path.replace('.txt', '.csv')
    df_normalized.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"💾 CSV файл: {csv_path}")
    
    # Файл з параметрами
    params_path = output_path.replace('.txt', '_params.txt')
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(f"Параметри нормалізації ({method})\n")
        f.write("=" * 60 + "\n\n")
        for col, param in params.items():
            f.write(f"{col}:\n")
            for key, value in param.items():
                f.write(f"  {key}: {value:.6f}\n")
            f.write("\n")
    print(f"💾 Параметри: {params_path}")


def main():
    """
    Основна функція програми
    """
    print("\n" + "="*80)
    print("📊 НОРМАЛІЗАЦІЯ ДАНИХ")
    print("="*80)
    
    # Запитуємо шлях до файлу
    print("\n📂 Введіть шлях до текстової таблиці даних:")
    print("   (Приклад: G:\\path\\to\\file.txt)")
    print("   Або натисніть Enter для використання файлу з поточної папки")
    
    file_path = input("\n▶ Шлях: ").strip()
    
    # Видаляємо лапки
    file_path = file_path.strip('"').strip("'")
    
    # Якщо не ввели шлях
    if not file_path:
        txt_files = list(Path('.').glob('*.txt'))
        if txt_files:
            print(f"\n📁 Знайдено {len(txt_files)} текстових файлів:")
            for i, f in enumerate(txt_files, 1):
                print(f"  {i}. {f.name}")
            
            choice = input("\n▶ Оберіть номер файлу: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(txt_files):
                    file_path = str(txt_files[idx].absolute())
                else:
                    print("❌ Невірний номер!")
                    return
            except ValueError:
                print("❌ Потрібно ввести номер!")
                return
        else:
            print("❌ Файли не знайдені!")
            return
    
    # Нормалізуємо шлях
    file_path = os.path.expanduser(file_path)
    file_path = os.path.normpath(file_path)
    
    if not os.path.exists(file_path):
        print(f"\n❌ Файл не знайдено: {file_path}")
        return
    
    try:
        # Завантажуємо дані
        print(f"\n📂 Завантаження файлу: {os.path.basename(file_path)}")
        df = load_and_parse_data(file_path)
        print(f"✓ Завантажено {len(df)} записів")
        
        # Показуємо статистику до нормалізації
        print("\n" + "="*80)
        print("📈 СТАТИСТИКА ДО НОРМАЛІЗАЦІЇ")
        print("="*80)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                print(f"\n{col}:")
                print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
                print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
                print(f"  Median: {df[col].median():.2f}")
        
        # Вибір методу нормалізації
        print("\n" + "="*80)
        print("📊 ОБЕРІТЬ МЕТОД НОРМАЛІЗАЦІЇ:")
        print("="*80)
        print("  1. Min-Max нормалізація [0, 1]")
        print("  2. Z-score стандартизація (mean=0, std=1)")
        print("  3. Robust scaling (медіана + IQR)")
        print("  4. Decimal scaling")
        
        method_choice = input("\n▶ Ваш вибір (1-4): ").strip()
        
        method_map = {
            '1': ('MinMax', normalize_minmax),
            '2': ('ZScore', normalize_zscore),
            '3': ('Robust', normalize_robust),
            '4': ('Decimal', normalize_decimal_scaling)
        }
        
        if method_choice not in method_map:
            print("❌ Невірний вибір!")
            return
        
        method_name, normalize_func = method_map[method_choice]
        
        # Нормалізуємо
        df_normalized, params = normalize_func(df)
        
        # Статистика після нормалізації
        print("\n" + "="*80)
        print("📈 СТАТИСТИКА ПІСЛЯ НОРМАЛІЗАЦІЇ")
        print("="*80)
        
        for col in numeric_cols:
            if df_normalized[col].notna().sum() > 0:
                print(f"\n{col}:")
                print(f"  Min: {df_normalized[col].min():.6f}, Max: {df_normalized[col].max():.6f}")
                print(f"  Mean: {df_normalized[col].mean():.6f}, Std: {df_normalized[col].std():.6f}")
                print(f"  Median: {df_normalized[col].median():.6f}")
        
        # Зберігаємо результати
        output_filename = f"Normalized_{method_name}.txt"
        
        save_choice = input(f"\n💾 Зберегти результати у '{output_filename}'? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results(df, df_normalized, output_filename, method_name, params)
            print("\n✅ Нормалізація завершена!")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
