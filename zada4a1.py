import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_parse_data(file_path):
    """
    Універсальна функція для завантаження даних з текстового файлу
    Підтримує різні формати організації даних
    """
    
    # Перевіряємо існування файлу
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")
    
    # Читаємо файл
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Очищуємо від пробільних символів
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        raise ValueError("Файл порожній!")
    
    print("\n🔍 Аналіз структури файлу...")
    
    # Визначаємо формат
    # Формат 1: Vertical (кожне значення на окремому рядку)
    # Формат 2: Horizontal (значення через табуляцію на одному рядку)
    
    # Шукаємо перший рядок з даними
    data_line_index = 0
    for i, line in enumerate(lines):
        if not line.startswith('─') and len(line.strip()) > 0:
            data_line_index = i
            break
    
    # Перевіряємо, чи є табуляція (горизонтальний формат)
    has_tabs = any('\t' in line for line in lines[:20])
    
    if has_tabs:
        print("  ✓ Визначено формат: ГОРИЗОНТАЛЬНИЙ (значення через табуляцію)")
        return parse_horizontal_format(lines)
    else:
        print("  ✓ Визначено формат: ВЕРТИКАЛЬНИЙ (значення на окремих рядках)")
        return parse_vertical_format(lines)


def parse_horizontal_format(lines):
    """
    Парсинг формату де кожне значення на окремому рядку
    Структура: Video, Time_s, Positive_count, Negative_count
    """
    data = []
    i = 4  # Починаємо після заголовків
    
    while i < len(lines):
        line = lines[i]
        
        # Шукаємо рядок з назвою запису (Video, User, тощо)
        if any(line.startswith(prefix) for prefix in ['Video', 'User', 'Entry']):
            record_name = line
            
            # Наступні рядки містять значення
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
    
    if not data:
        raise ValueError("Не вдалося розпарсити дані!")
    
    # Визначаємо кількість колон
    num_cols = len(data[0]) - 1
    headers = ['Record'] + [f'Column_{i+1}' for i in range(num_cols)]
    
    df = pd.DataFrame(data, columns=headers[:len(data[0])])
    
    # Конвертуємо числові колони
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def parse_vertical_format(lines):
    """
    Парсинг формату де значення розташовані горизонтально через табуляцію
    Структура: назва, значення1, значення2, ...
    """
    data = []
    i = 0
    
    # Пропускаємо заголовок
    while i < len(lines) and (lines[i].startswith('База') or lines[i].startswith('─') or not lines[i]):
        i += 1
    
    # Перший не-заголовок рядок - це назви колон
    if i < len(lines):
        headers_line = lines[i]
        headers = [h.strip() for h in headers_line.split('\t') if h.strip()] if '\t' in headers_line else ['Record']
        i += 1
    else:
        headers = ['Record']
    
    # Читаємо дані
    while i < len(lines):
        line = lines[i]
        if line and not line.startswith('─'):
            values = [v.strip() for v in line.split('\t') if v.strip()] if '\t' in line else [line]
            data.append(values)
        i += 1
    
    if not data:
        raise ValueError("Не вдалося розпарсити дані!")
    
    # Вирівнюємо кількість колон
    max_cols = max(len(row) for row in data)
    headers = headers if len(headers) >= max_cols else headers + [f'Column_{i+1}' for i in range(len(headers), max_cols)]
    
    df = pd.DataFrame(data, columns=headers[:max_cols])
    
    # Конвертуємо числові колони
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def analyze_data(df):
    """
    Поверхневий аналіз даних
    """
    print("\n" + "="*80)
    print("📊 ПОВЕРХНЕВИЙ АНАЛІЗ ДАНИХ")
    print("="*80)
    
    # Загальна інформація
    print(f"\n📈 Загальна інформація:")
    print(f"  • Кількість записів: {len(df)}")
    print(f"  • Кількість атрибутів: {len(df.columns)}")
    print(f"  • Типи даних:")
    for col, dtype in df.dtypes.items():
        print(f"    - {col}: {dtype}")
    
    # Пропущені значення
    print(f"\n🔍 Пропущені значення:")
    missing = df.isnull().sum()
    has_missing = False
    for col in missing.index:
        if missing[col] > 0:
            pct = (missing[col] / len(df)) * 100
            print(f"  • {col}: {missing[col]} ({pct:.1f}%)")
            has_missing = True
    if not has_missing:
        print("  ✓ Пропущених значень немає")
    else:
        print(f"  ВСЬОГО: {missing.sum()} пропущених значень")
    
    # Статистика числових колон
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        print(f"\n📐 СТАТИСТИКА ЧИСЛОВИХ АТРИБУТІВ:\n")
        
        for col in numeric_cols:
            print(f"{'─'*60}")
            print(f"📌 {col}:")
            
            stats = {
                'Кількість': df[col].count(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Std Dev': df[col].std(),
                'Variance': df[col].var(),
                'Q1 (25%)': df[col].quantile(0.25),
                'Q3 (75%)': df[col].quantile(0.75),
            }
            
            for stat_name, stat_value in stats.items():
                if pd.notna(stat_value):
                    if isinstance(stat_value, (int, np.integer)):
                        print(f"  {stat_name:12}: {stat_value}")
                    else:
                        print(f"  {stat_name:12}: {stat_value:.2f}")
    
    # Кореляційна матриця
    if len(numeric_cols) > 1:
        print(f"\n{'─'*60}")
        print(f"📊 КОРЕЛЯЦІЙНА МАТРИЦЯ:\n")
        correlation = df[numeric_cols].corr()
        print(correlation.to_string())
    
    # Опис даних
    print(f"\n{'─'*60}")
    print(f"📋 ОПИС ДАНИХ:\n")
    print(df.describe().to_string())
    
    print(f"\n{'='*80}")
    print("✅ Аналіз завершено!")
    
    return df


def save_results(df, output_path=None):
    """
    Збереження результатів аналізу
    """
    if output_path is None:
        # Зберігаємо в тій же папці з префіксом 'analysis_'
        output_path = Path(os.getcwd()) / 'analysis_results.csv'
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 Результати збережено: {output_path}")


def main():
    """
    Основна функція програми
    """
    print("\n" + "="*80)
    print("🔧 УНІВЕРСАЛЬНИЙ АНАЛІЗАТОР ТЕКСТОВИХ ТАБЛИЦЬ ДАНИХ")
    print("="*80)
    
    # Запитуємо шлях до файлу
    print("\n📂 Введіть шлях до текстової таблиці даних:")
    print("   (Приклад: G:\\path\\to\\file.txt)")
    print("   Або натисніть Enter для використання файлу з поточної папки")
    
    file_path = input("\n▶ Шлях: ").strip()
    
    # Видаляємо лапки, якщо користувач їх ввів
    file_path = file_path.strip('"').strip("'")
    
    # Якщо користувач не ввів шлях, шукаємо файли в поточній папці
    if not file_path:
        txt_files = list(Path('.').glob('*.txt'))
        if txt_files:
            print(f"\n📁 Знайдено {len(txt_files)} текстових файлів:")
            for i, f in enumerate(txt_files, 1):
                print(f"  {i}. {f.name}")
            
            choice = input("\n▶ Оберіть номер файлу (або введіть шлях): ").strip()
            # Видаляємо лапки з вибору
            choice = choice.strip('"').strip("'")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(txt_files):
                    file_path = str(txt_files[idx].absolute())
                else:
                    file_path = choice
            except ValueError:
                file_path = choice
        else:
            print("❌ Файли не знайдені!")
            return
    
    # Розширюємо ~ до повного шляху та нормалізуємо
    file_path = os.path.expanduser(file_path)
    file_path = os.path.normpath(file_path)
    
    # Перевіряємо існування файлу з детальною інформацією
    if not os.path.exists(file_path):
        print(f"\n❌ Файл не знайдено!")
        print(f"   Шлях: {file_path}")
        print(f"   Перевірте правильність шляху")
        
        # Пробуємо знайти файл в поточній директорії
        filename = os.path.basename(file_path)
        local_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(local_path):
            print(f"\n💡 Знайдено файл в поточній папці: {filename}")
            use_local = input("   Використати цей файл? (y/n): ").strip().lower()
            if use_local == 'y':
                file_path = local_path
            else:
                return
        else:
            return
    
    try:
        # Завантажуємо дані
        print(f"\n📂 Завантаження файлу: {file_path}")
        df = load_and_parse_data(file_path)
        print(f"✓ Успішно завантажено {len(df)} записів")
        
        # Проводимо аналіз
        df = analyze_data(df)
        
        # Запитуємо про збереження результатів
        save_choice = input("\n\n💾 Зберегти результати в CSV? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_file = input("▶ Введіть назву файлу (або Enter для використання за замовчуванням): ").strip()
            # Видаляємо лапки з назви файлу
            output_file = output_file.strip('"').strip("'")
            if output_file:
                save_results(df, output_file)
            else:
                save_results(df)
        
    except ValueError as e:
        print(f"\n❌ Помилка при парсингу: {e}")
    except Exception as e:
        print(f"\n❌ Непередбачена помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
