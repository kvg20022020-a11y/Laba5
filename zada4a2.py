import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats

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


def check_errors(df):
    """
    Перевірка даних на похибки (пропущені значення)
    """
    print("\n" + "="*80)
    print("🔍 ПЕРЕВІРКА ДАНИХ НА ПОХИБКИ")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Загальна статистика пропущених значень
    missing = df[numeric_cols].isnull().sum()
    total_missing = missing.sum()
    
    if total_missing == 0:
        print("\n✅ Пропущених значень не знайдено!")
        return False
    
    print(f"\n⚠️  Знайдено {total_missing} пропущених значень:\n")
    
    for col in numeric_cols:
        if missing[col] > 0:
            pct = (missing[col] / len(df)) * 100
            print(f"  • {col}: {missing[col]} пропущених ({pct:.1f}%)")
            
            # Показуємо які саме записи мають пропущені значення
            missing_records = df[df[col].isnull()]['Record'].tolist()
            print(f"    Записи: {', '.join(missing_records[:5])}", end="")
            if len(missing_records) > 5:
                print(f" ... та ще {len(missing_records) - 5}")
            else:
                print()
    
    return True


def calculate_mode(series):
    """
    Обчислення моди (найчастіше значення)
    """
    mode_result = series.mode()
    if len(mode_result) > 0:
        return mode_result[0]
    return series.mean()  # Якщо моди немає, повертаємо середнє


def euclidean_distance(row1, row2, cols):
    """
    Евклідова відстань між двома рядками
    """
    distance = 0
    count = 0
    for col in cols:
        if pd.notna(row1[col]) and pd.notna(row2[col]):
            distance += (row1[col] - row2[col]) ** 2
            count += 1
    
    if count == 0:
        return float('inf')
    
    return np.sqrt(distance)


def manhattan_distance(row1, row2, cols):
    """
    Манхеттенська відстань між двома рядками
    """
    distance = 0
    count = 0
    for col in cols:
        if pd.notna(row1[col]) and pd.notna(row2[col]):
            distance += abs(row1[col] - row2[col])
            count += 1
    
    if count == 0:
        return float('inf')
    
    return distance


def chebyshev_distance(row1, row2, cols):
    """
    Відстань Чебишева між двома рядками
    """
    max_dist = 0
    count = 0
    for col in cols:
        if pd.notna(row1[col]) and pd.notna(row2[col]):
            max_dist = max(max_dist, abs(row1[col] - row2[col]))
            count += 1
    
    if count == 0:
        return float('inf')
    
    return max_dist


def fix_errors_with_metric(df, method='mean', k=5, distance_metric='euclidean'):
    """
    Виправлення похибок за допомогою різних метрик
    
    Parameters:
    -----------
    method : str
        'mean' - середнє арифметичне
        'median' - медіана
        'mode' - мода
        'knn' - k-найближчих сусідів
    k : int
        Кількість сусідів для KNN
    distance_metric : str
        'euclidean' - евклідова відстань
        'manhattan' - манхеттенська відстань
        'chebyshev' - відстань Чебишева
    """
    df_fixed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*80)
    print(f"🔧 ВИПРАВЛЕННЯ ПОХИБОК: {method.upper()}")
    if method == 'knn':
        print(f"   Метрика відстані: {distance_metric}, k={k}")
    print("="*80)
    
    if method == 'mean':
        # Заповнення середнім значенням
        for col in numeric_cols:
            if df_fixed[col].isnull().any():
                mean_val = df_fixed[col].mean()
                df_fixed[col].fillna(mean_val, inplace=True)
                print(f"✓ {col}: заповнено середнім ({mean_val:.2f})")
    
    elif method == 'median':
        # Заповнення медіаною
        for col in numeric_cols:
            if df_fixed[col].isnull().any():
                median_val = df_fixed[col].median()
                df_fixed[col].fillna(median_val, inplace=True)
                print(f"✓ {col}: заповнено медіаною ({median_val:.2f})")
    
    elif method == 'mode':
        # Заповнення модою
        for col in numeric_cols:
            if df_fixed[col].isnull().any():
                mode_val = calculate_mode(df_fixed[col])
                df_fixed[col].fillna(mode_val, inplace=True)
                print(f"✓ {col}: заповнено модою ({mode_val:.2f})")
    
    elif method == 'knn':
        # Заповнення методом k-найближчих сусідів
        distance_func = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'chebyshev': chebyshev_distance
        }.get(distance_metric, euclidean_distance)
        
        for idx in df_fixed.index:
            row = df_fixed.loc[idx]
            
            # Перевіряємо чи є пропущені значення в цьому рядку
            if row[numeric_cols].isnull().any():
                # Знаходимо рядки без пропущених значень
                complete_rows = df_fixed[df_fixed[numeric_cols].notna().all(axis=1)]
                
                if len(complete_rows) == 0:
                    continue
                
                # Обчислюємо відстані до всіх повних рядків
                distances = []
                for comp_idx in complete_rows.index:
                    if comp_idx != idx:
                        dist = distance_func(row, complete_rows.loc[comp_idx], numeric_cols)
                        distances.append((comp_idx, dist))
                
                # Сортуємо за відстанню і беремо k найближчих
                distances.sort(key=lambda x: x[1])
                k_nearest = distances[:min(k, len(distances))]
                
                # Заповнюємо пропущені значення середнім k-найближчих
                for col in numeric_cols:
                    if pd.isna(row[col]):
                        values = [df_fixed.loc[neighbor_idx][col] for neighbor_idx, _ in k_nearest]
                        df_fixed.loc[idx, col] = np.mean(values)
                        print(f"✓ {row['Record']}, {col}: відновлено ({df_fixed.loc[idx, col]:.2f})")
    
    return df_fixed


def save_results(df, output_filename, method):
    """
    Збереження результатів
    """
    # Зберігаємо в форматі як у вхідному файлі
    output_path = output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Заголовок
        f.write("Виправлені дані\n")
        f.write(f"Метод: {method}\n")
        f.write("─" * 60 + "\n\n")
        
        # Назви колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            f.write(f"\t{col}\n")
        
        # Дані
        for idx, row in df.iterrows():
            f.write(f"{row['Record']}\n")
            for col in numeric_cols:
                f.write(f"\t{row[col]:.2f}\n")
    
    print(f"\n💾 Результати збережено: {output_path}")
    
    # Також зберігаємо CSV для зручності
    csv_path = output_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"💾 CSV файл: {csv_path}")


def main():
    """
    Основна функція програми
    """
    print("\n" + "="*80)
    print("🔧 ПЕРЕВІРКА ТА ВИПРАВЛЕННЯ ПОХИБОК У ДАНИХ")
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
    
    # Перевіряємо існування
    if not os.path.exists(file_path):
        print(f"\n❌ Файл не знайдено: {file_path}")
        return
    
    try:
        # Завантажуємо дані
        print(f"\n📂 Завантаження файлу: {os.path.basename(file_path)}")
        df = load_and_parse_data(file_path)
        print(f"✓ Завантажено {len(df)} записів")
        
        # Перевіряємо на похибки
        has_errors = check_errors(df)
        
        if not has_errors:
            print("\n✅ Дані не потребують виправлення!")
            return
        
        # Вибір методу виправлення
        print("\n" + "="*80)
        print("📊 ОБЕРІТЬ МЕТОД ВИПРАВЛЕННЯ:")
        print("="*80)
        print("  1. Середнє арифметичне (Mean)")
        print("  2. Медіана (Median)")
        print("  3. Мода (Mode)")
        print("  4. KNN з Евклідовою відстанню")
        print("  5. KNN з Манхеттенською відстанню")
        print("  6. KNN з відстанню Чебишева")
        
        method_choice = input("\n▶ Ваш вибір (1-6): ").strip()
        
        method_map = {
            '1': ('mean', None),
            '2': ('median', None),
            '3': ('mode', None),
            '4': ('knn', 'euclidean'),
            '5': ('knn', 'manhattan'),
            '6': ('knn', 'chebyshev')
        }
        
        if method_choice not in method_map:
            print("❌ Невірний вибір!")
            return
        
        method, distance_metric = method_map[method_choice]
        
        # Для KNN запитуємо k
        k = 5
        if method == 'knn':
            k_input = input(f"▶ Кількість сусідів k (за замовчуванням {k}): ").strip()
            if k_input:
                try:
                    k = int(k_input)
                except ValueError:
                    print(f"⚠️  Використано k={k}")
        
        # Виправляємо похибки
        df_fixed = fix_errors_with_metric(df, method=method, k=k, distance_metric=distance_metric)
        
        # Перевіряємо результат
        print("\n" + "="*80)
        print("✅ РЕЗУЛЬТАТ ВИПРАВЛЕННЯ")
        print("="*80)
        
        remaining_errors = df_fixed.select_dtypes(include=[np.number]).isnull().sum().sum()
        if remaining_errors == 0:
            print("\n✅ Всі похибки успішно виправлено!")
        else:
            print(f"\n⚠️  Залишилось {remaining_errors} похибок")
        
        # Статистика після виправлення
        print("\n📊 Статистика після виправлення:\n")
        numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            print(f"  {col}:")
            print(f"    Min: {df_fixed[col].min():.2f}, Max: {df_fixed[col].max():.2f}")
            print(f"    Mean: {df_fixed[col].mean():.2f}, Median: {df_fixed[col].median():.2f}")
        
        # Зберігаємо результати
        method_name = f"{method}_{distance_metric}" if distance_metric else method
        output_filename = f"Fixed_{method_name}.txt"
        
        save_choice = input(f"\n💾 Зберегти результати у '{output_filename}'? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results(df_fixed, output_filename, method_name)
            print("\n✅ Готово!")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
