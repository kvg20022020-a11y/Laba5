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


def euclidean_distance(p1, p2):
    """
    Евклідова відстань
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


def manhattan_distance(p1, p2):
    """
    Манхеттенська відстань
    """
    return np.sum(np.abs(p1 - p2))


def chebyshev_distance(p1, p2):
    """
    Відстань Чебишева
    """
    return np.max(np.abs(p1 - p2))


def cosine_similarity(p1, p2):
    """
    Косинус схожості (повертаємо відстань як 1 - схожість)
    """
    dot_product = np.dot(p1, p2)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)
    
    if norm_p1 == 0 or norm_p2 == 0:
        return float('inf')
    
    similarity = dot_product / (norm_p1 * norm_p2)
    return 1 - similarity


def find_recommendations(df, user_params, k=5, metric='euclidean'):
    """
    Знайти k найближчих відео до вказаних параметрів
    
    Parameters:
    -----------
    df : DataFrame
        Таблиця з відео та параметрами
    user_params : dict
        Параметри користувача {'Time_s': ..., 'Positive_count': ..., 'Negative_count': ...}
    k : int
        Кількість рекомендацій
    metric : str
        Метрика відстані: 'euclidean', 'manhattan', 'chebyshev', 'cosine'
    """
    
    # Вибираємо тільки повні рядки (без N/A)
    df_clean = df.dropna(subset=['Time_s', 'Positive_count', 'Negative_count'])
    
    if len(df_clean) == 0:
        print("❌ Немає повних даних для пошуку!")
        return None
    
    # Параметри користувача як вектор
    user_vector = np.array([
        user_params['Time_s'],
        user_params['Positive_count'],
        user_params['Negative_count']
    ])
    
    # Вибираємо метрику
    distance_functions = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'chebyshev': chebyshev_distance,
        'cosine': cosine_similarity
    }
    
    distance_func = distance_functions.get(metric, euclidean_distance)
    
    # Обчислюємо відстані до всіх відео
    distances = []
    for idx, row in df_clean.iterrows():
        video_vector = np.array([
            row['Time_s'],
            row['Positive_count'],
            row['Negative_count']
        ])
        
        distance = distance_func(user_vector, video_vector)
        distances.append({
            'Record': row['Record'],
            'Time_s': row['Time_s'],
            'Positive_count': row['Positive_count'],
            'Negative_count': row['Negative_count'],
            'Distance': distance
        })
    
    # Сортуємо за відстанню
    distances.sort(key=lambda x: x['Distance'])
    
    # Беремо k найближчих
    recommendations = distances[:min(k, len(distances))]
    
    return recommendations


def print_recommendations(recommendations, metric):
    """
    Вивід рекомендацій у красивому форматі
    """
    print("\n" + "="*80)
    print("🎬 РЕКОМЕНДОВАНІ ВІДЕО (5 найближчих)")
    print(f"   Метрика: {metric}")
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['Record']}")
        print(f"   Time_s: {rec['Time_s']:.2f} сек")
        print(f"   Positive_count: {rec['Positive_count']:.0f}")
        print(f"   Negative_count: {rec['Negative_count']:.0f}")
        print(f"   Відстань: {rec['Distance']:.4f}")


def save_recommendations(recommendations, user_params, output_filename, metric):
    """
    Збереження рекомендацій у файл
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("СИСТЕМА РЕКОМЕНДАЦІЙ ВИДЕО\n")
        f.write("="*60 + "\n\n")
        
        f.write("Параметри пошуку користувача:\n")
        f.write("-"*60 + "\n")
        f.write(f"Time_s: {user_params['Time_s']:.2f} сек\n")
        f.write(f"Positive_count: {user_params['Positive_count']:.0f}\n")
        f.write(f"Negative_count: {user_params['Negative_count']:.0f}\n")
        f.write(f"Метрика: {metric}\n\n")
        
        f.write("РЕКОМЕНДОВАНІ ВІДЕО (5 найближчих):\n")
        f.write("-"*60 + "\n\n")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec['Record']}\n")
            f.write(f"   Time_s: {rec['Time_s']:.2f} сек\n")
            f.write(f"   Positive_count: {rec['Positive_count']:.0f}\n")
            f.write(f"   Negative_count: {rec['Negative_count']:.0f}\n")
            f.write(f"   Відстань: {rec['Distance']:.4f}\n\n")
    
    print(f"\n💾 Рекомендації збережено: {output_filename}")


def main():
    """
    Основна функція програми
    """
    print("\n" + "="*80)
    print("🎬 РЕКОМЕНДАЦІЙНА СИСТЕМА ДЛЯ ВІДЕОХОСТИНГУ")
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
        
        # Показуємо діапазони параметрів
        print("\n" + "="*80)
        print("📊 ДІАПАЗОНИ ПАРАМЕТРІВ У БАЗІ:")
        print("="*80)
        
        df_clean = df.dropna(subset=['Time_s', 'Positive_count', 'Negative_count'])
        
        print(f"\nTime_s (тривалість в секундах):")
        print(f"  Min: {df_clean['Time_s'].min():.2f}, Max: {df_clean['Time_s'].max():.2f}")
        
        print(f"\nPositive_count (позитивні оцінки):")
        print(f"  Min: {df_clean['Positive_count'].min():.0f}, Max: {df_clean['Positive_count'].max():.0f}")
        
        print(f"\nNegative_count (негативні оцінки):")
        print(f"  Min: {df_clean['Negative_count'].min():.0f}, Max: {df_clean['Negative_count'].max():.0f}")
        
        # Запитуємо параметри користувача
        print("\n" + "="*80)
        print("🎯 ВВЕДІТЬ ПАРАМЕТРИ ДЛЯ ПОШУКУ РЕКОМЕНДАЦІЙ:")
        print("="*80)
        
        while True:
            try:
                time_s = float(input("\n▶ Time_s (тривалість в секундах): "))
                positive_count = float(input("▶ Positive_count (позитивні оцінки): "))
                negative_count = float(input("▶ Negative_count (негативні оцінки): "))
                break
            except ValueError:
                print("❌ Потрібно ввести числа!")
        
        user_params = {
            'Time_s': time_s,
            'Positive_count': positive_count,
            'Negative_count': negative_count
        }
        
        # Вибір метрики
        print("\n" + "="*80)
        print("📊 ОБЕРІТЬ МЕТРИКУ ВІДСТАНІ:")
        print("="*80)
        print("  1. Евклідова відстань")
        print("  2. Манхеттенська відстань")
        print("  3. Відстань Чебишева")
        print("  4. Косинус схожості")
        
        metric_choice = input("\n▶ Ваш вибір (1-4): ").strip()
        
        metric_map = {
            '1': 'euclidean',
            '2': 'manhattan',
            '3': 'chebyshev',
            '4': 'cosine'
        }
        
        metric = metric_map.get(metric_choice, 'euclidean')
        
        # Знаходимо рекомендації
        recommendations = find_recommendations(df, user_params, k=5, metric=metric)
        
        if recommendations is None:
            return
        
        # Виводимо результати
        print_recommendations(recommendations, metric)
        
        # Зберігаємо результати
        output_filename = "Recommendations.txt"
        
        save_choice = input(f"\n💾 Зберегти рекомендації у '{output_filename}'? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_recommendations(recommendations, user_params, output_filename, metric)
            print("✅ Готово!")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
