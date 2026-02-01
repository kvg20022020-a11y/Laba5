import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter

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


def create_video_keywords():
    """
    Створити словник ключових слів для відео
    """
    keywords_dict = {
        'Video1': ['музика', 'клип', 'популяр'],
        'Video2': ['спорт', 'футбол', 'гол'],
        'Video3': ['природа', 'пейзаж', 'красиво'],
        'Video4': ['технологія', 'гаджет', 'новинка'],
        'Video5': ['комедія', 'смішно', 'розваги'],
        'Video6': ['освіта', 'навчання', 'курс'],
        'Video7': ['музика', 'хіп-хоп', 'реп'],
        'Video8': ['кулінарія', 'рецепт', 'готування'],
        'Video9': ['подорожі', 'туризм', 'країни'],
        'Video10': ['спорт', 'тренування', 'фітнес'],
        'Video11': ['кино', 'трейлер', 'фільм'],
        'Video12': ['музика', 'концерт', 'рок'],
        'Video13': ['комедія', 'гумор', 'жарти'],
        'Video14': ['природа', 'тварини', 'дикі'],
        'Video15': ['мода', 'стиль', 'одяг'],
        'Video16': ['освіта', 'лекція', 'лекції'],
        'Video17': ['спорт', 'баскетбол', 'гра'],
        'Video18': ['музика', 'поп', 'пісня'],
        'Video19': ['комедія', 'скетч', 'серіал'],
        'Video20': ['технологія', 'програмування', 'код'],
        'Video21': ['кулінарія', 'десерт', 'солодке'],
        'Video22': ['подорожі', 'пригода', 'відкриття'],
        'Video23': ['кино', 'драма', 'емоційно'],
        'Video24': ['музика', 'джаз', 'класика'],
        'Video25': ['спорт', 'теніс', 'матч'],
        'Video26': ['природа', 'ліс', 'озеро'],
        'Video27': ['освіта', 'наука', 'експеримент'],
        'Video28': ['комедія', 'карикатура', 'мультфільм'],
        'Video29': ['подорожі', 'екзотика', 'далекі'],
        'Video30': ['музика', 'соул', 'вокал'],
        'Video31': ['спорт', 'волейбол', 'команда'],
        'Video32': ['кулінарія', 'м\'ясо', 'стейк'],
        'Video33': ['кино', 'трилер', 'напруга'],
        'Video34': ['природа', 'гори', 'вершина'],
        'Video35': ['технологія', 'штучний інтелект', 'нейромережа'],
        'Video36': ['освіта', 'історія', 'виклад'],
        'Video37': ['комедія', 'пародія', 'віддзеркалення'],
        'Video38': ['подорожі', 'море', 'пляж'],
        'Video39': ['музика', 'блюз', 'грусть'],
        'Video40': ['спорт', 'легка атлетика', 'біг']
    }
    
    return keywords_dict


def jaccard_similarity(set1, set2):
    """
    Обчислення Jaccard схожості між двома наборами ключових слів
    Формула: |A ∩ B| / |A ∪ B|
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


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


def combined_distance(user_vector, video_vector, user_keywords, video_keywords, 
                     alpha=0.7, metric='euclidean'):
    """
    Комбінована метрика: числові параметри + ключові слова
    
    Parameters:
    -----------
    alpha : float (0-1)
        Вага для числових параметрів (1-alpha для ключових слів)
    """
    
    # Обчислюємо відстань за числовими параметрами
    if metric == 'euclidean':
        numerical_distance = euclidean_distance(user_vector, video_vector)
    else:
        numerical_distance = manhattan_distance(user_vector, video_vector)
    
    # Нормалізуємо до [0, 1]
    normalized_numerical = numerical_distance / (1 + numerical_distance)
    
    # Обчислюємо Jaccard схожість за ключовими словами
    jaccard = jaccard_similarity(user_keywords, video_keywords)
    
    # Перетворюємо схожість на відстань (1 - схожість)
    keywords_distance = 1 - jaccard
    
    # Комбінуємо
    combined = alpha * normalized_numerical + (1 - alpha) * keywords_distance
    
    return combined, normalized_numerical, keywords_distance, jaccard


def find_recommendations_with_keywords(df, user_params, user_keywords, k=5, 
                                      alpha=0.7, metric='euclidean', keywords_dict=None):
    """
    Знайти k найближчих відео з урахуванням ключових слів
    """
    
    # Вибираємо тільки повні рядки
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
    
    # Набір ключових слів користувача
    user_keywords_set = set(word.lower() for word in user_keywords)
    
    # Обчислюємо відстані до всіх відео
    distances = []
    for idx, row in df_clean.iterrows():
        video_vector = np.array([
            row['Time_s'],
            row['Positive_count'],
            row['Negative_count']
        ])
        
        # Ключові слова для відео
        video_name = row['Record']
        video_keywords = set(w.lower() for w in keywords_dict.get(video_name, []))
        
        # Комбінована метрика
        combined, numerical, keywords_dist, jaccard = combined_distance(
            user_vector, video_vector, 
            user_keywords_set, video_keywords,
            alpha=alpha, metric=metric
        )
        
        distances.append({
            'Record': video_name,
            'Time_s': row['Time_s'],
            'Positive_count': row['Positive_count'],
            'Negative_count': row['Negative_count'],
            'Keywords': list(keywords_dict.get(video_name, [])),
            'Combined_Distance': combined,
            'Numerical_Distance': numerical,
            'Keywords_Distance': keywords_dist,
            'Jaccard_Similarity': jaccard
        })
    
    # Сортуємо за комбінованою метрикою
    distances.sort(key=lambda x: x['Combined_Distance'])
    
    # Беремо k найближчих
    recommendations = distances[:min(k, len(distances))]
    
    return recommendations


def print_recommendations(recommendations, user_keywords, alpha):
    """
    Вивід рекомендацій
    """
    print("\n" + "="*80)
    print("🎬 РЕКОМЕНДОВАНІ ВІДЕО (5 найближчих)")
    print(f"   Ваші ключові слова: {', '.join(user_keywords)}")
    print(f"   Вага числових параметрів: {alpha*100:.0f}%, Ключові слова: {(1-alpha)*100:.0f}%")
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['Record']}")
        print(f"   Параметри: Time={rec['Time_s']:.0f}s, Positive={rec['Positive_count']:.0f}, Negative={rec['Negative_count']:.0f}")
        print(f"   Ключові слова: {', '.join(rec['Keywords'])}")
        print(f"   Jaccard схожість (ключові слова): {rec['Jaccard_Similarity']:.2%}")
        print(f"   Комбінована відстань: {rec['Combined_Distance']:.4f}")
        print(f"     - Числові параметри: {rec['Numerical_Distance']:.4f}")
        print(f"     - Ключові слова: {rec['Keywords_Distance']:.4f}")


def save_recommendations(recommendations, user_params, user_keywords, alpha, 
                        output_filename, keywords_dict):
    """
    Збереження рекомендацій у файл
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("УДОСКОНАЛЕНА СИСТЕМА РЕКОМЕНДАЦІЙ ВИДЕО\n")
        f.write("З УРАХУВАННЯМ КЛЮЧОВИХ СЛІВ\n")
        f.write("="*60 + "\n\n")
        
        f.write("Параметри пошуку користувача:\n")
        f.write("-"*60 + "\n")
        f.write(f"Time_s: {user_params['Time_s']:.2f} сек\n")
        f.write(f"Positive_count: {user_params['Positive_count']:.0f}\n")
        f.write(f"Negative_count: {user_params['Negative_count']:.0f}\n")
        f.write(f"Ключові слова: {', '.join(user_keywords)}\n")
        f.write(f"Вага числових параметрів: {alpha*100:.0f}%\n")
        f.write(f"Вага ключових слів: {(1-alpha)*100:.0f}%\n\n")
        
        f.write("РЕКОМЕНДОВАНІ ВІДЕО (5 найближчих):\n")
        f.write("-"*60 + "\n\n")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec['Record']}\n")
            f.write(f"   Time_s: {rec['Time_s']:.2f} сек\n")
            f.write(f"   Positive_count: {rec['Positive_count']:.0f}\n")
            f.write(f"   Negative_count: {rec['Negative_count']:.0f}\n")
            f.write(f"   Ключові слова: {', '.join(rec['Keywords'])}\n")
            f.write(f"   Jaccard схожість: {rec['Jaccard_Similarity']:.2%}\n")
            f.write(f"   Комбінована відстань: {rec['Combined_Distance']:.4f}\n\n")
    
    print(f"\n💾 Рекомендації збережено: {output_filename}")


def main():
    """
    Основна функція програми
    """
    print("\n" + "="*80)
    print("🎬 УДОСКОНАЛЕНА СИСТЕМА РЕКОМЕНДАЦІЙ ВИДЕО")
    print("   З УРАХУВАННЯМ КЛЮЧОВИХ СЛІВ")
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
        
        # Завантажуємо ключові слова для відео
        keywords_dict = create_video_keywords()
        print(f"✓ Завантажено ключові слова для {len(keywords_dict)} відео")
        
        # Показуємо приклади ключових слів
        print("\n📚 ПРИКЛАДИ КЛЮЧОВИХ СЛІВ:")
        print("-"*60)
        for video, keywords in list(keywords_dict.items())[:5]:
            print(f"  {video}: {', '.join(keywords)}")
        print(f"  ... та інші\n")
        
        # Запитуємо параметри користувача
        print("="*80)
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
        
        # Запитуємо ключові слова
        print("\n▶ Введіть ключові слова (розділяйте комою):")
        print("   Приклади: музика, спорт, комедія, природа, технологія")
        keywords_input = input("▶ Ключові слова: ").strip()
        user_keywords = [kw.strip().lower() for kw in keywords_input.split(',')]
        
        user_params = {
            'Time_s': time_s,
            'Positive_count': positive_count,
            'Negative_count': negative_count
        }
        
        # Вибір ваги для числових параметрів
        print("\n📊 Вибір важливості компонентів:")
        print("  1. Числові параметри важливіші (75% вага числам, 25% ключовим словам)")
        print("  2. Рівна вага (50% на 50%)")
        print("  3. Ключові слова важливіші (25% вага числам, 75% ключовим словам)")
        
        alpha_choice = input("\n▶ Ваш вибір (1-3): ").strip()
        
        alpha_map = {
            '1': 0.75,
            '2': 0.5,
            '3': 0.25
        }
        
        alpha = alpha_map.get(alpha_choice, 0.5)
        
        # Знаходимо рекомендації
        recommendations = find_recommendations_with_keywords(
            df, user_params, user_keywords, k=5, 
            alpha=alpha, metric='euclidean', keywords_dict=keywords_dict
        )
        
        if recommendations is None:
            return
        
        # Виводимо результати
        print_recommendations(recommendations, user_keywords, alpha)
        
        # Зберігаємо результати
        output_filename = "Recommendations_with_Keywords.txt"
        
        save_choice = input(f"\n💾 Зберегти рекомендації у '{output_filename}'? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_recommendations(recommendations, user_params, user_keywords, 
                               alpha, output_filename, keywords_dict)
            print("✅ Готово!")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
