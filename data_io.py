import itertools
import pickle
import pandas as pd
import os


''' 
Функции для манипуляций с данными на диске или около того
'''

def generate_combinations(params):
    # Преобразуем значения словаря в списки, если они не являются списками
    params_lists = {k: v if isinstance(v, list) else [v] for k, v in params.items()}
    
    # Получаем все возможные комбинации параметров
    keys = params_lists.keys()
    values = params_lists.values()
    combinations = itertools.product(*values)
    
    # Создаем список словарей с комбинациями параметров
    result = [dict(zip(keys, combination)) for combination in combinations]
    
    return result

def dump_eq(fpath, obj):
    """ сохраняем eq в файл """
    with open(fpath , 'wb') as file:
        pickle.dump(obj, file)

def load_eq(fpath):
    """ Загружает объект из файла """
    with open(fpath, 'rb') as file:
        obj = pickle.load(file)
    return obj

def collect_csv_data(root_path, csv_name):
    """
    Для всех подпапок в root_path читает параметры из 'параметры расчёта.csv'
    и собирает сводный DataFrame, где строки — папки, а столбцы — параметры.
    
    Parameters:
        root_path (str): Путь к корневой папке с подпапками, содержащими CSV-файлы.
    
    Returns:
        pd.DataFrame: Сводная таблица параметров.
    """
    records = {}
    for sub in os.listdir(root_path):
        sub_path = os.path.join(root_path, sub)
        csv_path = os.path.join(sub_path, csv_name)
        if not os.path.isfile(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, sep=";", decimal=",").set_index("Параметр")
            records[sub] = df["Значение"].to_dict()
        except Exception as e:
            print(f"Ошибка при обработке {csv_path}: {e}")

    summary_df = pd.DataFrame.from_dict(records, orient="index")

    return summary_df