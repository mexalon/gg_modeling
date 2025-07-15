import numpy as np
import itertools

# генерируем слоистую проницаемость
def generate_rand_k(eq, alpha=0.001):
    rand_k = np.ones(eq.shape) * eq.k
    rand_k[:,3:-3,:]  = rand_k[:,3:-3,:] + alpha * np.random.randn(*(rand_k[:,3:-3,:].shape)) * eq.k 
    lines = np.random.randn(1, rand_k.shape[1], 1)
    rand_k = rand_k * (1 + alpha*(lines))
    return rand_k

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


'''  
Конвертим проницаемсть в размер и капиллярное давление
'''
def d_to_k(d_mm: float, m=0.4) -> float:
    """
    Пермеабельность k [Darcy] из размера зёрен d [мм] по Козени-Кармену.
    """
    DARCI_TO_M2 = 9.869233e-13          #   1 Darcy  → м²
    d_m = d_mm * 1e-3                                     # мм → м
    k_m2 = (m**3 * d_m**2) / (180 * (1 - m)**2)       # базовая Kозени-Карман-формула
    return k_m2 / DARCI_TO_M2                         # м² → Darcy

def k_to_d(k_darcy: float, m=0.4) -> float:
    """
    Размер зёрен d [мм] из проницаемости k [Darcy] по Козени-Кармену.
    """
    DARCI_TO_M2 = 9.869233e-13          #   1 Darcy  → м²
    k_m2 = k_darcy * DARCI_TO_M2                          # Darcy → м²
    d_m  = np.sqrt(k_m2 * 180 * (1 - m)**2 / m**3)   # обратная KC-формула
    return d_m * 1e3  # мм

def r_to_Pc(d_mm: float, alpha=0.04):
    Pc = 2*alpha/(d_mm*1e-3)*1e-6 # MPa
    return Pc