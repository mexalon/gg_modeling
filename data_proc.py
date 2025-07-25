import numpy as np

''' 
Функции для обработки данных
'''

''' Генерируем слоистую проницаемость _________________________________________________________________'''
def generate_rand_k(eq, alpha=0.001):
    rand_k = np.ones(eq.shape) * eq.k
    rand_k[:,3:-3,:]  = rand_k[:,3:-3,:] + alpha * np.random.randn(*(rand_k[:,3:-3,:].shape)) * eq.k 
    lines = np.random.randn(1, rand_k.shape[1], 1)
    rand_k = rand_k * (1 + alpha*(lines))
    return rand_k

''' Конвертим проницаемсть в размер и капиллярное давление ___________________________________________ '''
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

''' Манипуляции с фронтом насыщенности ________________________________________________________________ '''
def get_frac_depth_and_time_idx(eq, press):
    """
    Находит индексы первого превышения гидростатики над литостатикой
    """
    solid_ro_g_h = eq.get_ro_g_h(eq.ro_solid) + eq.P0 # литостатика

    mask = (press - eq.Pc > solid_ro_g_h).any(axis=(1, 3))
    t_idx, y_idx = np.where(mask) # индексы где не ноль
    # 3) Сортируем сначала по времени, затем по глубине
    order = np.lexsort((y_idx, t_idx))
    t_frac_idx, y_frac_idx = int(t_idx[order[0]]), int(y_idx[order[0]]) # индексы первого вхождения 

    return t_frac_idx, y_frac_idx 

def get_depth_and_time(eq, t_idx, y_idx):
    '''Возвращает время в сутках и глубину в метрах по индексам t и y'''
    y_ax = [np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides)][1]
    y_ax = np.flip(y_ax)
    t = t_idx * eq.timestep * eq.t_scale/(3600*24)
    depth = y_ax[y_idx] # метров под поверхностью дна
    
    return t, depth

def get_front_indices(arr: np.ndarray, tresh: float) -> np.ndarray:
    """
    Находит максимальный индекс, на которой концентрация превышает пороговое значение. Это для движения фронта и годографа
    """
    if arr.ndim == 4:
        yy = np.ones(arr.shape)*np.arange(arr.shape[2])[None,None,:,None]
        mask = (arr >= tresh).astype(int) 
        depth_idxs = np.max(np.argmax(yy*mask, axis=2), axis=(1,2))
    elif arr.ndim == 3:
        yy = np.ones(arr.shape)*np.arange(arr.shape[2])[None,None,:]
        mask = (arr >= tresh).astype(int) 
        depth_idxs = np.max(np.argmax(yy*mask, axis=2), axis=(1,))

    return depth_idxs

def get_relese_time(sat, tresh, eq):
    ''' момент выходв примеси на поверхность в сутках '''
    depth_idxs = get_front_indices(sat, tresh)
    relese_time = np.argmax(depth_idxs==eq.shape[1]-1) * eq.timestep * eq.t_scale/(3600*24)
    return relese_time 

