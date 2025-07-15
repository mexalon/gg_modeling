import numpy as np
from pde import  PDEBase, FieldCollection, ScalarField, VectorField, CartesianGrid
from pde import  solve_poisson_equation


class Unstable_Two_Phase_Gas_Grav(PDEBase):
  
    def __init__(self, **kwargs):
        super().__init__()
        # параметры модели
        self.t_range = 1000 # запихал это сюда, чтобы все было в одном месте, 
        self.timestep = 10  # чтобы было удобно сохранить весь обьект и не морочиьтся со всякими ямлами 
        self.shape = (64, 160, 2) # points
        self.sides = ((0, 6.4), (0, 16), (0, 0.2)) # meters
        
        # геометрия области гидрата
        self.H = 10 # глубина верхней границы гидратов, от поверхности дна, метры
        self.b = 2 # толщина области разложения газогидрата, метры
         
        # параметры среды
        self.k = 0.1 # Darcy проницаемсть
        self.m = 0.4 # поистость

        # параметры флюидов
        self.ro_gas = 1.28 # начальная плотность газа kg/m3
        self.ro_liq = 1000 # плотность жидкоти kg/m3
        self.ro_solid = 1900 # плотность грунта kg/m3

        self.nu_gas = 0.01 # вязкость газа cP
        self.nu_liq = 1 # вязкость жидкости cP

        # эффективный коэффициент диффузии для газа
        '''
        Iversen, N., and Jørgensen, B. B. (1993). Diffusion coefficients of sulfate and
        methane in marine sediments: Influence of porosity. Geochim. Cosmochim. Acta 57
        (3), 571–578. doi: 10.1016/0016-7037(93)90368-7
        '''
        self.D = 3e-9 

        # давление
        self.P0 = 0.13 # MPa press - давление на нулевой глубине - 20 метров = 0.1 + 0.2 Мпа
        self.P_gas = 0.0 # MPa давление в газовой области (превышение над гидростатическим)
        self.Pc = 0.001 # МPа капиллярное давление, Pc = 2*alpha/r alpha - коэффициент пов. нат., r - характерный размер пор. можно оценить r~sqrt(k/m)
        '''
        k = 1 D => r = sqrt(2) mkm => Pc = 0.056 MPa (alpha=0.04 N/m, 1 Pa = 1N/m2) 
        k = 0.1 D => Pc = 0.18 MPa 
        k = 0.01 D => Pc = 0.56 MPa 
        '''

        # начальная насыщенность
        self.s0 = 0.05 # во всей области
        self.s_star = 0.2 # миниммальная подвижная насыщенность 
        self.s_gas = 0.2 # начальная насыщенность в области с газом (почти равна s_sta чтобы не ждать, когда газ до неё докопится)
        self.s_eps = 0.0001  # минимально возможная газонасыщенность за колторую заходить нельзя


        # распределённый источник газа 
        self.q = 1 * 1e-7 # это m*d(s*ro)/dt, если выделение массы газа в единице объёма (в 1 кг куб. метре) в секунду, то q=1 кг/м3*сек

        # апдейт параметров из кваргов
        self.__dict__.update(kwargs) # Обновляем параметры моделирования из кваргов

        '''задача начальных полей'''
        # generate grid
        self.grid = CartesianGrid(self.sides, self.shape)  

        # выделенная область с газом
        xyz = self.grid.cell_coords # координаты сетки
        total_H = self.sides[1][1] # общая глубина модели
        xh, yh, zh = np.where(
            (xyz[:,:,:,1] <= total_H - self.H) & 
            (xyz[:,:,:,1] >= total_H - self.H - self.b) ) # координаты области с газогидратом

        # поле проницаемости со случайными вариациями 
        k = np.ones(self.shape) * self.k 
        self.k_field = ScalarField(self.grid, data=k)

        # g field  - поле силы тяжести
        g = np.zeros((3,) + self.shape) # 
        g[1,:] = - 9.81 * 1e-6 # gravity by Y ax; 1e-6 - to be good with pressire in MPa
        self.g_field = VectorField(self.grid, data=g) # grav field

        # hydrostatic field - гидростатическое поле давления
        self.ro_g_h = self.get_ro_g_h(self.ro_liq)

        # Pore initial field - начальное поровое давление газа
        P_ini = np.ones(self.shape) * self.ro_g_h + self.P0 + self.Pc # давление во всём объёме
        P_ini[xh, yh, zh] = P_ini[xh, yh, zh] + self.P_gas # давлеие в области с газом 
        self.p_ini_field = ScalarField(self.grid, data=P_ini) # MPa

        # s_gas initial field - начальное поле насыщенности газом
        s_ini = np.ones(self.shape) * self.s0 # насыщенность во всём объёме
        s_ini[xh, yh, zh] = self.s_gas # насыщенность в области с газом 
        self.s_ini_field = ScalarField(self.grid, data=s_ini) 

        # source field - распределённый источник газа. q по сути это dro/dt в области выделения, если выделение газа 1 кг в куб. метре в секунду, то q=1 кг/м3*сек
        q = np.zeros(self.shape)
        q[xh, yh, zh] = self.q 
        self.source_field = ScalarField(self.grid, data=q) 

        # boundary condition
        ro_g = 9.81 * 1e-6 * self.ro_liq
        self.p_gas_bc = [{'derivative': 0}, [{'derivative': ro_g}, {'value': self.P0 + self.Pc}], {'derivative': 0}]

    # метод для вычисления гидростатического давления, плотность должна быть в кг/м3
    def get_ro_g_h(self, ro):
        ro_g = self.g_field.to_scalar().data * ro * self.grid.cell_volume_data[1] 
        ro_g_h = np.cumsum(ro_g, axis=1) - ro_g/2 # гидростатическое давление, среднее по ячейке, поэтому минус половина ro_g
        ro_g_h = np.flip(ro_g_h, axis=1) # Y направлена снизу вверх
        return ro_g_h

    # относительные фазовые проницаемости: 
    # для газа
    def k_s_gas(self, s, ks_min=0):
        sd = s.data
        s_star = np.ones_like(sd) * self.s_star # минимальная подвижная насыщенность, при меньшей проницаемость ks_min, при большей - линейная
        ks = (sd-s_star)/(1-s_star)
        ks[ks < ks_min] = ks_min
        return ScalarField(self.grid, data=ks)
    
    # # для жидкости
    def k_s_liq(self, s):
        return s # просто s, смачивающая фаза
          
    # уравнение состояния для газа
    def P(self, ro):
        P = ro * 0.1 / self.ro_gas # 0.1 МПа
        return P
    
    def ro(self, P):
        ro = P * self.ro_gas / 0.1
        return ro    

    def get_P_liq(self, P_gas):
        return P_gas - self.Pc

    # делаем, чтобы изменение насыщенности оставалось в рамках [eps, 1-eps] 
    def sanity_check(self, s_field, ds_dt_field, d_dt_field):
        '''  
        попвтка заставить не уходить вличины за рамки разумного.
        При приближении к гранчным значениям насыщенности 
        поле d_dt_field должно пересатавть меняться там, 
        где насыщенность продолжает меняться в ненужную сторону
        '''
        s = s_field.data
        ds_dt = ds_dt_field.data
        d_dt = d_dt_field.data
        
        d_dt[(s <= self.s_eps) & (ds_dt < 0 )] = 0
        d_dt[(s >= 1 - self.s_eps) & (ds_dt > 0 )] = 0
            
        return ScalarField(self.grid, data=d_dt)
    
    # вся магия тут
    def evolution_rate(self, state, t=0):
        ''' Basniev pp.256'''
        P_gas, s_gas = state # искомые поля - давление и насыщенность газом
        s_liq = 1-s_gas # насыщенность жидкости = 1 - s газа
        ro = self.ro(P_gas) # распреджеление плотности газа исходя из давления

        kk_gas = - (1e-3 * self.k_field / self.nu_gas) * self.k_s_gas(s_gas) # то, на что умножается градиент в законе Дарси
        kk_liq = - (1e-3 * self.k_field / self.nu_liq) * self.k_s_liq(s_liq)

        # градиент от этого (просто проницаемость за скобку для стабильности)
        grad_kk_gas = - (1e-3 * self.k_field / self.nu_gas) * self.k_s_gas(s_gas).gradient({'derivative': 0}) 
        grad_kk_liq = - (1e-3 * self.k_field / self.nu_liq) * self.k_s_liq(s_liq).gradient({'derivative': 0})

        grad_P_gas = P_gas.gradient(self.p_gas_bc) 
        grad_P_liq = grad_P_gas

        laplace_P_gas = P_gas.laplace(self.p_gas_bc) # лаплас, градиент ro*g равен нулю
        laplace_P_liq = laplace_P_gas

        div_w1 = kk_gas * laplace_P_gas + (grad_P_gas - ro * self.g_field) @ grad_kk_gas # дивергенция от закона Дарси по правиду дивергенции произведения скалярного и векторного поля
        div_w2 = kk_liq * laplace_P_liq + (grad_P_liq - self.ro_liq * self.g_field) @ grad_kk_liq

        s_diff = self.D * s_gas.laplace({'derivative': 0}) # диффузия газа, чтобы было немного ровнее с градиентом газонасыщенности

        ds_gas_dt = (1/self.m) * div_w2 + s_diff # уравнение на изменение насыщенности 
         
        source = self.source_field # поле источника газа
        dro_dt = (s_gas * self.m)**-1 * (source - ro * (div_w1 + div_w2)) # пьезопроводность (относительно плотности газа)
        # dro_dt = (self.s_star * self.m)**-1 * (source - ro * (div_w1 + div_w2)) # ускорение

        # регуляризация производной насыщенности, чтобы за границы не выходила (в таком порядке)
        dro_dt = self.sanity_check(s_gas, ds_gas_dt, dro_dt)
        ds_gas_dt = self.sanity_check(s_gas, ds_gas_dt, ds_gas_dt)
         
        dP_dt = self.P(dro_dt) # переводим изменение плотности в изменение давления

        return FieldCollection([dP_dt, ds_gas_dt])



'''
Vanilla two phase flow like in Basniev book
'''

class Two_Phase_Vanilla(PDEBase):

    def __init__(self):
        super().__init__()
        self.eps = 1e-3  # just 0.001
        self.shape = (40, 100, 2) # points
        self.sides = ((0, 4), (0, 10), (0, 1)) # meters
        
        self.O = (2, 2, 0.5) # центр области с газогидратом, метры
        self.R0 = 1 # радиус области разложения газогидрата, метры

        self.t_scale = 3600 # sec - масштаб времени

        # параметры среды
        self.k = 0.01 # Darcy проницаемсть
        self.m = 0.5 # поистость

        # параметры флюидов
        self.ro_gas = 1.28 # начальная плотность газа kg/m3
        self.ro_liq = 1000 # плотность жидкоти kg/m3

        self.nu_gas = 0.01 / self.t_scale # вязкость газа cP
        self.nu_liq = 1 / self.t_scale # вязкость жидкости cP

        # начальная насыщенность
        self.s0 = 0 # во всей области
        self.s0_gas = 0.2 # начальная насыщенность в области с газом
    
        # пороговый градиент
        self.G = 0.01 # МПа/м  - равен гидростатики например

        # распределённый источник газа 
        self.q = 1e-6 * self.t_scale # это dro/dt, если выделение газа 1 кг в куб. метре в секунду, то q=1 кг/м3*сек

        # generate grid
        self.grid = CartesianGrid(self.sides, self.shape)  

        # выделенная область с газом
        xyz = self.grid.cell_coords # координаты сетки
        xh, yh, zh = np.where(self.grid.distance(xyz, self.O) <= self.R0) # координаты области с газогидратом

        # поле проницаемости со случайными вариациями 
        rand_k = np.ones(self.shape) * self.k 
        rand_k[:,3:-3,:]  = rand_k[:,3:-3,:] + 0.2 * np.random.randn(*(rand_k[:,3:-3,:].shape)) * self.k 
        self.k_field = ScalarField(self.grid, data=rand_k)

        # g field  - поле силы тяжести
        g = np.zeros((3,) + self.shape) # 
        g[1,:] = - 9.81 * 1e-6 # gravity by Y ax; 1e-6 - to be good with pressire in MPa
        self.g_field = VectorField(self.grid, data=g) # grav field

        # s initial field - начальное поле насыщенности газом
        s_ini = np.ones(self.shape) * self.s0 # насыщенность во всём объёме
        s_ini[xh, yh, zh] = self.s0_gas # насыщенность в области с газом 
        self.s_ini_field = ScalarField(self.grid, data=s_ini) 

        # source field - распределённый источник газа. q по сути это dro/dt в области выделения, если выделение газа 1 кг в куб. метре в секунду, то q=1 кг/м3*сек
        q = np.zeros(self.shape)
        q[xh, yh, zh] = self.q 
        self.source_field = ScalarField(self.grid, data=q) 

        # boundary condition       
        # hydrostatic field - гидростатическое поле давления
        self.ro_g_h = self.get_ro_g_h(self.ro_liq)
        P_up = self.ro_g_h[:,-1,:]
        P_down = self.ro_g_h[:,0,:]
        self.p_bc = [{'derivative': 0}, [{'value':P_down}, {'value':P_up}], {'derivative': 0}] # граничное условие на давление (непроницаемые границы, сверху сток)
        self.s_bc = {'derivative': 0} # граничные условия на S 

    # метод для вычисления гидростатического давления, плотность должна быть в кг/м3
    def get_ro_g_h(self, ro):
        ro_g = self.g_field.to_scalar().data * ro * self.grid.cell_volume_data[1] 
        ro_g_h = np.cumsum(ro_g, axis=1) - ro_g/2 # гидростатическое давление, среднее по ячейке, поэтому минус половина ro_g
        ro_g_h = np.flip(ro_g_h, axis=1) # Y направлена снизу вверх
        return ro_g_h

    # относительные фазовые проницаемости: 
    # для газа
    def k_s_gas(self, s):
        return s
    
    # для жидкости
    def k_s_liq(self, s):
        return 1 - s # s - насыщенность газа!!!
    
    def f(self, s):
        fs = s/(s + (self.nu_gas/self.nu_liq) * (1-s))
        return fs
    
    def df_ds(self, s):
        dfs_ds = (self.nu_gas/self.nu_liq)/((s + (self.nu_gas/self.nu_liq) * (1-s))**2)
        return dfs_ds
        
    # обобщённый закон Дарси для Газа
    def darcy_gas(self, grad_P, ro, nu, s):
        a = 1e-3 * self.k_field * self.k_s_gas(s) / nu #  const, [k] = [D], [nu] = cP => 1e-3 to be good with pressure gradient in MPa/m
        g = self.g_field
        w = - a * (grad_P - ro * g) # Darcy law
        return w
    
    # обобщённый закон Дарси для Жидкости
    def darcy_liq(self, grad_P, ro, nu, s):
        "s - это насыщенность газа!"
        a = 1e-3 * self.k_field * self.k_s_liq(s) / nu #  const, [k] = [D], [nu] = cP => 1e-3 to be good with pressure gradient in MPa/m
        g = self.g_field
        w = - a * (grad_P - ro * g) # Darcy law
        return w
    
    # обобщённый закон Дарси для жидкости с пороговым градиентом G (пока постоянным)
    def darcy_liq_TPG(self, grad_P, ro, nu, s):
        "s - это насыщенность газа!"
        a = 1e-3 * self.k_field * self.k_s_liq(s) / nu #  const, [k] = [D], [nu] = cP => 1e-3 to be good with pressure gradient in MPa/m
        g = self.g_field
        grad = grad_P - ro * g # обычный градиент
        gd = grad.to_scalar().data # модуль вектора обычного градиента
        Gd = np.ones_like(gd) * self.G # модуль G
        k_tpg = np.max((1 - Gd/gd, np.zeros_like(gd)), axis=0) # ноль если градиент меньше G, градиент минус G, если больше
        w = - a * k_tpg * grad  # TPG + Darcy law
        return w 

    # делаем насыщенность от eps до 1-eps 
    def check_s(self, s):
        sd = s.data
        eps = np.ones_like(sd) * self.eps
        new_s = np.max((sd, eps), axis=0)
        new_s = np.min((new_s, 1-eps), axis=0)
        return ScalarField(self.grid, data=new_s)
    
    # делаем, чтобы изменение насыщенности было в рамках [eps, 1-eps] 
    def check_ds_dt(self, s, ds_dt):
        sd = s.data
        ds_dtd = ds_dt.data
        eps = np.ones_like(sd) * self.eps
        new_ds_dt = np.zeros_like(sd)
        valid_ds_dt = (sd + ds_dtd > eps) & (sd + ds_dtd < 1 - eps)
        new_ds_dt[valid_ds_dt] = ds_dtd[valid_ds_dt]
        return ScalarField(self.grid, data=new_ds_dt)
    
    # метод для решения уравнения Пуассона для давления
    def solve_P_eq(self, s):
        s = self.check_s(s)
        q = self.source_field
        rhs = - (q/self.ro_gas) * self.nu_gas * self.f(s)/self.k_s_gas(s)
        P = solve_poisson_equation(rhs, self.p_bc)
        return P
    

    # вся магия тут
    def evolution_rate(self, state, t=0):
        ''' Basniev pp.256'''
        s = state 
        s = self.check_s(s)

        P = self.solve_P_eq(s)

        grad_P = P.gradient(self.p_bc) # он одинаковый для обеих фаз, так ка давления отличаются на константу

        w1 = self.darcy_gas(grad_P, 0, self.nu_gas, s) # скорость фильтрации газа. 
        # тут плотность ноль, чтобы пренебречь гидростатикой, так как она много меньше по сравнению с гидростатикой жидкости 

        w2 = self.darcy_liq(grad_P, self.ro_liq, self.nu_liq, s) # скорость фильтрации жидкости по дарси, либо 
        # w2 = self.darcy_liq_TPG( grad_P_liq,  self.ro_liq, self.nu_liq, s) # по Дарси с пороговым градиентом

        q = self.source_field # поле источника газа
        ds_dt = - (1/self.m) * self.df_ds(s)*((w1 + w2) @ s.gradient(self.s_bc)) + (q/self.ro_gas) * (1 - self.f(s))

        ds_dt = self.check_ds_dt(s, ds_dt) # на всякий случай обрезаем изменение насыщенности, чтобы она не выходила за границы разумного

        return ds_dt