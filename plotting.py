import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from data_proc import get_depth_and_time, get_frac_depth_and_time_idx, get_front_indices



def plot_sat(eq, data, mask=None, vmin_vmax=None, title=None, fname=None):
    cmap = mpl.cm.viridis

    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides))  # km
    mid_z = eq.shape[-1]//2

    fig, ax = plt.subplots(figsize=(4.1, 3.05), layout="constrained")

    ax.imshow(data[:, :, mid_z].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[-1], y_ax[-0]], origin='lower', cmap=cmap, norm=norm)

    if mask is not None:
        ax.imshow(mask[:, :, mid_z].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[-1], y_ax[-0]], origin='lower', cmap="cool", alpha=mask[:, :, 0].transpose()) # mask

    if title:
        ax.set_title(title)
    else:
        ax.set_title(r'$Разрез$')

    ax.set_xlabel(r'$x,\ м$')
    ax.set_ylabel(r'$Глубина,\ м$')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label=r'$Газонасыщенность$')

    if fname:
        plt.savefig(fname, dpi = 300,  bbox_inches='tight', transparent=False)
    else:
        plt.show()
        
    plt.close()




def plot_press_curves(eq, press, plot_step=None, title=None, fname=None):
    steps = press.shape[0]
    if not plot_step:
        plot_step = steps//20

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = mpl.cm.inferno_r
    norm = mpl.colors.Normalize(vmin=0, vmax=steps * eq.timestep * eq.t_scale/(3600*24))

    y_ax = [np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides)][1]
    depth = np.flip(y_ax)

    ax.invert_yaxis()
  
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='horizontal', location='top', label=r'$Время,\ сут.$')

    ax_to_reduce = tuple(i for i in range(len(eq.shape)) if i != 1) # aaaah

    for ii in range(steps):
        hh = ii * eq.timestep * eq.t_scale/(3600*24)
        if ii % plot_step == 0:
          max_data = np.max(press[ii], axis = ax_to_reduce) - eq.Pc
          l1, = ax.plot(max_data, depth, color=cmap(norm(hh)))

    solid_ro_g_h = eq.get_ro_g_h(eq.ro_solid) + eq.P0 # литостатика
    lith = np.mean(solid_ro_g_h, axis = (-3, -1))
    l2, = ax.plot(lith, depth, color='r', linestyle = ':')

    ax.legend([l1, l2], [r'Давление жидкости', r'Литостатическое давление'])
    
    ax.set_xlabel(r'$Поровое\ давление,\ МПа$')
    ax.set_ylabel(r'$Глубина,\ м$')

    if title:
        ax.set_title(title)
    if fname:
        plt.savefig(fname, dpi = 300,  bbox_inches='tight', transparent=False)
    else:
        plt.show()
        
    plt.close()




def plot_sat_curves(eq, sat, plot_step=None, title=None, fname=None):
    steps = sat.shape[0]
    if not plot_step:
        plot_step = steps//20
        
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = mpl.cm.viridis_r
    norm = mpl.colors.Normalize(vmin=0, vmax=steps * eq.timestep * eq.t_scale/(3600*24))

    y_ax = [np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides)][1]
    depth = np.flip(y_ax)

    ax.invert_yaxis()
  
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='horizontal', location='top', label=r'$Время,\ сут.$')

    ax_to_reduce = tuple(i for i in range(len(eq.shape)) if i != 1) # aaaah

    for ii in range(steps):
        hh = ii * eq.timestep * eq.t_scale/(3600*24)
        if ii % plot_step == 0:
          max_data = np.mean(sat[ii], axis = ax_to_reduce)
          l1, = plt.plot(max_data, depth, color=cmap(norm(hh)))

    ax.set_xlabel(r'$Газонасыщенность$')
    ax.set_ylabel(r'$Глубина,\ м$')

    if title:
        ax.set_title(title)
    if fname:
        plt.savefig(fname, dpi = 300,  bbox_inches='tight', transparent=False)
    else:
        plt.show()
        
    plt.close()



def plot_concentration_front_with_frac(eq, press, sat, tresh, title=None, fname=None):
    """
    Строит график зависимости глубины фронта концентрации от времени. Добавляет точку, когда произошел разрыв
    """
    depth_idxs = get_front_indices(sat, tresh)
    y_ax = [np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides)][1]
    y_ax = np.flip(y_ax)
    depth = y_ax[depth_idxs] # Глубина фронта (ось Y)

    relese_time = np.argmax(depth_idxs==eq.shape[1]-1) * eq.timestep * eq.t_scale/(3600*24) # момент выхода фронта на поверхность в сутках

    # Время (ось X)
    time = np.arange(sat.shape[0]) * eq.timestep * eq.t_scale/(3600*24) # время в сутках

    # Построение графика
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.invert_yaxis()
    ax.plot(time[depth_idxs>0], depth[depth_idxs>0], linestyle='-', color='b', label='Глубина фронта')
    if relese_time:
        ax.axvline(relese_time, color='r', linestyle=':', label=f'Момент выхода на поверхность')
        
    t_frac_idx, y_frac_idx = get_frac_depth_and_time_idx(eq, press) # индексы разрыва
    t_frac, depth_frac = get_depth_and_time(eq, t_frac_idx, y_frac_idx) # превращает их в сутки и глубину
    front_at_t_frac = depth[t_frac_idx] # где был фронт в моментт возникновения разрыва

    ax.scatter(t_frac, depth_frac, marker='*', c='r', label=f'Момент возникновения разрыва' )
    
    ax.legend()
    ax.set_xlabel(r'$Время,\ сут.$')
    ax.set_ylabel(r'$Глубина\ фронта,\ м$')
    ax.grid(True)
     
    if title:
        ax.set_title(title)
    if fname:
        plt.savefig(fname, dpi = 300,  bbox_inches='tight', transparent=False)
    else:
        plt.show()
        
    plt.close()
    
    return relese_time, t_frac, depth_frac, front_at_t_frac



# без разрыва, на всякий случай оставлю это тут
def plot_concentration_front(eq, sat, tresh, title=None, fname=None):
    """
    Строит график зависимости глубины фронта концентрации от времени.
    """
    depth_idxs = get_front_indices(sat, tresh)
    y_ax = [np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides)][1]
    y_ax = np.flip(y_ax)
    depth = y_ax[depth_idxs] # Глубина фронта (ось Y)

    relese_time = np.argmax(depth_idxs==eq.shape[1]-1) * eq.timestep * eq.t_scale/(3600*24) # момент выхода фронта на поверхность в часаах

    # Время (ось X)
    time = np.arange(sat.shape[0]) * eq.timestep * eq.t_scale/(3600*24) # вчеря в часах

    # Построение графика
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.invert_yaxis()
    ax.plot(time[depth_idxs>0], depth[depth_idxs>0], linestyle='-', color='b', label='Глубина фронта')
    if relese_time:
        ax.axvline(relese_time, color='r', linestyle=':', label=f'Момент выхода {relese_time:.2f} сут.')
        
    ax.legend()
    ax.set_xlabel(r'$Время,\ сут.$')
    ax.set_ylabel(r'$Глубина\ фронта,\ м $')
    ax.grid(True)
     
    if title:
        ax.set_title(title)
    if fname:
        plt.savefig(fname, dpi = 300,  bbox_inches='tight', transparent=False)
    else:
        plt.show()
        
    plt.close()
    
    return relese_time


''' __________________________ Сводные графики ________________________________________________________________ '''

def plot_summary_front_velocities(df_all, fname=None):
    """
    Рисует сводный график скосростей подьема фронта
    """
    # названия полей в CSV
    FILT  = "Выделение массы газа в единице объёма, кг/м3*сек" # цветом
    XPAR  = "Проницаемость, Д"
    YPAR  = "Средняя скорость фронта,  м/сут"
    SPAR  = "Капиллярное давление, МПа" # штрихами

    # уникальные значения для отображения
    rates = sorted(df_all[FILT].unique())
    pcs = sorted(df_all[SPAR].unique())

    # цветовая схема для расстояний
    cmap = plt.get_cmap('tab10')
    color_map = {q: cmap(i / max(1, len(rates)-1)) for i, q in enumerate(rates)}

    # стили для пористостей
    line_styles = ['-', '--', '-.', ':']
    style_map = {pc: line_styles[i % len(line_styles)] for i, pc in enumerate(pcs)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # строим кривые
    for q in rates:
        for pc in pcs:
            grp = df_all[(df_all[FILT] == q) & (df_all[SPAR] == pc)].sort_values(XPAR)
            if grp.empty:
                continue
            ax.plot(grp[XPAR], grp[YPAR], marker='o',
                    color=color_map[q], linestyle=style_map[pc],
                    label=f"Q={q:.1e} $кг/м^3с$, P$_c$={pc} МПа")

    ax.set_xlabel(f"{XPAR}")
    ax.set_ylabel(f"{YPAR}")
    ax.set_xscale('log')

    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # сохраняем или показываем
    if fname:
        # сохраняем график
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)

    else:
        plt.show()




def plot_summary_front_relese_times(df_all, fname=None):
    """
    Рисует сводный график скосростей подьема фронта
    """
    # названия полей в CSV
    FILT  = "Выделение массы газа в единице объёма, кг/м3*сек" # цветом
    XPAR  = "Проницаемость, Д"
    YPAR  = "Время выхода фронта на поверхность, сут"
    SPAR  = "Капиллярное давление, МПа" # штрихами

    # уникальные значения для отображения
    rates = sorted(df_all[FILT].unique())
    pcs = sorted(df_all[SPAR].unique())

    # цветовая схема для расстояний
    cmap = plt.get_cmap('tab10')
    color_map = {q: cmap(i / max(1, len(rates)-1)) for i, q in enumerate(rates)}

    # стили для пористостей
    line_styles = ['-', '--', '-.', ':']
    style_map = {pc: line_styles[i % len(line_styles)] for i, pc in enumerate(pcs)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # строим кривые
    for q in rates:
        for pc in pcs:
            grp = df_all[(df_all[FILT] == q) & (df_all[SPAR] == pc)].sort_values(XPAR)
            if grp.empty:
                continue
            ax.plot(grp[XPAR], grp[YPAR], marker='o',
                    color=color_map[q], linestyle=style_map[pc],
                    label=f"Q={q:.1e} $кг/м^3с$, P$_c$={pc} МПа")

    ax.set_xlabel(f"{XPAR}")
    ax.set_ylabel(f"{YPAR}")
    ax.set_xscale('log')

    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # сохраняем или показываем
    if fname:
        # сохраняем график
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)

    else:
        plt.show()


        

def plot_summary_front_frac_depths(df_all, fname=None):
    """
    Рисует сводный график скосростей подьема фронта
    """
    # названия полей в CSV
    FILT  = "Выделение массы газа в единице объёма, кг/м3*сек" # цветом
    XPAR  = "Проницаемость, Д"
    YPAR  = "Глубина возникновения разрыва, м"
    SPAR  = "Капиллярное давление, МПа" # штрихами

    # уникальные значения для отображения
    rates = sorted(df_all[FILT].unique())
    pcs = sorted(df_all[SPAR].unique())

    # цветовая схема для расстояний
    cmap = plt.get_cmap('tab10')
    color_map = {q: cmap(i / max(1, len(rates)-1)) for i, q in enumerate(rates)}

    # стили для пористостей
    line_styles = ['-', '--', '-.', ':']
    style_map = {pc: line_styles[i % len(line_styles)] for i, pc in enumerate(pcs)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # строим кривые
    for q in rates:
        for pc in pcs:
            grp = df_all[(df_all[FILT] == q) & (df_all[SPAR] == pc)].sort_values(XPAR)
            if grp.empty:
                continue
            ax.plot(grp[XPAR], grp[YPAR], marker='*',
                    color=color_map[q], linestyle=style_map[pc],
                    label=f"Q={q:.1e} $кг/м^3с$, P$_c$={pc} МПа")

    ax.set_xlabel(f"{XPAR}")
    ax.set_ylabel(f"{YPAR}")
    ax.set_xscale('log')

    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # сохраняем или показываем
    if fname:
        # сохраняем график
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)

    else:
        plt.show()