# show_palettes

# Local Libs
from jm_datascience import jm_pandas as jm_pd

palette_group = input("Ingrese el grupo de paletas a mostrar ['Qualitatives', 'Sequential', 'Diverging', 'Cyclic', 'Sample']: ")
n_items = input("Ingrese el número de colores que quiere mostrar [> 1 and < 26]: ")

palette_group = 'Sequential' if palette_group == '' else palette_group

try:
    n_items = int(n_items)
except ValueError:
    n_items = 14

fig = jm_pd.show_plt_palettes(palette_group, n_items)
# input("Presione Enter para continuar...")

fig_2 = jm_pd.show_sns_palettes(palette_group, n_items)
