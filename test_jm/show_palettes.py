# show_palettes

# Local Libs
from jm_datascience import jm_pandas as jm_pd

palette_group = input("Ingrese el grupo de paletas a mostrar ['Qualitatives', 'Sequential', 'Diverging', 'Cyclic', 'Sample']: ")
n_items = input("Ingrese el número de colores que quiere mostrar [> 1 and < 26]: ")
try:
    n_items = int(n_items)
except ValueError:
    n_items = None

fig = jm_pd.show_plt_palettes(palette_group, n_items)
# fig.show()
# input("Presione Enter para continuar...")
