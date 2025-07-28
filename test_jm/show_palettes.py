# show_palettes

# Local Libs
from jm_datascience import jm_pandas as jm_pd

palette_group = input("Ingrese el grupo de paletas a mostrar ['Qualitatives', 'Sequential', 'Diverging', 'Cyclic', 'Mix']: ")
n_items = int(input("Ingrese el número de colores que quiere mostrar [>1 < 25]: "))

fig = jm_pd.show_palettes(palette_group, n_items)
# fig.show()
# input("Presione Enter para continuar...")
