# show_palettes

# Local Libs
from jm_datascience import jm_pandas as jm_pd

palette_group = input("Ingrese el grupo de paletas a mostrar ['Quali', 'Sequen', 'Diverg', 'Cyclic', 'Mix']: ")
n_items = int(input("Ingrese el número de colores que quiere mostrar [>1 < 25]: "))

jm_pd.show_palettes(palette_group, n_items)