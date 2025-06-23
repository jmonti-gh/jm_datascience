"""
jm_pdaccessor
Additional methods to pandas df and functions

Methods:
    .infojm():
    .filter_rows(): según el valor de c/campo (col1=X1, ..., coln=Xn)
    .rename_columns(): mapping(dicts), prefix=, suffix=
"""

__version__ = "0.1.0"
__author__ = "Jorge Monti"
__description__ = "Utilities I use frequently - Several modules"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"


## TO-DO: .infoplus(), and infomax() valid for series too?
# fix .infoplus() and .infomax() categorical (Gender example) n-unique() and value_counts()
# refactor .generals()


import pandas as pd
## qwen

@pd.api.extensions.register_dataframe_accessor("jm")
class JMAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def generals(self):
        """Información extendida del DataFrame"""
        return {
            'shape': self._obj.shape,
            'memory_usage': self._obj.memory_usage(deep=True).sum(),
            'memory_usage_mb': self._obj.memory_usage(deep=True).sum() / 1024**2,
            'null_count': self._obj.isnull().sum().sum(),
            'null_counts': self._obj.isnull().sum().to_dict(),
            'columns': list(self._obj.columns),
            'dtypes': self._obj.dtypes.to_dict()
        }
    
    
    def infomax(self):
        info = {
                'Column': self._obj.columns,
                'Dtype': self._obj.dtypes.values,
                'N-Nulls': self._obj.isnull().sum().values,
                'N-Total': self._obj.count().values,
                'N-Uniques': [self._obj[col].nunique(dropna=True) for col in self._obj.columns],
                'Pct-Nulls': [
                    round((self._obj[col].isnull().sum() / len(self._obj)) * 100, 1) 
                    for col in self._obj.columns
                ],
                'Memory-Usage': [
                    self._obj[col].memory_usage(deep=True) 
                    for col in self._obj.columns
                ],
                'Min-Value': [
                    self._obj[col].min() if pd.api.types.is_numeric_dtype(self._obj[col]) or 
                                        pd.api.types.is_datetime64_any_dtype(self._obj[col]) else None
                    for col in self._obj.columns
                ],
                'Max-Value': [
                    self._obj[col].max() if pd.api.types.is_numeric_dtype(self._obj[col]) or 
                                        pd.api.types.is_datetime64_any_dtype(self._obj[col]) else None
                    for col in self._obj.columns
                ],
                'Most-Frequent': [
                    self._obj[col].mode().iloc[0] if len(self._obj[col].mode()) > 0 else None
                    for col in self._obj.columns
                ],
                'Freq-Count': [
                    self._obj[col].value_counts().iloc[0] if len(self._obj[col].value_counts()) > 0 else 0
                    for col in self._obj.columns
                ],
                'Has-Duplicates': [
                    self._obj[col].duplicated().any()
                    for col in self._obj.columns
                ],
                # 'Sample-Values': [
                #     list(self._obj[col].dropna().head(3).values) if not self._obj[col].empty else []
                #     for col in self._obj.columns
                # ]
                'Sample-Values': [
                   self._obj[col].unique()[:4].tolist() if not self._obj[col].empty else [] 
                   for col in self._obj.columns
                ]
            }
        return pd.DataFrame(info)

## OJO
# C:\Users\jmonti\Documents\mchi\Dev\git_github_gitlab\PortableGit\__localRepos\jm_pkgs\jm_utils\src\jm_utils\jm_pdaccessor.py:85:
#  FutureWarning: is_categorical_dtype is deprecated and will be removed 
# in a future version. Use isinstance(dtype, CategoricalDtype) instead self._obj[col].nunique() if 
# pd.api.types.is_categorical_dtype(self._obj[col]) or

    def infoplus(self):
        info = {
                'Column': self._obj.columns,
                'Dtype': self._obj.dtypes.values,
                'N-Nulls': self._obj.isnull().sum().values,
                'N-Total': self._obj.count().values,
                'N-Uniques': [self._obj[col].nunique(dropna=True) for col in self._obj.columns],
                'Has-Duplicates': [
                    self._obj[col].duplicated().any()
                    for col in self._obj.columns
                ],
            }
        return pd.DataFrame(info)

        # self._obj_info = pd.DataFrame(info)
        # self._obj_info.index = pd.RangeIndex(start=0, stop=len(self._obj_info), step=1)
        # return self._obj_info
    

    def info_cmp(self, df2, format='alt'):
        '''
        Compara la información básica (similar a df.info()) de dos DataFrames.

        Args:
            df2 (pd.DataFrame): Segundo DataFrame a comparar
            format (str): Formato de salida para las columnas
                - 'grouped': Todas las columnas de df1, luego todas las de df2
                - 'alternated': Columnas del mismo tipo alternadas entre df1 y df2

        Returns:
            pd.DataFrame: Tabla comparativa con tipos de datos y cantidad de nulos/no nulos.
        '''

        df1_nm = 'df1'
        df2_nm = 'df2'

        # Get dfs.infoplus()
        df1_info = self.infoplus()
        df2_info = df2.jm.infoplus()

        # Renombrar columnas para diferenciar
        df1_info = df1_info.rename(columns={col: f'{col}_{df1_nm}' for col in df1_info.columns if col != 'Column'})
        df2_info = df2_info.rename(columns={col: f'{col}_{df2_nm}' for col in df2_info.columns if col != 'Column'})

        # Unir por columna
        cmp = pd.merge(df1_info, df2_info, on='Column', how='outer')

        if format == 'group':
            # Formato agrupado: todas las columnas de df1, luego todas las de df2
            df1_cols = [col for col in cmp.columns if col.endswith(f'_{df1_nm}')]
            df2_cols = [col for col in cmp.columns if col.endswith(f'_{df2_nm}')]
            column_order = ['Column'] + df1_cols + df2_cols
            
        elif format == 'alt':
            # Formato alternado: columnas del mismo tipo intercaladas
            base_cols = ['Dtype', 'N-Nulls', 'N-Total', 'N-Uniques', 'Pct-Nulls', 'Memory-Usage', 
                        'Min-Value', 'Max-Value', 'Most-Frequent', 'Freq-Count', 'Has-Duplicates', 'Sample-Values']
            column_order = ['Column']
            
            for base_col in base_cols:
                df1_col = f'{base_col}_{df1_nm}'
                df2_col = f'{base_col}_{df2_nm}'
                if df1_col in cmp.columns:
                    column_order.append(df1_col)
                if df2_col in cmp.columns:
                    column_order.append(df2_col)
        
        else:
            raise ValueError("format debe ser 'grouped' o 'alternated'")

        # Reordenar columnas según el formato seleccionado
        cmp = cmp[column_order]

        # Convertir columnas numéricas a enteros (manejando NaN)
        numeric_cols = [col for col in cmp.columns if col.startswith(('N-Nulls', 'N-Total', 'N-Uniques', 'Freq-Count'))]
        for col in numeric_cols:
            cmp[col] = cmp[col].astype('Int64')  # Usa 'Int64' para manejar NaN

        return cmp
    

    def filter_rows(self, **kwargs):
        '''
        Filtra filas según condiciones de igualdad o rango.
        Ejemplos:
            .filter_rows(A=5)
            .filter_rows(B=(2, 8))  # B entre 2 y 8
            .filter_rows(C='texto')
        Args:
            kwargs: Pares clave-valor donde el valor puede ser:
                    - Un valor simple (igualdad)
                    - Una tupla (min, max) (rango)
        Returns:
            pd.DataFrame: DataFrame filtrado.
        '''
        df = self._obj.copy()
        for col, value in kwargs.items():
            if isinstance(value, tuple):
                min_val, max_val = value
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
            else:
                df = df[df[col] == value]
        return df
    

    def rename_columns(self, mapping=None, prefix="", suffix=""):
        '''
        Renombra columnas con mapeo explícito o añadiendo prefijo/sufijo.
        Args:
            mapping (dict): Mapeo {viejo_nombre: nuevo_nombre}
            prefix (str): Cadena a añadir al inicio de todas las columnas
            suffix (str): Cadena a añadir al final
        Returns:
            pd.DataFrame: DataFrame con columnas renombradas.
        '''
        df = self._obj.copy()
        if mapping:
            df.rename(columns=mapping, inplace=True)
        if prefix or suffix:
            df.columns = [f"{prefix}{col}{suffix}" for col in df.columns]
        return df


    
if __name__ == "__main__":
    from time import sleep
    ## Cómo PORBAR esto !? en test!

    # df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.5, 6.1], 'C': ['x', 'y', 'z']})
    # df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['x', 'y', 'z']})
    # print(df1.jm.filter_rows(A=2, B=(4, 6), C='y'))

    df = pd.DataFrame({
        'Id': [1, 2, 3, 4, 5],
        'Gender': ['F', None, 'M', 'F', 'M'],
        'Age': [26, 32, 24, 27, 40],
        'D': pd.Categorical(['cat1', 'cat2', 'cat1', 'cat2', 'cat1'])
    })

    # print(df.jm.infoplus())
    print(df.jm.infoplus())

    # print(df)
    # print()
    # sleep(1)
    # df.jm.multiplicar_numerico(3)
    # print(df)
    # sleep(2)
    # print()
    # print(df.jm.infojm())
    # print()

    input("Press Enter to exit...")



# def df_info_comp(df1, df2, nombre_df1='DataFrame 1', nombre_df2='DataFrame 2'):
#     """
#     Compara la información básica (similar a df.info()) de dos DataFrames.
    
#     Args:
#         df1 (pd.DataFrame): Primer DataFrame.
#         df2 (pd.DataFrame): Segundo DataFrame.
#         nombre_df1 (str): Nombre del primer DataFrame para mostrar.
#         nombre_df2 (str): Nombre del segundo DataFrame para mostrar.
        
#     Returns:
#         pd.DataFrame: Tabla comparativa con tipos de datos y cantidad de nulos/no nulos.
#     """
#     # Extraer info del primer dataframe
#     info1 = {
#         'columna': df1.columns,
#         'tipo_dato': df1.dtypes.values,
#         'no_nulos': df1.notnull().sum().values,
#         'nulos': df1.isnull().sum().values
#     }
    
#     # Extraer info del segundo dataframe
#     info2 = {
#         'columna': df2.columns,
#         'tipo_dato': df2.dtypes.values,
#         'no_nulos': df2.notnull().sum().values,
#         'nulos': df2.isnull().sum().values
#     }

#     # Crear dataframes con la info
#     df_info1 = pd.DataFrame(info1)
#     df_info2 = pd.DataFrame(info2)

#     # Renombrar columnas para diferenciar
#     df_info1.rename(columns={col: f'{col}_{nombre_df1}' for col in df_info1.columns if col != 'columna'}, inplace=True)
#     df_info2.rename(columns={col: f'{col}_{nombre_df2}' for col in df_info2.columns if col != 'columna'}, inplace=True)

#     # Unir por columna
#     comp = pd.merge(df_info1, df_info2, on='columna', how='outer')

#     return comp
