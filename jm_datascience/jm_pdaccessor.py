"""
jm_pdaccessor
Additional methods to pd.Series and pd.Dataframes
- at the end of the class (out of the class) you will find:

    @pd.api.extensions.register_series_accessor("jm")
    class JMSeriesAccessor(JMAccessor):
        pass
        
    @pd.api.extensions.register_dataframe_accessor("jm")
    class JMDataFrameAccessor(JMAccessor):
        pass

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


# jm.convert_dtypes.... DONE
#   - generals or profile almost the same and info_summary??? - FUTURE
#   - infos just print directly w/some resume previous view native info()
#       - infobase() for base of infoplus and info_cmp() - DONE
#       - infoplus for Series and DF - DONE - Future add more info like shape, size, etc...
#       - info_cmp(df) only for DF in base of infobase - FUTURE for series.
#       - infomax(df) OK for DF - FUTURE for series (infomax vs describe!?)
#       - generals vs profile vs infoplus vs infomax .... FUTURE

## TO-DO: SERIES ?? .infoplus(), and infomax() valid for series too?
# Info() e infomax() pueden hacerse que permitan trasponer filas por columns
# fix .infoplus() and .infomax() categorical (Gender example) n-unique() and value_counts()
# refactor .generals()

## Estoy un poco desordenado en las tareas por hacer en este módulo y en jm_pandas
# el tema es que describeplus() me gustaría escribirlo como método - dual para series y para df - y no como función 
# por otro lado describeplus hace un convert_dtypes() pero no es sificiente, por ahí revisemos los conver_dtypes jm....¡?
# convert_dtypeplus() y convert_dtypemax()
## OJO tengo que hacer convert_dtypes válido para DFs y para Series... !!

## Otra coas, tengo jm_pd.fmt_values_for_pd() y tengo jm_prt.fmt_nums() - dado que el primero se usa solo en describe
# si logro hacer dscribe un método ya no tendré más la function 'jm_pd.fmt_values_for_pd()' 

## Trabajar en describeplus (para Series y dfs)!?

## CHARTs jm_pandas
#   - plt_pie() DONE
#   - plt_parteto



# Standard Libs
from typing import Any, Optional, Union

# Third-party Libs
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
## qwen - claude


## Aux function to set unit of memory data
def to_bytes_multiplier(value, decimals=1):

    if value < 0:
        return '0 bytes'

    units = ['bytes', 'KB', 'GB', 'TB', 'PB']
    unit_ix = 0

    while value >= 1024 and unit_ix < len(units) - 1:
        value /= 1024
        unit_ix += 1

    if unit_ix == 0:
        decimals = 0
    
    return f"{value:.{decimals}f} {units[unit_ix]}"


class JMAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._invalid_object_msg = (
            f"\n [ERROR] - Invalid object: {self._obj} - {type(self._obj)} \n "
        )

    
    ## Aux method to set unit of memory data
    def _to_bytes_mult(value, decimals=1):

        if value < 0:
            return '0 bytes'

        units = ['bytes', 'KB', 'GB', 'TB', 'PB']
        unit_ix = 0

        while value >= 1024 and unit_ix < len(units) - 1:
            value /= 1024
            unit_ix += 1

        if unit_ix == 0:
            decimals = 0
        
        return f"{value:.{decimals}f} {units[unit_ix]}"


    ## """Información extendida del objeto pandas"""
    def generals(self):
        """Información extendida del objeto pandas"""
        # Acá solo llamamos la sub_funcíon según el tipo de objeto pandas
        if isinstance(self._obj, pd.DataFrame):
            return self._df_generals()
        elif isinstance(self._obj, pd.Series):
            return self._sr_generals()
        else:
            raise ValueError(self._invalid_object_msg)
    
    def _df_generals(self):
        """Información específica para DataFrame"""
        return {
            'type': 'DataFrame',
            'shape': self._obj.shape,
            'memory_usage': self._obj.memory_usage(deep=True).sum(),
            'memory_usage_mb': self._obj.memory_usage(deep=True).sum() / 1024**2,
            'null_count': self._obj.isnull().sum().sum(),
            'null_counts': self._obj.isnull().sum().to_dict(),
            'columns': list(self._obj.columns),
            'dtypes': self._obj.dtypes.to_dict()
        }
    
    def _sr_generals(self):
        """Información específica para Series"""
        return {
            'type': 'Series',
            'shape': self._obj.shape,
            'memory_usage': self._obj.memory_usage(deep=True),
            'memory_usage_mb': self._obj.memory_usage(deep=True) / 1024**2,
            'null_count': self._obj.isnull().sum(),
            'dtype': str(self._obj.dtype),
            'name': self._obj.name,
            'unique_values': self._obj.nunique(),
            'value_counts': self._obj.value_counts().head().to_dict()
        }
    

    ## """.jm.infomax() para Series y DFs"""
    def infomax(self):                                      # <- .infomax() Series and DF selector
        """ Datail pandas object information."""
        if isinstance(self._obj, pd.DataFrame):
            return self._df_infomax()
        elif isinstance(self._obj, pd.Series):
            # return self._sr_infomax()
            return self._sr_infomax()
        else:
            raise ValueError(self._invalid_object_msg)
        
        
    def _df_infomax(self):
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
                'Sample-Values': [
                   self._obj[col].unique()[:4].tolist() if not self._obj[col].empty else [] 
                   for col in self._obj.columns
                ]
            }
        return pd.DataFrame(info)
    

    def _sr_infomax(self, 
                sample_size: int = 10,
                memory_unit: str = 'bytes',
                include_samples: bool = True,
                include_distribution: bool = True,
                top_n_values: int = 5) -> dict[str, Any]:
        """
        Proporciona información detallada sobre la Series.
        
        Parameters:
        -----------
        sample_size : int, default 10
            Número de valores únicos de muestra a mostrar
        memory_unit : str, default 'bytes'
            Unidad de memoria ('bytes', 'KB', 'MB', 'GB')
        include_samples : bool, default True
            Si incluir valores de muestra en el resultado
        include_distribution : bool, default True
            Si incluir información de distribución de valores
        top_n_values : int, default 5
            Número de valores más frecuentes a mostrar
            
        Returns:
        --------
        dict[str, Any]
            Diccionario con información detallada de la Series
        """
        if self._obj.empty:
            return {'error': 'Series is empty'}
        
        # Validar parámetros
        valid_units = {'bytes': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        if memory_unit not in valid_units:
            raise ValueError(f"memory_unit debe ser uno de: {list(valid_units.keys())}")
        
        divisor = valid_units[memory_unit]
        
        # Información básica
        basic_info = self._get_basic_info(divisor, memory_unit)
        
        # Información estadística
        stats_info = self._get_statistical_info()
        
        # Información de valores
        values_info = self._get_values_info(sample_size, include_samples, 
                                          top_n_values, include_distribution)
        
        # Combinar toda la información
        result = {**basic_info, **stats_info, **values_info}
        
        return result
    
    def _get_basic_info(self, memory_divisor: float, memory_unit: str) -> dict[str, Any]:
        """Obtiene información básica de la Series."""
        n_total = len(self._obj)
        null_count = self._obj.isnull().sum()
        non_null_count = n_total - null_count
        
        return {
            'name': self._obj.name,
            'dtype': str(self._obj.dtype),
            'shape': self._obj.shape,
            'size': self._obj.size,
            'n_total': n_total,
            'n_nulls': null_count,
            'n_non_nulls': non_null_count,
            'pct_nulls': round((null_count / n_total) * 100, 2) if n_total > 0 else 0,
            'memory_usage': round(self._obj.memory_usage(deep=True) / memory_divisor, 2),
            'memory_unit': memory_unit
        }
    
    def _get_statistical_info(self) -> dict[str, Any]:
        """Obtiene información estadística de la Series."""
        info = {}
        
        # Información de valores únicos
        info['n_uniques'] = self._obj.nunique(dropna=True)
        info['n_duplicates'] = self._obj.duplicated().sum()
        info['has_duplicates'] = self._obj.duplicated().any()
        info['uniqueness_ratio'] = round(self._obj.nunique() / len(self._obj), 3) if len(self._obj) > 0 else 0
        
        # Min/Max para tipos apropiados
        min_val, max_val = self._get_min_max_values()
        info['min_value'] = min_val
        info['max_value'] = max_val
        
        # Estadísticas numéricas si aplica
        if pd.api.types.is_numeric_dtype(self._obj):
            numeric_stats = self._get_numeric_statistics()
            info.update(numeric_stats)
        
        # Estadísticas de texto si aplica
        if pd.api.types.is_string_dtype(self._obj) or self._obj.dtype == 'object':
            text_stats = self._get_text_statistics()
            info.update(text_stats)
            
        return info
    
    def _get_numeric_statistics(self) -> dict[str, Any]:
        """Obtiene estadísticas específicas para datos numéricos."""
        try:
            numeric_series = pd.to_numeric(self._obj, errors='coerce')
            return {
                'mean': round(numeric_series.mean(), 3),
                'median': round(numeric_series.median(), 3),
                'std': round(numeric_series.std(), 3),
                'variance': round(numeric_series.var(), 3),
                'skewness': round(numeric_series.skew(), 3),
                'kurtosis': round(numeric_series.kurtosis(), 3),
                'q25': round(numeric_series.quantile(0.25), 3),
                'q75': round(numeric_series.quantile(0.75), 3),
                'iqr': round(numeric_series.quantile(0.75) - numeric_series.quantile(0.25), 3)
            }
        except Exception:
            return {}
    
    def _get_text_statistics(self) -> dict[str, Any]:
        """Obtiene estadísticas específicas para datos de texto."""
        try:
            # Filtrar valores no nulos y convertir a string
            text_series = self._obj.dropna().astype(str)
            if text_series.empty:
                return {}
            
            lengths = text_series.str.len()
            return {
                'avg_length': round(lengths.mean(), 1),
                'min_length': lengths.min(),
                'max_length': lengths.max(),
                'empty_strings': (text_series == '').sum(),
                'whitespace_only': text_series.str.isspace().sum()
            }
        except Exception:
            return {}
    
    def _get_values_info(self, sample_size: int, include_samples: bool,
                        top_n_values: int, include_distribution: bool) -> dict[str, Any]:
        """Obtiene información sobre los valores de la Series."""
        info = {}
        
        # Valor más frecuente
        most_frequent, freq_count = self._get_most_frequent()
        info['most_frequent_value'] = most_frequent
        info['most_frequent_count'] = freq_count
        info['most_frequent_pct'] = round((freq_count / len(self._obj)) * 100, 2) if len(self._obj) > 0 else 0
        
        # Valores de muestra
        if include_samples:
            info['sample_values'] = self._get_sample_values(sample_size)
        
        # Distribución de valores más frecuentes
        if include_distribution:
            info['top_values_distribution'] = self._get_top_values(top_n_values)
            
        return info
    
    def _get_min_max_values(self) -> tuple:
        """Obtiene valores mín/máx para tipos de datos apropiados."""
        try:
            if (pd.api.types.is_numeric_dtype(self._obj) or 
                pd.api.types.is_datetime64_any_dtype(self._obj)):
                return self._obj.min(), self._obj.max()
            else:
                return None, None
        except Exception:
            return None, None
    
    def _get_most_frequent(self) -> tuple:
        """Obtiene el valor más frecuente y su conteo."""
        try:
            if self._obj.empty:
                return None, 0
            
            value_counts = self._obj.value_counts(dropna=False)
            if len(value_counts) > 0:
                most_frequent = value_counts.index[0]
                freq_count = value_counts.iloc[0]
                return most_frequent, freq_count
            else:
                return None, 0
        except Exception:
            return None, 0
    
    def _get_sample_values(self, sample_size: int) -> list:
        """Obtiene valores de muestra de la Series."""
        try:
            if self._obj.empty:
                return []
            
            # Obtener valores únicos, excluyendo NaN
            unique_vals = self._obj.dropna().unique()
            
            # Limitar al tamaño de muestra
            sample_vals = unique_vals[:sample_size] if len(unique_vals) > sample_size else unique_vals
            
            # Convertir a lista, manejando tipos especiales
            return [self._format_sample_value(val) for val in sample_vals]
        except Exception:
            return []
    
    def _get_top_values(self, top_n: int) -> dict:
        """Obtiene los valores más frecuentes con sus conteos."""
        try:
            value_counts = self._obj.value_counts(dropna=False).head(top_n)
            total = len(self._obj)
            
            return {
                str(value): {
                    'count': count,
                    'percentage': round((count / total) * 100, 2)
                }
                for value, count in value_counts.items()
            }
        except Exception:
            return {}
    
    def _format_sample_value(self, value: Any) -> Any:
        """Formatea valores de muestra para mejor legibilidad."""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, str) and len(value) > 50:
            return value[:47] + "..."
        else:
            return value

    # def info_summary(self) -> str:
    #     """Proporciona un resumen textual legible de la Series."""
    #     info = self.infomax(include_samples=False, include_distribution=False)
        
    #     summary_parts = [
    #         f"Series: {info.get('name', 'Unnamed')}",
    #         f"Shape: {info['shape']} | Dtype: {info['dtype']}",
    #         f"Non-null: {info['n_non_nulls']} ({100 - info['pct_nulls']:.1f}%)",
    #         f"Unique values: {info['n_uniques']} ({info['uniqueness_ratio']:.1%} uniqueness)",
    #         f"Memory usage: {info['memory_usage']} {info['memory_unit']}",
    #     ]
        
    #     if info.get('mean') is not None:
    #         summary_parts.append(f"Mean: {info['mean']} | Std: {info.get('std', 'N/A')}")
        
    #     if info.get('most_frequent_value') is not None:
    #         most_freq = info['most_frequent_value']
    #         if isinstance(most_freq, str) and len(most_freq) > 20:
    #             most_freq = most_freq[:17] + "..."
    #         summary_parts.append(f"Most frequent: {most_freq} ({info['most_frequent_count']} times)")
        
    #     return " | ".join(summary_parts)

    def profile(self) -> pd.DataFrame:
        """Genera un DataFrame con el perfil completo de la Series para fácil visualización."""
        info = self.infomax()
        
        # Crear lista de tuplas (métrica, valor)
        profile_data = []
        
        # Información básica
        basic_metrics = ['name', 'dtype', 'n_total', 'n_nulls', 'n_non_nulls', 
                        'pct_nulls', 'memory_usage', 'n_uniques', 'uniqueness_ratio']
        
        for metric in basic_metrics:
            if metric in info:
                profile_data.append((metric.replace('_', ' ').title(), info[metric]))
        
        # Estadísticas numéricas si existen
        numeric_metrics = ['mean', 'median', 'std', 'min_value', 'max_value', 'q25', 'q75']
        for metric in numeric_metrics:
            if metric in info:
                profile_data.append((metric.replace('_', ' ').title(), info[metric]))
        
        # Estadísticas de texto si existen
        text_metrics = ['avg_length', 'min_length', 'max_length']
        for metric in text_metrics:
            if metric in info:
                profile_data.append((metric.replace('_', ' ').title(), info[metric]))
        
        return pd.DataFrame(profile_data, columns=['Metric', 'Value'])
        
    

    # "Extended convert_dtypes -convert_dtypesplus() and convert_dtypesmax()- <- _dtypeplus()
    def convert_dtypesplus(self):
        """
        Converts DataFrame columns or Series to appropriate dtypes, handling mixed (num and str) data.

        This method improves upon pandas' `convert_dtypes()` by:
        - Handling object-dtyped columns that contain mostly numeric values with some strings.
        - Performing a second conversion pass to ensure numeric types are optimized (e.g., Int64 instead of float).

        The method first applies `convert_dtypes()`, then checks each column:
        - If it's object-typed, tries to convert to numeric and keeps the result if at least 60% are valid numbers.
        - After each conversion round, re-applies `convert_dtypes()` to optimize types.

        Returns:
            pd.DataFrame or pd.Series: A new pandas_object with improved dtypes based on cleaned and rounded data.

        Notes:
            - Assumes access to `self._obj` as the original DataFrame or Series.
            - Uses `Int64` dtype to preserve NaN support in integer columns.
        """
        # Here we just call the sub_function according to the pandas object type
        if isinstance(self._obj, pd.DataFrame):
            return self._df_convert_dtypesplus()
        elif isinstance(self._obj, pd.Series):
            return self._sr_convert_dtypesplus()
        else:
            raise ValueError(self._invalid_object_msg)
        
        
    def _df_convert_dtypesplus(self):
        df = self._obj.copy()

        def _apply_func(col: pd.Series):
            return col.jm._sr_convert_dtypesplus()

        return df.apply(_apply_func)
    
    
    def _sr_convert_dtypesplus(self):
        sr = self._obj.convert_dtypes()

        if pd.api.types.is_object_dtype(sr):
            converted = pd.to_numeric(sr, errors='coerce')         # 'coerce' replace errors (strings) to NaNs
            numeric_ratio = converted.notna().sum() / len(sr)   
            if numeric_ratio >= 0.6:                               # if cant of numerics values > 60% of total IS NUMERIC
                sr = converted

        sr = sr.convert_dtypes()                                    # To avoid floats when all values are ints
        return sr    
    

    # "Extended convert_dtypes -convert_dtypesplus() and convert_dtypesmax()- <- _dtypemax()
    def convert_dtypesmax(self):
        """
        Converts DataFrame columns or Series to appropriate dtypes, handling mixed or noisy numeric data.

        Summary:
        - convert_dtypesplus + handling noisy numeric data (miconesim values near 0, too small or too big decimal part)
        - Most cases useful use of cnovert_dtypesmax() instead of plus..

        This method improves upon pandas' `convert_dtypes()` by:
        - Handling object-dtyped columns that contain mostly numeric values with some strings.
        - Performing a second conversion pass to ensure numeric types are optimized (e.g., Int64 instead of float).
        - Rounding floating-point values to eliminate insignificant decimal noise.
        - Optionally converting back to integer types when possible after rounding.

        The method first applies `convert_dtypes()`, then checks each column:
        - If it's object-typed, tries to convert to numeric and keeps the result if at least 60% are valid numbers.
        - After each conversion round, re-applies `convert_dtypes()` to optimize types.

        Additionally, for float columns, this method rounds small decimal residues (e.g., due to float imprecision)
        and may convert such columns to integer types if all values are effectively whole numbers.

        Returns:
            pd.DataFrame: A new DataFrame with improved dtypes based on cleaned and rounded data.

        Notes:
            - Designed for internal use (leading underscore).
            - Assumes access to `self._obj` as the original DataFrame.
            - Uses `Int64` dtype to preserve NaN support in integer columns.
        """
        # Here we just call the sub_function according to the pandas object type
        if isinstance(self._obj, pd.DataFrame):
            return self._df_convert_dtypesmax()
        elif isinstance(self._obj, pd.Series):
            return self._sr_convert_dtypesmax()
        else:
            raise ValueError(self._invalid_object_msg)
        

    def _df_convert_dtypesmax(self):
        # ._convert_dtypeplus() + micron noise control
        df = self._obj.copy()

        def _apply_func(col: pd.Series):
            return col.jm._sr_convert_dtypesmax()
                        
        return df.apply(_apply_func)
    

    def _sr_convert_dtypesmax(self):
        # ._convert_dtypeplus() + micron noise control
        sr = self._obj.jm.convert_dtypesplus()

        if pd.api.types.is_float_dtype(sr):
            sr = sr.apply(lambda x: round(x) if x % 1 > 0.999999 else x)
            sr = sr.convert_dtypes()      # Eliminate bad nans
            if sr.apply(lambda x: True if x % 1 < 0.000001 or pd.isna(x) else False).all():
                sr = sr.round(8).astype('Int64')

        sr = sr.convert_dtypes()                                    # To avoid floats in case all ints.
        return sr    





    # def infomax(self):
    #     info = {
    #             'Column': self._obj.columns,
    #             'Dtype': self._obj.dtypes.values,
    #             'N-Nulls': self._obj.isnull().sum().values,
    #             'N-Total': self._obj.count().values,
    #             'N-Uniques': [self._obj[col].nunique(dropna=True) for col in self._obj.columns],
    #             'Pct-Nulls': [
    #                 round((self._obj[col].isnull().sum() / len(self._obj)) * 100, 1) 
    #                 for col in self._obj.columns
    #             ],
    #             'Memory-Usage': [
    #                 self._obj[col].memory_usage(deep=True) 
    #                 for col in self._obj.columns
    #             ],
    #             'Min-Value': [
    #                 self._obj[col].min() if pd.api.types.is_numeric_dtype(self._obj[col]) or 
    #                                     pd.api.types.is_datetime64_any_dtype(self._obj[col]) else None
    #                 for col in self._obj.columns
    #             ],
    #             'Max-Value': [
    #                 self._obj[col].max() if pd.api.types.is_numeric_dtype(self._obj[col]) or 
    #                                     pd.api.types.is_datetime64_any_dtype(self._obj[col]) else None
    #                 for col in self._obj.columns
    #             ],
    #             'Most-Frequent': [
    #                 self._obj[col].mode().iloc[0] if len(self._obj[col].mode()) > 0 else None
    #                 for col in self._obj.columns
    #             ],
    #             'Freq-Count': [
    #                 self._obj[col].value_counts().iloc[0] if len(self._obj[col].value_counts()) > 0 else 0
    #                 for col in self._obj.columns
    #             ],
    #             'Has-Duplicates': [
    #                 self._obj[col].duplicated().any()
    #                 for col in self._obj.columns
    #             ],
    #             'Sample-Values': [
    #                self._obj[col].unique()[:4].tolist() if not self._obj[col].empty else [] 
    #                for col in self._obj.columns
    #             ]
    #         }
    #     return pd.DataFrame(info)

## OJO
# C:\Users\jmonti\Documents\mchi\Dev\git_github_gitlab\PortableGit\__localRepos\jm_pkgs\jm_utils\src\jm_utils\jm_pdaccessor.py:85:
#  FutureWarning: is_categorical_dtype is deprecated and will be removed 
# in a future version. Use isinstance(dtype, CategoricalDtype) instead self._obj[col].nunique() if 
# pd.api.types.is_categorical_dtype(self._obj[col]) or

    ## Extended info() -infoplus() and infomax()- <- infoplus()
    def infoplus(self):
        """

        """
        # Here we just call the sub_function according to the pandas object type
        if isinstance(self._obj, pd.DataFrame):
            return self._df_infoplus()
        elif isinstance(self._obj, pd.Series):
            return self._sr_infoplus()
        else:
            raise ValueError(self._invalid_object_msg)
        
    # _infobase() - Base para infoplus() e info_cmp()
    def _df_infobase(self):
        """Return basic information about DataFrame columns in a structured format.
    
        Returns a DataFrame with column-wise information including data type,
        null/non-null counts, unique values count, and duplicate detection.
        
        Returns:
            pd.DataFrame. A DataFrame with columns:
            - Column: column names
            - Dtype: data types
            - N-Nulls: number of null values
            - N-Non-Nulls: number of non-null values
            - N-Uniques: number of unique values (excluding NaN)
            - Has-Duplicates: whether column has duplicate values
        """
        info = {
            'Column': self._obj.columns,
            'Dtype': self._obj.dtypes.values,
            'N-Nulls': self._obj.isnull().sum().values,
            'N-Non-Nulls': self._obj.count().values,
            'N-Uniques': [self._obj[col].nunique(dropna=True) for col in self._obj.columns],
            'Has-Duplicates': [
                self._obj[col].duplicated().any()
                for col in self._obj.columns
            ],
        }
        return(pd.DataFrame(info))
    

    def _sr_infobase(self):
        """Return basic information about a Series in a structured format.
        
        Returns a DataFrame with a single row containing information about the 
        Series including name, data type, null/non-null counts, unique values 
        count, and duplicate detection.
        
        Returns:
            pd.DataFrame. A DataFrame with one row containing:
            - Name: series name (or 'Series' if unnamed)
            - Dtype: data type
            - N-Nulls: number of null values
            - N-Non-Nulls: number of non-null values
            - N-Uniques: number of unique values (excluding NaN)
            - Has-Duplicates: whether series has duplicate values
        """
        info = {
            'Name': [self._obj.name if self._obj.name is not None else 'Series'],
            'Dtype': [self._obj.dtype],
            'N-Nulls': [self._obj.isnull().sum()],
            'N-Non-Nulls': [self._obj.count()],
            'N-Uniques': [self._obj.nunique(dropna=True)],
            'Has-Duplicates': [self._obj.duplicated().any()],
        }
        return pd.DataFrame(info)
    
    
    def _df_infoplus(self):
        ''' infobase() plus similar to .info() presentation data'''
        df = self._df_infobase()

        print(f"{type(self._obj)} | {self._obj.index}")
        print(f"Data columns (total {len(self._obj.columns)} columns):")
        print('-' * 32)
        print(df)
        print(f"memory usage: {to_bytes_multiplier(self._obj.memory_usage(deep=True).sum())}")
        

    def _sr_infoplus(self):
        ''' infobase() plus similar to .info() presentation data'''
        sr = self._sr_infobase()

        print(f"{type(self._obj)} | {self._obj.index}")
        print(sr)
        print(f"memory usage: {to_bytes_multiplier(self._obj.memory_usage(deep=True))}")
    

    ## info_cmp(obj1, obj2) - Show data of columns of two pandas objects
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

        # Get dfs.infobase()
        df1_info = self._df_infobase()
        df2_info = df2.jm._df_infobase()

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


@pd.api.extensions.register_series_accessor("jm")
class JMSeriesAccessor(JMAccessor):
    pass
        
@pd.api.extensions.register_dataframe_accessor("jm")
class JMDataFrameAccessor(JMAccessor):
    pass


    
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
