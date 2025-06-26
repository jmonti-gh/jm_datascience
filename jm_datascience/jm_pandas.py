"""
jm_pandas
"""

__version__ = "0.1.0"
__description__ = "Custom pandas functions for data cleaning and manipulation."
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-30"


import numpy as np
import pandas as pd
## Claude



def format_value(value, width=8, decimals=2, miles=','):
    ''' Format numeric and string values ​​with right-aligned padding.
    Args:
        value: The value to format (can be numeric or string).
        width: Total width of the formatted string.
        decimals: Number of decimal places for numeric values.
        miles: Character to use for thousands separator (',', '_', or None).
    '''
    if not isinstance(width, int) or width <= 0:
        raise ValueError("Width must be a positive integer.")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("Decimals must be a non-negative integer.")
    
    if miles not in [',', '_', None]:
        raise ValueError("Miles must be either ',', '_', or None.")

    try:
        num = float(value)                                  # Convert to float if possible
        if miles:
            return f"{num:>{width}{miles}.{decimals}f}"     # Ancho fijo, x decimales, alineado a la derecha
        else:
            return f"{num:>{width}.{decimals}f}"
        
    except (ValueError, TypeError):
        return str(value).rjust(width)                      # Alinea también strings, para mantener la grilla
    

def to_serie(serie):
    ''' Convert df-column, or numpy-array, or list to pd.Serie '''
    try:
        if isinstance(serie, pd.DataFrame) and len(serie.columns) == 1:     # If serie is a DataFrame with one column
            pdserie = serie.iloc[:, 0]                                      # Convert to pd.Series
        elif isinstance(serie, pd.Series):                                  # If serie is already a Series
            pdserie = serie                                                 # No conversion needed      
        elif isinstance(serie, np.ndarray):                                 # If serie is a NumPy array   
            pdserie = pd.Series(serie.flatten())
        elif isinstance(serie, list):                                       # If serie is a list
            pdserie = pd.Series(serie)
        elif not isinstance(serie, pd.Series):                              # If serie is not a Series
            raise ValueError("Input must be a pandas Series, DataFrame with one column, NumPy array, or list.")
    except Exception as e:
        raise ValueError("Input must be a pandas Series, DataFrame with one column, NumPy array, or list.") from e
    
    return pdserie
    

def clean_df(df):
    ''' Delete duplicates and nulls'''
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.dropna(how='all', axis=1)
    return df_clean


if __name__ == "__main__":

    df = pd.DataFrame({'A': [1, 2, pd.NA, pd.NA, 1],
                       'B': [4.0, pd.NA, pd.NA, 6.1, 4.0],
                       'C': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                       'D': ['x', 'y', pd.NA, 'z', 'x'],
                       'E': ['x', 'y', pd.NA, 'z', 'x']})
    
    print(df)

    df2 = clean_df(df)
    
    print(df2)
        


