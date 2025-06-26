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
import matplotlib.pyplot as plt
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


def describeplus(data, decimals: int = 2) -> pd.DataFrame:
    ''' Descriptive sats of data'''

    serie = to_serie(data)          # Convert data to a pandas Series
    
    # Calc valid values for numerical and categorical series
    non_null_count = serie.count()
    null_count = serie.isnull().sum()
    num_uniques = serie.nunique()

    if len(serie) == non_null_count + null_count:
        total_count = len(serie)
    else:
        total_count = -1            # Error !?
    
    # Calc valid mode for any dtype
    modes = serie.mode()

    if len(modes) == 0:
        mode_str = "No mode"
    elif len(modes) == 1:
        mode_str = str(modes.iloc[0])
    else:
        mode_str = ", ".join(str(val) for val in modes)

    # Calc valid freq. (mode freq.) for any dtype
    if mode_str != "No mode":
        mode_freq = serie.value_counts().iloc[0] 
    else:
        mode_freq = mode_str

    # Avoid Object dtypes to calc stats
    serie = serie.convert_dtypes()

    # Calc. stats for numeric series
    try:
        stats = {
            'Non-null Count': non_null_count,
            'Null Count': null_count,
            'Total Count': total_count,
            'Unique Count': num_uniques,
            'Mean': serie.mean(),
            'Median (50%)': serie.median(),
            'Mode(s)': mode_str,
            'Mode_freq': mode_freq,
            'Skewness': serie.skew(),
            'Variance': serie.var(),
            'Standard Deviation': serie.std(),
            'Kurtosis': serie.kurt(),
            'Minimum': serie.min(),
            'Maximum': serie.max(),
            'Range': serie.max() - serie.min(),
            '25th Percentile': serie.quantile(0.25),
            '50th Percentile': serie.quantile(0.50),
            '75th Percentile': serie.quantile(0.75)
        }
    except:                                 # If the series is not numeric, or ? we will catch the exception and set categorical flag
        stats = {
            'Non-null Count': non_null_count,
            'Null Count': null_count,
            'Total Count': total_count,
            'Unique Count': num_uniques,
            'Top (mode)': mode_str,
            'Freq. mode': mode_freq
        }
    
    df = pd.DataFrame.from_dict(stats, orient='index', columns=[serie.name])
    
    if pd.api.types.is_numeric_dtype(serie):
        df['formatted'] = df[serie.name].apply(
            lambda x: format_value(x, width=8, decimals=decimals))                      # Apply formatting to the stats values
    
    return df
    

def clean_df(df):
    ''' Delete duplicates and nulls'''
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.dropna(how='all', axis=1)
    return df_clean


## CHARTs Functions

def plt_piechart(data, title='Pie Chart', figsize=(6, 6), autopct='%.2f%%', palette='viridis', rotate=45):
    """
    Create a pie chart from a specified column in a DataFrame.
    
    Parameters:
    - data: df[col], pd.Series, list, tuple, or array-like object to create the pie chart from.
    - title (str): Title of the pie chart.
    - figsize (float, float): Size of the figure.
    - autopct (str_format): Format for displaying percentages.
    """
    serie = to_serie(data)
    if len(serie) > 6:
        raise ValueError("Too many values. Max values count: six (6)")

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(palette, len(data))            # Use 'viridis' colormap
    colors_virdis = [cmap(i) for i in range(len(data))]     # Get colors from the colormap

    ax.pie(x=data,
           colors=colors_virdis, 
           autopct=autopct,
           textprops={'size': 'x-large',
                      'color': 'w',
                      'rotation': rotate,
                      'weight': 'bold'})

    ax.set_title(title, fontdict={'size': 15, 'weight': 'bold'})
    
    ax.legend(data.index,
              loc='upper right',
              bbox_to_anchor=(1, 0, 0.2, 1),
              prop={'size': 'small'})

    return fig, ax




if __name__ == "__main__":

    df = pd.DataFrame({'A': [1, 2, pd.NA, pd.NA, 1],
                       'B': [4.0, pd.NA, pd.NA, 6.1, 4.0],
                       'C': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                       'D': ['x', 'y', pd.NA, 'z', 'x'],
                       'E': ['x', 'y', pd.NA, 'z', 'x']})
    
    print(df)

    df2 = clean_df(df)
    
    print(df2)
        


