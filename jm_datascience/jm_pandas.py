"""
jm_pandas
"""

## TO-DO

## fdt!
# Considerar la opción de adicionar los nulls-nans-pd.NAs opcionalmente a la fdt
# EN REALIDAD el tema de los nans lo tengo que ver en el to_series_with_count()

## pie - paretto - en lo posible mismos parámetros

## OJO con la doble función de formateo de datos que tengo... OJO
# porque debería ajustar tanto esta que tengo acá com la de jm_rchprt o DEJAR SOLO UNA!!!???
# NO SE si conviene hacer dos porque en el caso de series tengo que considerar que NO es bueno mezclar n decimal con 0 decimals en una MISMA Series
# caso del cumulative relative frequency.

__version__ = "0.1.0"
__description__ = "Custom pandas functions for data cleaning and manipulation."
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-30"


## Standard Libs
from typing import Union, Optional, Any, Literal, Sequence, TypeAlias
import random

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # for pareto chart and ?
import seaborn as sns


## Custom types for non-included typing annotations
IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]
# IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]


# An auxiliar function to change num format - OJO se puede hacer más amplia como jm_utils.jm_rchprt.fmt...
def _fmt_value_for_pd(value, width=8, n_decimals=3, thousands_sep=',') -> str:
    """
    Format a value (numeric or string) into a right-aligned string of fixed width.

    Converts numeric values to formatted strings with thousands separators and
    specified decimal places. Strings are padded to the same width for consistent alignment.

    Parameters:
        value (int, float, str): The value to be formatted.
        width (int): Total width of the output string. Must be a positive integer.
        decimals (int): Number of decimal places for numeric values. Must be >= 0.
        miles (str or None): Thousands separator. Valid options: ',', '_', or None.

    Returns:
        str: The formatted string with right alignment.

    Raises:
        ValueError: If width <= 0, decimals < 0, or miles is invalid.

    Examples:
        >>> format_value(123456.789)
        '123,456.79'
        >>> format_value("text", width=10)
        '      text'
        >>> format_value(9876, miles=None)
        '    9876.00'
    """
    # Parameter Value validation <- vamos a tener que analizar este tema por si es un list , etc,,
    #   - En realidad acá tenemos que evaluar algo similar a jm_utils - fmt_values() FUTURE
    # if not isinstance(value, (int, float, np.integer, np.floating)) or pd.api.types.is_any_real_numeric_dtype(value)

    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"Width must be a positive integer. Not '{width}'")
    
    if not isinstance(n_decimals, int) or n_decimals < 0:
        raise ValueError(f"Decimals must be a non-negative integer. Not '{n_decimals}")
    
    if thousands_sep not in [',', '_', None]:
        raise ValueError(f"Miles must be either ',', '_', or None. Not '{thousands_sep}")
    
    try:
        num = float(value)                                          # Convert to float if possible
        if num % 1 == 0:                                            # it its a total integer number
            decimals = 0
        if thousands_sep:
            return f"{num:>{width}{thousands_sep}.{n_decimals}f}"   # Fixed width, 'x' decimal places, right aligned
        else:
            return f"{num:>{width}.{n_decimals}f}"
        
    except (ValueError, TypeError):
        return str(value).rjust(width)                              # Also align strings, to maintain the grid


def _validate_numeric_series(
        pd_data: Union[pd.Series, pd.DataFrame],
        positive: Optional[bool] = True
) -> Union[None, Exception]:

    # Validate data parameter a pandas object
    if not isinstance(pd_data, (pd.Series, pd.DataFrame)):     # pd.Series or pd.Datafram
        raise TypeError(
            f"Input data must be a pandas Series or DataFrame. Got {type(pd_data)} instead."
        )
              
    if positive:
        if not all(                                             # Only positve numeric values
            isinstance(val, (int, float, np.integer, np.floating)) and val > 0 for val in pd_data.values
        ):
            raise ValueError(f"All values in the data must be positive numeric..")
        pass
    else:                                                       # Just only numeric values
        if not all(isinstance(val, (int, float, np.integer, np.floating)) for val in pd_data.values):
            raise ValueError(f"All values in the data must be numeric values.")
        pass


def to_series(
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    index: Optional[Union[pd.Index, Sequence[IndexElement]]] = None,
    name: Optional[str] = None
) -> pd.Series:
    """
    Converts input data into a pandas Series with optional custom index and name.

    This function standardizes various data types into a pandas Series. It supports
    arrays, dictionaries, lists, sets, DataFrames, and existing Series. Optionally,
    a custom index or series name can be assigned.

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]):
            Input data to convert. Supported types:
            - pd.Series: returned as-is (can be overridden with new index/name).
            - np.ndarray: flattened and converted to a Series.
            - dict: keys become the index, values become the data.
            - list or set: converted to a Series with default integer index.
            - pd.DataFrame:
                - 1 column: converted directly to a Series.
                - 2 columns: first column becomes the index, second becomes the values.
        index (Union[pd.Index, Sequence], optional): Custom index to assign to the Series.
            If provided, overrides the original index. Default is None.
        name (str, optional): Name to assign to the Series. Default is None.

    Returns:
        pd.Series: A pandas Series constructed from the input data, with optional
            custom index and name.

    Raises:
        TypeError: If the input data type is not supported.
        ValueError: If the DataFrame has more than 2 columns.

    Examples:
        >>> import pandas as pd
        >>> to_series([1, 2, 3, 4])
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> to_series({'A': 10, 'B': 20, 'C': 30})
        A    10
        B    20
        C    30
        dtype: int64

        >>> df = pd.DataFrame({'Label': ['X', 'Y'], 'Value': [100, 200]})
        >>> to_series(df)
        Label
        X    100
        Y    200
        Name: Value, dtype: int64

        >>> to_series([10, 20, 30], index=['a', 'b', 'c'], name='Measurements')
        a    10
        b    20
        c    30
        Name: Measurements, dtype: int64
    """
    
    # Validate parameters - FUTURE
    
    if isinstance(data, pd.Series):                 # If series is already a Series no conversion needed
        series = data                                  
    elif isinstance(data, np.ndarray):              # If data is a NumPy array   
        series = pd.Series(data.flatten())
    elif isinstance(data, (dict, list)):
        series = pd.Series(data)
    elif isinstance(data, (set)):
        series = pd.Series(list(data))
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:                      # Also len(data.columns == 1)
            series = data.iloc[:, 0]
        elif data.shape[1] == 2:                    # Index: first col, Data: 2nd Col
            series = data.set_index(data.columns[0])[data.columns[1]]
        else:
            raise ValueError("DataFrame must have 1 oer 2 columns. Categories and values for 2 columns cases.")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. "
                    "Supported types: pd.Series, np.ndarray, pd.DataFrame, dict, list, set, and pd.DataFrame")

    if name:
        series.name = name

    if index:
        series.index = index

    return series

                      
def get_fdt(
        data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
        value_counts: Optional[bool] = False,
        dropna: Optional[bool] = True,
        na_position: Optional[str] = 'last',
        include_pcts: Optional[bool] = True,
        include_flat_relatives: Optional[bool] = True,
        fmt_values: Optional[bool] = False,
        order: Optional[str] = 'desc',
        na_aside_calc: Optional[bool] = True,
        index_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a Frequency Distribution Table (FDT) with absolute, relative, and cumulative frequencies.

    This function converts various input data types into a structured DataFrame containing:
    - Absolute frequencies
    - Cumulative frequencies
    - Relative frequencies (proportions and percentages)
    - Cumulative relative frequencies (percentages)

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, pd.DataFrame]): Input data.
            If DataFrame, it will be converted to a Series using `to_series`.
        value_counts (bool, optional): Whether to count occurrences if input is raw data.
            Assumes data is not pre-counted. Default is False.
        dropna (bool, optional): Whether to exclude NaN values when counting frequencies.
            Default is True.
        na_position (str, optional): Position of NaN values in the output:
            - 'first': Place NaN at the top.
            - 'last': Place NaN at the bottom (default).
            - 'value': Keep NaN in its natural order.
            Default is 'last'.
        include_pcts (bool, optional): Whether to include percentage columns.
            If False, only absolute and cumulative frequencies are returned.
            Default is True.
        include_flat_relatives (bool, optional): Whether to return relative and cumulative relative values.
            If False, only frequency and percentage columns are included.
            Default is True.
        fmt_values (bool, optional): Whether to format numeric values using `_fmt_value_for_pd`.
            Useful for improving readability in reports. Default is False.
        order (str, optional): Sort order for the output:
            - 'asc': Sort values ascending.
            - 'desc': Sort values descending (default).
            - 'ix_asc': Sort by index ascending.
            - 'ix_desc': Sort by index descending.
            - None: No sorting.
            Default is 'desc'.
        na_aside_calc (bool, optional): Whether to separate NaN values from calculations but keep them in the output.
            If True, NaNs are added at the end and not included in cumulative or relative calculations.
            Default is True.
        index_name (str, optional): Set an specific index name for the fdt.

    Returns:
        pd.DataFrame: A DataFrame containing the frequency distribution table with the following columns
        (depending on parameters):
            - Frequency
            - Cumulative Frequency
            - Relative Frequency
            - Cumulative Relative Freq.
            - Relative Freq. [%]
            - Cumulative Freq. [%]

    Raises:
        ValueError: If `sort` or `na_position` receive invalid values.

    Notes:
        - This function uses `to_series` to convert input data into a pandas Series.
        - If `na_aside=True` and NaNs are present, they are placed separately and not included in relative calculations.
        - Useful for exploratory data analysis and generating clean statistical summaries.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'B', None])
        >>> fdt = get_fdt(data, sort='desc', fmt_values=True)
        >>> print(fdt)
              Frequency  Cumulative Frequency  Relative Freq. [%]  Cumulative Freq. [%]
        B           3                   3                42.86                  42.86
        A           2                   5                28.57                  71.43
        C           1                   6                14.29                  85.71
        Nulls       1                   7                14.29                 100.00
    """
    columns = [
        'Frequency',
        'Cumulative Frequency',
        'Relative Frequency',
        'Cumulative Relative Freq.',
        'Relative Freq. [%]',
        'Cumulative Freq. [%]'
    ]
    # def _calculate_fdt_relatives(series):     # Revisar, no me gusta el flujo actual
    
    sr = to_series(data)
    
    if dropna:
        sr = sr.dropna()                        # Drop all nulls values of the Series
        sr = sr.drop(np.nan, errors='ignore')   # For series with NaNs as a category with their count (errors='ignore': does not fail if it does not exist)

    if value_counts:
        sr = sr.value_counts(dropna=dropna, sort=False)

    # Validate that all the values are positive numbers
    _validate_numeric_series(sr)

    # Order de original Series to obtain the fdt in the same order as the original data
    match order:
        case 'asc':
            sr = sr.sort_values()
        case 'desc':
            sr = sr.sort_values(ascending=False)
        case 'ix_asc':
            sr = sr.sort_index()
        case 'ix_desc':
            sr = sr.sort_index(ascending=False)
        case None:
            pass
        case _:
            raise ValueError(f"Valid values for order: 'asc', 'desc', 'ix_asc', 'ix_desc', or None. Got '{order}'")
        
    # Handle NaNs values. Two cases: 1. na_aside: don't use for calcs and at the end; 2. use for calcs and locate according na_position
    #   - Determine the number of nans
    if pd.isna(sr.index).any():
        n_nans = sr[np.nan]
    else:
        n_nans = 0

    #   - Locale NaNs row in the Series 'sr'
    if na_aside_calc:
        sr = sr.drop(np.nan, errors='ignore')                   # Drop NaNs from the Series for calculations
        # Column that will then be concatenated to the end of the DF - Only 'Frequency' column, no calculated columns
        nan_row_df = pd.DataFrame(data = [n_nans], columns=[columns[0]], index=[np.nan])
    else:
        # As we use NaNs for calculations decide where locate these values
        sr_without_nan = sr.drop(np.nan, errors='ignore')       # Aux. sr wo/nans allow us to locate the NaNs
        match na_position:             
            case 'first':
                sr = pd.concat([pd.Series({np.nan: n_nans}), sr_without_nan])
            case 'last':
                sr = pd.concat([sr_without_nan, pd.Series({np.nan: n_nans})])
            case 'value' | None:
                pass                # Locates the Nulls row based on the value or index ordering
            case _:
                raise ValueError(f"Valid values for na_position: 'first', 'last', 'value' or None. Got '{na_position}'")

    # Central rutine: create the fdt, including relative and cumulative columns.
    fdt = pd.DataFrame(sr)
    fdt.columns = [columns[0]]
    fdt[columns[1]] = fdt['Frequency'].cumsum()
    fdt[columns[2]] = fdt['Frequency'] / fdt['Frequency'].sum()
    fdt[columns[3]] = fdt['Relative Frequency'].cumsum()
    fdt[columns[4]] = fdt['Relative Frequency'] * 100
    fdt[columns[5]] = fdt['Cumulative Relative Freq.'] * 100

    if na_aside_calc and not dropna:            # We add nan_columns at the end
        fdt = pd.concat([fdt, nan_row_df])

    # Logic to include: only frequencies, or only flat relatives, or percentage (pcts)
    if not include_pcts and not include_flat_relatives:
        fdt = fdt[[columns[0]]]                             # Only 'Frecquency' (col[0]) - doble[[]] to get a DF
    elif not include_pcts and include_flat_relatives:
        fdt = fdt[columns[0:4]]                             # 'Frequency' + plain_relative cols (col[0,1,2,3])
    elif include_pcts and not include_pcts:
        fdt = fdt[[columns[0], columns[4], columns[5]]]     # 'Frequency' + pcts cols (last two cols)

    if fmt_values:
        fdt = fdt.map(_fmt_value_for_pd)

    # Set the index name
    fdt.index.name = index_name if index_name else sr.index.name
        
    return fdt


def describeplus(data, decimals=2, miles=',') -> pd.DataFrame:
    ''' Descriptive sats of data'''

    serie = to_series(data)          # Convert data to a pandas Series
    
    # Calc valid values for numerical and categorical series
    non_null_count = serie.count()
    null_count = serie.isnull().sum()
    num_uniques = serie.nunique()

    if len(serie) == non_null_count + null_count:
        total_count = len(serie)
    else:
        total_count = '[ERROR ¡?]'           # Error !?
    
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
            lambda x: _fmt_value_for_pd(x, width=8, decimals=decimals, miles=miles))      # Apply formatting to the stats values
    
    return df
    

def clean_df(df):
    ''' Delete duplicates and nulls'''
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.dropna(how='all', axis=1)
    return df_clean


def is_mostly_numeric(series, threshold):
    ''' Checks if at least 'threshold'% of the values ​​can be numeric'''
    converted = pd.to_numeric(series, errors='coerce')
    numeric_ratio = converted.notna().sum() / len(series)
    return numeric_ratio >= threshold


def petty_decimals_and_str(series):
    for ix, value in series.items():
        if isinstance(value, str):
            print(f"String -> {ix = } - {value = }")
        elif isinstance(value, float):
            if value % 1 > 0:
                print(f"float -> {ix = } - {value = }")

#--------------------------------------------------------------------------------------------------------------------------------#
#  CHARTs Functions:
#--------------------------------------------------------------------------------------------------------------------------------#
#   - Aux: get_colorblind_palette_list(), get_colors_list(),  _validate_numeric_series()
# Common parameters for categorical charts:
#   - data: Union[pd.Series, pd.DataFrame], | One or two col DF. Case two cols 1se col is index (categories) and 2nd values
#   - value_counts: Optional[bool] = False, | You can plot native values or aggregated ones by categories
#   - scale: Optional[int] = 1,             | All sizes, widths, etc. are scaled from this number (from 1 to 9)
#   - ...


def get_colorblind_color_list():
    """
    Retorna una lista de colores (hexadecimales) amigables para personas
    con daltonismo, equivalentes a sns.color_palette('colorblind').
    """
    return [
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9',
        '#5D8C3B', '#A93967', '#888888', '#FFC107', '#7C9680',
        '#E377C2', '#BCBD22', '#AEC7E8', '#FFBB78', '#98DF8A',
        '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2', '#DBDB8D',
        '#9EDAE5', '#D68E3A', '#A65898', '#B2707D', '#8E6C87'
    ]


def get_color_list(palette: str, n_items: Optional[int] = 10) -> list[str]:
    """
    | Return a valid matplotlib palette list    | 'colorblind' is a kind of sns.colorblind 
    - Qualitatives (Cat) = ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Dark2', 'Paired', 'Accent', 'colorblind']
    - Sequential (Order) = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    - Diverging (MidPoint) = ['coolwarm', 'bwr', 'seismic', 'PiYG', 'PRGn', 'BrBG', 'RdGy', 'RdBu', 'Spectral', 'RdYlGn', 'PuOr', 'RdYlBu']
    - Cyclic (Repeat)= ['twilight', 'twilight_shifted', 'hsv', 'turbo', 'cubehelix', 'gist_rainbow', 'jet', 'nipy_spectral', 'rainbow_r']
    - Mix = ['rainbow', 'flag', 'prism', 'ocean', 'terrain', 'gnuplot', 'CMRmap', 'hot', 'afmhot', 'gist_heat', 'copper', 'bone', 'pink']
    """
    if palette == 'colorblind':
        color_list = get_colorblind_color_list()
    else:
        cmap = plt.get_cmap(palette, n_items)             # Use palette colormap
        color_list = [cmap(i) for i in range(n_items)]    # Get colors from the colormap

    return color_list


def show_plt_palettes(
        palette_group: Union[str, list[str]] = 'Sample',
        n_items: Optional[int]=14,
) -> plt.Figure:
    """
    Displays a visual comparison of Matplotlib color palettes in a two-column layout.

    This function creates a grid of bar charts, each showing the color progression of a
    specific Matplotlib colormap. It supports built-in palette groups (Qualitative, Sequential,
    Diverging, Cyclic), a default 'Sample' view, and custom lists of palettes.

    Parameters:
        palette_group (Union[str, list[str]]): Specifies which palettes to display:
            - If str: one of 'Qualitative', 'Sequential', 'Diverging', 'Cyclic', or 'Sample'.
              Case-insensitive; will be capitalized.
            - If list: a custom list of colormap names to display.
            - Default is 'Sample', which shows a representative selection from all groups.
        n_items (int, optional): Number of color swatches (bars) to display per palette.
            Must be between 1 and 25 (inclusive). Default is 16.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing all subplots.
            This allows further customization, saving, or inspection after display.

    Raises:
        TypeError: If `palette_group` is not a string or list of strings, or if `n_items`
            is not a number.
        ValueError: If `n_items` is not in the valid range (1–25).

    Notes:
        - Invalid or deprecated colormap names are handled gracefully and labeled accordingly.
        - The layout adapts to the number of palettes, using two columns for better readability.
        - Uses `get_colors_list` internally to extract colors from each colormap.
        - Ideal for exploring and selecting appropriate color schemes for data visualization.

    Example:
        >>> show_plt_palettes('Sequential', n_items=10)
        # Displays 10-color samples for all Sequential palettes.

        >>> show_plt_palettes(['viridis', 'plasma', 'coolwarm', 'rainbow'], n_items=12)
        # Shows a custom comparison of four specific palettes.

        >>> show_plt_palettes()
        # Shows a default sample of 4 palettes from each category.
    """
    
    # Verified n_times parameter
    if not isinstance(n_items, (int, float)):
        raise TypeError(f"'n_items' parameter not valid. Must be an int or float. Got '{type(n_items)}'.")

    if n_items < 1 or n_items > 25:
        raise ValueError(f"'n_items' parameter not valid. Must be > 1 and < 26. Got '{n_items}'.")
    n_items = int(n_items) + 1

    # Palette_group selection + custom palette_group and palette_group parameter validation
    Custom = []                             # Default empty list for custom palettes
    if isinstance(palette_group, str):
        palette_group_key = palette_group.strip().capitalize()
    elif isinstance(palette_group, list):
        palette_group_key = 'Custom'
        Custom = palette_group
    else:
        raise TypeError(f"'palette_group' parameter not valid. Must be a string or a list. Got {type(palette_group)}.")
    
    # 1. Native palette Group lists
    Qualitative = ['Accent', 'colorblind', 'Dark2', 'Dark2_r', 'flag', 'Paired', 'Pastel1', 'Pastel2',
                    'prism', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']

    Sequential = ['autumn', 'binary', 'Blues', 'brg', 'BuPu', 'cividis', 'cool', 'GnBu',
              'Greens', 'Greys', 'Greys_r', 'gnuplot', 'inferno', 'magma', 'ocean', 'Oranges',
                'OrRd', 'plasma', 'PuRd', 'Purples', 'Reds', 'terrain', 'viridis', 'Wistia']

    Diverging = ['BrBG', 'bwr', 'bwr_r', 'coolwarm', 'PiYG', 'PiYG_r', 'PRGn', 'PRGn_r',
                 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic', 'Spectral', 'Spectral_r']

    Cyclic = ['berlin', 'berlin_r', 'cubehelix', 'cubehelix_r', 'flag_r', 'gist_rainbow', 'hsv', 'jet_r',
              'managua', 'nipy_spectral', 'rainbow', 'rainbow_r', 'twilight', 'twilight_shifted', 'turbo', 'vanimo']

    # 2. Get the palette group (and _desc) based on the input string (the one selected by the user)
    palette_group_dic = {
        'Qualitative': (Qualitative, 'for categorical data'),
        'Sequential': (Sequential, 'for data that has an order'),
        'Diverging': (Diverging, 'for data that have a significant midpoint'),
        'Cyclic': (Cyclic, 'for data that repeats, such as angles or phases'),
        'Custom': (Custom, 'user selected palettes'),
    }

    defaults_for_sample = ('Sample', 'a sample of four of each category')
    # Get the list of palettes for the selected group and its description
    selected_palettes, palette_group_desc = palette_group_dic.get(palette_group_key, defaults_for_sample)

    # Adjust the palette_group for the 'Sample' (four from each group)) case. Any value different from the main groups
    if palette_group_key not in ('Qualitative', 'Sequential', 'Diverging', 'Cyclic', 'Custom'):
        palette_group_key = 'Sample'                      
        palette_group = [palette for p_g in [Qualitative, Sequential, Diverging, Cyclic] for palette in random.sample(p_g, k=4)]
    else:                                                   # If not 'Sample' (all others, custom included), use the selected palettes  
        palette_group = selected_palettes                   # Assign the actual list if not 'Sample'

    # Build a Series of n_items elements to show colors
    sr = to_series({str(i): 1 for i in range(1, n_items)})

    # Create a figure with two columns for the palettes - Bar charts showing palette colors
    rows = len(palette_group) // 2 if len(palette_group) % 2 == 0 else (len(palette_group) // 2) + 1
    width = 12
    height = rows / 1.25 if rows > 6 else rows / 1.05 
    
    fig, axs = plt.subplots(rows, 2, figsize=(width, height), tight_layout=True, sharex=True)

    # Set the figure title with the palette group key and description
    fig.suptitle(f"Matplolib {palette_group_key} palettes (cmap): {palette_group_desc}", fontsize=14, fontweight='medium', y=1.001)

    if palette_group_key == 'Sample':
        fig.text(0.15, 0.95, "4 Qualitative (for categorical data), 4 Sequential (for ordered data),"
                             "4 Diverging (significant midpoint), and 4 Cyclic (for repeated data)",
                    fontsize=10, transform=fig.transFigure)

    # Iterate over the axes and palette group to plot each palette                                           
    for ax, pltt in zip(axs.flatten(), palette_group):
        try:
            color_list = get_color_list(pltt, n_items=n_items)
            ax.bar(sr.index, sr, color=color_list, width=1, edgecolor='white', linewidth=0.2)
            ax.set_xlim(-0.5, n_items - 1.5)
            ax.set_ylim(0, 0.1)
            ax.set_title(pltt, loc='left', fontsize=10, fontweight='medium')
        except ValueError:
            err_msg = f"'{pltt}' is not currently a valid Matplotlib palette (cmap)"
            ax.set_title(err_msg, loc='left', fontsize=10, fontweight='medium', color='red')

        ax.set_yticks([])       # Hide y-ticks for cleaner look
        ax.set_xticks([])       # Hide x-ticks

    plt.show()
    return fig                  # Return the current figure for further manipulation if needed


def show_sns_palettes(
    palette_group: Union[str, list[str]] = 'Sample',
    n_items: Optional[int] = 14,
) -> plt.Figure:
    """
    Displays a visual comparison of Seaborn color palettes in a two-column layout.

    This function creates a grid of bar charts, each showing the color progression of a
    specific Seaborn (or Matplotlib-compatible) colormap. It supports built-in palette groups
    (Qualitative, Sequential, Diverging, Cyclic), a default 'Sample' view, and custom lists of palettes.

    Parameters:
        palette_group (Union[str, list[str]]): Specifies which palettes to display:
            - If str: one of 'Qualitative', 'Sequential', 'Diverging', 'Cyclic', or 'Sample'.
              Case-insensitive; will be capitalized.
            - If list: a custom list of colormap names to display.
            - Default is 'Sample', which shows a representative selection from all groups.
        n_items (int, optional): Number of color swatches (bars) to display per palette.
            Must be between 1 and 25 (inclusive). Default is 16.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing all subplots.
            This allows further customization, saving, or inspection after display.

    Raises:
        TypeError: If `palette_group` is not a string or list of strings, or if `n_items`
            is not a number.
        ValueError: If `n_items` is not in the valid range (1–25).

    Notes:
        - Invalid or deprecated colormap names are handled gracefully and labeled accordingly.
        - The layout adapts to the number of palettes, using two columns for better readability.
        - Uses `seaborn.color_palette` internally for color extraction and `matplotlib.axes.Axes.barh`
          for display.
        - Ideal for exploring and selecting appropriate color schemes for data visualization.

    Example:
        >>> show_sns_palettes('Sequential', n_items=10)
        # Displays 10-color samples for all Sequential palettes.

        >>> show_sns_palettes(['viridis', 'plasma', 'coolwarm', 'rainbow'], n_items=12)
        # Shows a custom comparison of four specific palettes.

        >>> show_sns_palettes()
        # Shows a default sample of 4 palettes from each category.
    """

    # Verified n_items parameter
    if not isinstance(n_items, (int, float)):
        raise TypeError(f"'n_items' parameter not valid. Must be an int or float. Got '{type(n_items)}'.")

    if n_items < 1 or n_items > 25:
        raise ValueError(f"'n_items' parameter not valid. Must be > 1 and < 26. Got '{n_items}'.")
    n_colors = int(n_items) # Use n_colors internally for consistency with seaborn

    # Palette_group selection + custom palette_group and palette_group parameter validation
    Custom = []                                     # Default empty list for custom palettes
    if isinstance(palette_group, str):
        palette_group_key = palette_group.strip().capitalize()
    elif isinstance(palette_group, list):
        # Convert all custom palette names to strings just in case
        Custom = [str(p) for p in palette_group]
        palette_group_key = 'Custom'
    else:
        raise TypeError(f"'palette_group' parameter not valid. Must be a string or a list. Got {type(palette_group)}.")

    # 1. Native palette Group lists (Seaborn often uses these names directly)
    Qualitative = ['Accent', 'bright', 'colorblind', 'dark', 'Dark2', 'Dark2_r', 'deep', 'flag',
                   'muted', 'Paired', 'Pastel1', 'Pastel2', 'prism', 'Set1', 'Set2', 'Set3',
                   'tab10', 'tab20', 'tab20b', 'tab20c']        # deep, muted, bright, pastel, dark, colorblind

    Sequential = ['autumn', 'binary', 'Blues', 'brg', 'BuPu', 'cividis', 'cool', 'crest',
                  'flare', 'GnBu', 'Greens', 'Greys', 'Greys_r', 'gnuplot', 'inferno', 'magma',
                  'mako', 'ocean', 'Oranges', 'OrRd', 'plasma', 'PuRd', 'Purples', 'Reds',
                  'rocket', 'terrain', 'viridis', 'Wistia']     # rocket, mako, flare, crest

    Diverging = ['BrBG', 'bwr', 'bwr_r', 'coolwarm', 'icefire', 'PiYG', 'PiYG_r', 'PRGn',
                 'PRGn_r', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic', 'Spectral',
                 'Spectral_r', 'vlag']                          # vlag, icefire

    Cyclic = ['berlin', 'cubehelix', 'cubehelix_r', 'flag_r', 'gist_rainbow', 'hsv', 'jet_r', 'managua',
              'nipy_spectral', 'rainbow', 'rainbow_r', 'twilight', 'twilight_shifted', 'turbo', 'vanimo',
              'twilight_r', 'twilight_shifted_r', 'turbo_r']

    # 2. Get the palette group (and _desc) based on the input string (the one selected by the user)
    palette_group_dic = {
        'Qualitative': (Qualitative, 'for categorical data'),
        'Sequential': (Sequential, 'for data that has an order'),
        'Diverging': (Diverging, 'for data that have a significant midpoint'),
        'Cyclic': (Cyclic, 'for data that repeats, such as angles or phases'),
        'Custom': (Custom, 'user selected palettes'),
    }

    defaults_for_sample = ('Sample', 'a sample of four from each category')
    # Get the list of palettes for the selected group and its description
    selected_palettes, palette_group_desc = palette_group_dic.get(palette_group_key, defaults_for_sample)

    # Adjust the palette_group for the 'Sample' (four from each group) case.
    # Any value different from the main groups
    if palette_group_key not in ('Qualitative', 'Sequential', 'Diverging', 'Cyclic', 'Custom'):
        palette_group_key = 'Sample'
        # Ensure k is not greater than list length for random.sample
        k_val_qual = min(4, len(Qualitative))
        k_val_seq = min(4, len(Sequential))
        k_val_div = min(4, len(Diverging))
        k_val_cyc = min(4, len(Cyclic))
        
        # Use separate random samples to ensure up to 4 distinct palettes from each group
        palette_group = (random.sample(Qualitative, k=k_val_qual) +
                         random.sample(Sequential, k=k_val_seq) +
                         random.sample(Diverging, k=k_val_div) +
                         random.sample(Cyclic, k=k_val_cyc))
    else:
        palette_group = selected_palettes # If not 'Sample' (all others, custom included), use the selected palettes

    # Create a figure with a flexible grid
    num_palettes = len(palette_group)
    cols = 2
    rows = (num_palettes + cols - 1) // cols # Ceiling division

    # Adjust figure size
    fig_width = 12
    # Base height per row for horizontal bars
    fig_height_per_row = 0.5
    # Total height, adding space for titles/supertitles
    fig_height = rows * fig_height_per_row + 1.5

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), layout='tight')
    axs = axs.flatten() # Flatten the array of axes for easy iteration

    # Set the figure title with the palette group key and description
    fig.suptitle(f"Seaborn {palette_group_key} palettes (cmap): {palette_group_desc}",
                 fontsize=14, fontweight='medium', y=1.001) # Adjust y for suptitle

    if palette_group_key == 'Sample':
        fig.text(0.5, 0.94, "4 Qualitative (for categorical data), 4 Sequential (for ordered data), "
                            "4 Diverging (significant midpoint), and 4 Cyclic (for repeated data)",
                 fontsize=10, ha='center', transform=fig.transFigure)

    # Iterate over the axes and palette group to plot each palette
    for i, pltt in enumerate(palette_group):
        ax = axs[i]
        try:
            # Use sns.color_palette to get the colors
            colors = sns.color_palette(pltt, n_colors=n_colors)
            # Manually draw horizontal bars
            for j, color in enumerate(colors):
                ax.barh(0, 1, left=j, color=color, height=1, edgecolor='none')
            ax.set_xlim(0, n_colors)
            ax.set_ylim(-0.5, 0.5)      # Center the bar vertically
            ax.set_title(pltt, loc='left', fontsize=10, fontweight='medium')
        except ValueError:         
            err_msg = f"'{pltt}' is not currently a valid Matplotlib palette (cmap)"
            ax.set_title(err_msg, loc='left', fontsize=10, fontweight='medium', color='red')

        ax.set_yticks([]) # Hide y-ticks
        ax.set_xticks([]) # Hide x-ticks

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.show()
    return fig


def plt_pie(
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    value_counts: Optional[bool] = False,
    dropna: Optional[bool] = True,
    order: Optional[str] = 'desc',
    scale: Optional[int] = 1,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    label_place: Optional[str] = 'ext',
    palette: Optional[list] = 'colorblind',
    startangle: Optional[float] = 90,
    pct_decimals: Optional[int] = 1,
    label_rotate: Optional[float] = 0,
    legend_loc: Optional[str] = 'best',
    show_stats_subtitle = True
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates a pie or donut chart with customizable label placement and styling.

    This function creates a pie or donut chart from categorical data using matplotlib.
    It supports internal, external, or aside label placement with optional percentage
    and value annotations.

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],):
            Input data be converted to a Series using `to_series`.
        value_counts (bool, optional): If True, counts occurrences of each category.
            Default is False.
        sort (bool, optional): If True and `value_counts=True`, sorts categories by frequency.
            Default is True.
        nans (bool, optional): If True, includes NaN values in the count. Default is False.
        scale (int, optional): Chart scaling factor (1 to 9). Affects figure size and font sizes.
            Default is 1.
        figsize (tuple, optional): Width and height of the figure in inches. Overrides `scale`.
            Default is None.
        title (str, optional): Chart title. If not provided, a default title is used.
        kind (str, optional): Type of chart to generate. Options:
            - 'pie': standard pie chart.
            - 'donut': donut chart with a hollow center.
        label_place (str, optional): Placement of labels. Options:
            - 'ext': external labels connected by arrows.
            - 'int': internal labels within each segment (shows absolute values and percentages).
            - 'aside': internal labels and a legend with extended labels on the side.
        palette (list or str, optional): Color palette for segments. If a string, uses a predefined
            palette (e.g., 'set2' or 'viridis'). Default is 'colorblind'.
        startangle (float, optional): Starting angle (in degrees) for the first wedge.
            Default is -40.
        pct_decimals (int, optional): Number of decimal places to display in percentage values.
            Default is 1.
        label_rotate (float, optional): Rotation angle for internal labels (only applies if
            `label_place='int'`). Default is 0.
        legend_loc (str, optional): Position of the legend (if displayed). See valid options in
            `matplotlib.legend`. Default is 'best'.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing:
            - fig: The Matplotlib Figure object.
            - ax: The Matplotlib Axes object for further customization.

    Raises:
        TypeError: If input data is not a pandas Series or DataFrame.
        ValueError: If `kind` is not 'pie' or 'donut'.
        ValueError: If more than 12 categories are provided.
        ValueError: If `scale` is not between 1 and 9.

    Notes:
        - This function uses `to_series` to convert DataFrame or other data types into a Series.
        - It supports rich annotations and color palettes for better visual clarity.
        - Maximum of 12 categories allowed for readability.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'B', 'A', 'A', 'B', 'C'])
        >>> fig, ax = plt_pie3(data, kind='donut', label_place='aside', title='Distribution of Categories')
        >>> plt.show()
    """
    # Get the data to graph: use controls and processing that get_fdt() does to obtain the series to graph (first column: 'Frequency')
    fdt = get_fdt(data, value_counts=value_counts, order=order,
                  dropna=False, na_position='value', na_aside_calc=False, include_flat_relatives=False, include_pcts=False)
        # - dropna=False            -> So that it doesn't remove NaNs, and then handle them
        # - na_aside_calc=False     -> So that it allows me to sort the nan value with na_position='value'
        # - na_position='value'     -> So that it allows me to sort the nan value within the list of values either desc or asc (according to order)
        # - We do not include relative flat rates or percentages. We calculate the percentages before presenting them.

    cat_name = fdt.index.name                       # Category name <- from fdt.index.name (could be 'Index' Warn!)

    sr = fdt.iloc[:, 0]                             # Get the Series with the frequencies (count)
    # As sr.index build the legends: If I want to change the legends, I'll have to see how I modify this sr.index (must be done previously aside)
         
    # Handling of nans since they are presented in the subtitle, whether or not they appear in the graph
    total_label = "Total (w/ nulls)"                # Default total_label to be presented in subtitle
    
    if pd.isna(sr.index).any():                     # There is np.nan [NaN] index, nans values
        n_nans = sr[np.nan]
        if dropna:                                  # No NaNs in the graph
            sr = sr.drop(np.nan, errors='ignore')   # Drop NaN row from the DataFrame
            total_label = "Total (wo/ nulls)"       # The total will be calculated wo/NaNs (likewise, n_nans will appear in the subtitle.)
    else:                                           # No np.nan row
        n_nans = 0

    # Validate kind parameter
    if kind.lower() not in ['pie', 'donut']:
        raise ValueError(f"Invalid 'kind' parameter: '{kind}'. Must be 'pie' or 'donut'.")
    
    # Validate maximum categories
    if len(sr) > 12:
        raise ValueError(f"Data contains {len(sr)} categories. "
                        "Maximum allowed is 12 categories.")
    
    # Build graphs size, and fonts size from scale, and validate scale from 1 to 9.
    if scale < 1 or scale > 9:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '9', not '{scale}'.")
    else:
        scale = round(scale)

    # Calculate figure dimensions
    if figsize is None:
        multiplier = scale + 7.5
        w_base, h_base = 1, 0.56
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
    
    # Calculate font sizes based on figure width
    label_size = width * 1.25
    title_size = width * 1.57

    # Base fig definitions
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

    # Configure wedge properties for donut or pie chart
    wedgeprops = {}
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.54, 'edgecolor': 'white', 'linewidth': 1}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 0.5}

    # Define colors
    color_palette = get_color_list(palette, len(sr))

    if label_place == 'ext':

        wedges, texts = ax.pie(sr, wedgeprops=wedgeprops, colors=color_palette, startangle=startangle)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        # Build the labels. Annotations and legend in same label (External)
        labels = [
            f"{sr.loc[sr == value].index[0]}\n{value}\n({round(value / sr.sum() * 100, pct_decimals)} %)"
            for value in sr.values
        ]
        
        # Draw the annotations (labels)
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, fontsize=label_size, **kw)
            
    elif label_place == 'int' or label_place == 'aside':
        label_size = label_size * 0.8
        legend_size = label_size * 1.1
        
        # Set autopct and legends, different for 'int' and 'aside' label_place
        if label_place == 'int':
            # autopct for internal annotations. A funtion to show both: absolute an pcts.
            format_string = f'%.{pct_decimals}f%%'

            def _make_autopct(values, fmt_str):     # A python Closuer
                value_iterator = iter(values)
                
                def my_autopct(pct):
                    absolute_value = next(value_iterator)
                    percentage_string = fmt_str % pct
                    return f"{absolute_value}\n({percentage_string})"  
                
                return my_autopct
            
            autopct_function = _make_autopct(sr.values, format_string)

            legends = sr.index

        else:                           # elif aside:  Valid autopct and legends in case of 'aside' label_place
            autopct_function = None     # No data inside de pie or donut
            # Custom legends w/labels values and pct aside of the pie or donut
            total = sr.values.sum()         
            legends = [f"{sr.index[i]} \n| {value} | {round(value / total * 100, pct_decimals)} %"
                    for i, value in enumerate(sr.values)] 

        ax.pie(x=sr,
            colors=color_palette,
            startangle=startangle,
            autopct=autopct_function,
            wedgeprops=wedgeprops,
            textprops={'size': label_size,
                        'color': 'w',
                        'rotation': label_rotate,
                        'weight': 'bold'})
        
        ax.legend(legends,
                loc=legend_loc,
                bbox_to_anchor=(1, 0, 0.2, 1),
                prop={'size': legend_size})

    else:
        raise ValueError(f"Invalid labe_place parameter. Must be 'ext', 'int' or 'aside', not '{label_place}'.")
            
    # Build title and set title
    if not title:
        title = f"Pie/Donut Chart ({cat_name} - {sr.name})"
    ax.set_title(title, fontsize=title_size, fontweight='bold')

    if show_stats_subtitle:                     # Enhanced subtitle with statistics
        total_items = sr.sum()                  # Total items in the series
        n_categories = len(sr)                  # len(categories)
        top_2_pct = (sr.head(2).sum() / total_items * 100) if n_categories >= 3 else (sr.sum() / total_items * 100)

        subtitle = f"{total_label} {total_items:,} | Categories: {n_categories} | Top 2: {top_2_pct:.1f}% | Nulls (nan): {n_nans}"
        ax.text(0, 1.18, subtitle, ha='center', va='center', fontsize=title_size * 0.6, color='dimgray')

    return fig, ax


def plt_pareto(
    data: Union[pd.Series, pd.DataFrame],
    value_counts: Optional[bool] = False,
    scale: Optional[int] = 2,
    title: Optional[str] = 'Pareto Chart',
    x_label: Optional[str] = None,
    y1_label: Optional[str] = None,
    y2_label: Optional[str] = None,
    palette: Optional[list] = None,
    color1: Optional[str] = 'midnightblue',
    color2: Optional[str] = 'darkorange',
    pct_decimals: Optional[int] = 1,
    label_rotate: Optional[float] = 45,
    figsize: Optional[tuple] = None,
    fig_margin: Optional[float] = 1.1,
    show_grid: Optional[bool] = True,
    bars_alpha: Optional[float] = 0.8,
    reference_pct: Optional[float] = 80,
    reference_linewidth: float = 1,
    reference_color: str = 'red',
    reference_alpha: Optional[float] = 0.6,
    show_reference_lines: bool = True,
    scaled_cumulative: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Generates a Pareto chart with frequency bars and cumulative percentage line.

    This function creates a dual-axis chart showing category frequencies as bars
    and their cumulative percentage as a line. It supports custom styling, scaling,
    and automatic formatting of data.

    Parameters:
        data (Union[pd.Series, pd.DataFrame]): Input data. If DataFrame, it will be
            converted to a Series using `to_series`.
        value_counts (bool, optional): Whether to treat the input as raw categories
            and count frequencies. Default is False.
        scale (int, optional): Chart scaling factor (1 to 9). Affects figure size and font sizes.
            Default is 2.
        title (str, optional): Chart title. Default is 'Pareto Chart'.
        x_label (str, optional): Label for the x-axis. Default is the index name of the data.
        y1_label (str, optional): Label for the primary y-axis (frequencies). Default is the first column name.
        y2_label (str, optional): Label for the secondary y-axis (cumulative percentages).
            Default is the last column name.
        palette (list, optional): List of color names or hex codes for bar colors.
            Overrides `color1` if provided.
        color1 (str, optional): Color for the bars and primary y-axis labels. Default is 'midnightblue'.
        color2 (str, optional): Color for the cumulative percentage line and secondary y-axis labels.
            Default is 'darkorange'.
        pct_decimals (int, optional): Number of decimal places to display in percentage labels.
            Default is 1.
        label_rotate (float, optional): Rotation angle for x-axis labels. Default is 45.
        figsize (tuple, optional): Width and height of the figure in inches. If not provided,
            it is calculated based on `scale`.
        fig_margin (float, optional): Margin multiplier for y-axis limits. Default is 1.1.
        show_grid (bool, optional): Whether to show grid lines. Default is True.
        bars_alpha (float, optional): Transparency level for bars. Default is 0.8.
        reference_pct (float, optional): Reference percentage line to draw on the chart.
            Must be between 0 and 100. Default is 80.
        reference_linewidth (float, optional): Width of the reference line. Default is 1.
        reference_color (str, optional): Color of the reference line. Default is 'red'.
        reference_alpha (float, optional): Transparency of the reference line. Default is 0.6.
        show_reference_lines (bool, optional): Whether to show the reference percentage line.
            Default is True.
        scaled_cumulative (bool, optional): Whether to scale the cumulative line to match the bar axis.
            If False, uses a separate percentage axis. Default is False.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]: A tuple containing:
            - fig: The Matplotlib Figure object.
            - (ax, ax2): Primary and secondary Axes objects for further customization.

    Raises:
        TypeError: If input data is not a pandas Series or DataFrame.
        ValueError: If scale is not between 1 and 9 or reference_pct is invalid.

    Notes:
        - This function uses `get_fdt` to compute frequency distribution tables.
        - It supports rich annotations, custom palettes, and reference lines for better insights.
        - The chart includes a subtitle with summary statistics: total items, number of categories,
          top 3 contribution, and null count.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'B', 'A', 'A', 'B', 'C'])
        >>> fig, (ax, ax2) = plt_pareto(data, title='Product Defects Distribution')
        >>> plt.show()
    """

    # Convert to serie en case of DF
    if isinstance(data, pd.DataFrame):
        data = to_series(data)

    # Validate data parameter a pandas object
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"Input data must be a pandas Series or DataFrame. Got {type(data)} instead."
        )
    
    # Validate and process scale parameter
    if not (1 <= scale <= 9):
        raise ValueError(f"Invalid 'scale' value. Must be between 1 and 9, got {scale}.")
    
    scale = round(scale)
    
    # Validate reference percentage
    if reference_pct is not None and not (0 < reference_pct <= 100):
        raise ValueError(f"reference_pct must be between 0 and 100, got {reference_pct}")
    
    # Validate reference linewidth
    if reference_linewidth < 0:
        raise ValueError(f"reference_linewidth must be non-negative, got {reference_linewidth}")

    # Before getting the Frequency Distribution Table get the nulls
    nulls = data.isna().sum()

    # Get de fdt. categories=fdt.index; frequencies=fdt.iloc[:, 0]; relative_pcts=fdt.iloc[:, -2]; cumulative_pcts=fdt.iloc[:, -1]
    fdt = get_fdt(data, value_counts=value_counts, plain_relatives=False)

    # Calculate figure dimensions
    if figsize is None:
        multiplier = 1.33333334 ** scale
        w_base, h_base = 4.45, 2.25
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
    
    # Calculate font sizes based on figure width
    bar_label_size = width
    axis_label_size = width * 1.25
    title_size = width * 1.57

    # Calculate cumulative_line sizes
    markersize = width * 0.3
    linewidth = width * 0.1

    # Set up colors
    if palette:
        color_palette = get_color_list(palette, fdt.shape[0])
        color1 = color_palette[0]                                   # In this case don't consider color1 parameter
    else:
        color_palette = color1

    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    
    # Create bar plot
    bars = ax.bar(fdt.index, fdt.iloc[:, 0], 
                  color=color_palette,
                  width=0.95, 
                  alpha=bars_alpha,
                  edgecolor='white', 
                  linewidth=0.5)

    # Add value labels on bars
    labels = [f"[{fdt.iloc[ix, 0]}]  {fdt.iloc[ix, -2]:.1f} %" for ix in range(fdt.shape[0])]
    ax.bar_label(bars,
                labels=labels,
                fontsize=bar_label_size * 0.9,
                fontweight='bold',
                color=color1,
                label_type='edge',  # Etiqueta fuera de la barra
                padding=2)          #, rotation=90)  # opcional

    # Create secondary y-axis for cumulative percentage
    ax2 = ax.twinx()        # create another y-axis sharing a common x-axis
    
    # Calculate cumulative values
    cumulative_percentages = fdt.iloc[:, -1]            # Last column: ['Cumulative Freq. [%]']
    
    if scaled_cumulative:                               # Scaling mode fixed
        total_sum = fdt.iloc[:, 0].sum()
        
        # Convert cumulative percentages to scaled heightsdas
        scaled_values = (cumulative_percentages / 100) * total_sum
        
        # Draw the scaled line on the main axis (x=index, y=scaled_values)
        line = ax.plot(fdt.index, scaled_values,
                       color=color2,
                       marker="D",
                       markersize=markersize,
                       linewidth=linewidth,
                       markeredgecolor='white',
                       markeredgewidth=0.2)
        
        # Adjust main axis limits to include the line
        max_freq = fdt.iloc[:, 0].max()
        max_scaled = scaled_values.max()
        # Use the maximum between the bars and the scaled line, with margin
        ax.set_ylim(0, max(max_freq, max_scaled) * fig_margin)
        
        # CORRECCIÓN: Configurar ax2 para que coincida con la escala del eje principal
        ax2.set_ylim(0, max(max_freq, max_scaled) * fig_margin)
        
        # Create custom stickers for ax2 that show percentages, corresponding to the climbed heights
        ax2_ticks = []
        ax2_labels = []
        for pct in [0, 20, 40, 60, 80, 100]:
            scaled_tick = (pct / 100) * total_sum
            if scaled_tick <= max(max_freq, max_scaled) * fig_margin:
                ax2_ticks.append(scaled_tick)
                ax2_labels.append(f'{pct}%')
        
        ax2.set_yticks(ax2_ticks)
        ax2.set_yticklabels(ax2_labels)
        
        # % point labels
        formatted_weights = [f'{x:.{pct_decimals}f}%' for x in cumulative_percentages]
        for i, txt in enumerate(formatted_weights):
            if i == 0:              # To change only % annotate of the first bar         
                distance = 0.08     # first % annotate, away from the bar
            else:
                distance = 0.025    # The others % annotates, not so far
            ax.annotate(txt,
                       (fdt.index[i], scaled_values.iloc[i] + (max(max_freq, max_scaled) * distance)),
                       color=color2,
                       fontsize=bar_label_size,
                       ha='center')
        
        # Reference lines in scaled mode
        if show_reference_lines and reference_pct is not None:
            reference_scaled_height = (reference_pct / 100) * total_sum
            
            # AXHLINE and its text
            ax.axhline(y=reference_scaled_height, color=reference_color, linestyle='--', 
                      alpha=reference_alpha, linewidth=reference_linewidth)
            
            ax.text(0.01, reference_scaled_height + (max(max_freq, max_scaled) * 0.02), 
                   f'{reference_pct}%', 
                   transform=ax.get_yaxis_transform(), 
                   color=reference_color, fontsize=bar_label_size*0.8)
    
    else:                                           # Native scaling
        ax2.set_ylim(0, 100 * fig_margin)
        
        line = ax2.plot(fdt.index, cumulative_percentages,
                        color=color2,
                        marker="D",
                        markersize=markersize,
                        linewidth=linewidth,
                        markeredgecolor='white',
                        markeredgewidth=0.2)
        
        ax2.yaxis.set_major_formatter(PercentFormatter())

        formatted_weights = [f'{x:.{pct_decimals}f}%' for x in cumulative_percentages]  
        for i, txt in enumerate(formatted_weights):
                ax2.annotate(txt,
                            (fdt.index[i], cumulative_percentages.iloc[i] - 6),
                            color=color2,
                            fontsize=bar_label_size,
                            ha='center')
        
        if show_reference_lines and reference_pct is not None:
            ax2.axhline(y=reference_pct, color=reference_color, linestyle='--', 
                       alpha=reference_alpha, linewidth=reference_linewidth)
            
            ax2.text(0.01, reference_pct + 3, f'{reference_pct}%', 
                        transform=ax2.get_yaxis_transform(), 
                        color=reference_color, fontsize=bar_label_size*0.8)

    # Configure tick parameters
    ax.tick_params(axis='y', colors=color1, labelsize=bar_label_size)
    ax.tick_params(axis='x', rotation=label_rotate, labelsize=bar_label_size)
    ax2.tick_params(axis='y', colors=color2, labelsize=bar_label_size)

    # Set y-axis limits (solo para modo original)
    if not scaled_cumulative:
        max_freq = fdt.iloc[:, 0].max()
        ax.set_ylim(0, max_freq * fig_margin)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    # Set title and labels
    if not x_label:
         x_label = fdt.index.name
    
    if not y1_label:
         y1_label = fdt.columns[0]

    if not y2_label:
         y2_label = fdt.columns[-1]

    # Enhanced subtitle with statistics
    total_items = fdt.iloc[:, 0].sum()      # frequencies.sum()
    n_categories = len(fdt.index)           # len(categories)
    top_3_pct = cumulative_percentages.iloc[min(2, len(cumulative_percentages)-1)]      # if len(cum_pcts) < 2
    subtitle = f"Total: {total_items:,} | Categories: {n_categories} | Top 3: {top_3_pct:.1f}% | Nulls: {nulls}"

    # Apply title and labels
    fig.suptitle(title, fontsize=title_size, fontweight='bold')
    ax.set_title(subtitle, fontsize=axis_label_size*0.8, color=color1, pad=10)
    ax.set_xlabel(x_label, fontsize=axis_label_size, fontweight='medium')
    ax.set_ylabel(y1_label, fontsize=axis_label_size, color=color1, fontweight='medium')
    ax2.set_ylabel(y2_label, fontsize=axis_label_size, color=color2, fontweight='medium')

    return fig, (ax, ax2)


def sns_pareto(
    data: Union[pd.Series, pd.DataFrame],
    value_counts: bool = False,
    scale: Optional[int] = 2,
    title: Optional[str] = 'Pareto Chart',
    x_label: Optional[str] = None,
    y1_label: Optional[str] = None,
    y2_label: Optional[str] = None,
    palette: Optional[str] = 'husl',
    palette_type: Literal['qualitative', 'sequential', 'diverging'] = 'qualitative',
    color1: Optional[str] = 'steelblue',
    color2: Optional[str] = 'coral',
    theme: Optional[str] = 'whitegrid',
    context: Literal['paper', 'notebook', 'talk', 'poster'] = 'notebook',
    pct_decimals: Optional[int] = 1,
    label_rotate: Optional[float] = 45,
    figsize: Optional[tuple] = None,
    fig_margin: Optional[float] = 1.15,
    show_grid: Optional[bool] = True,
    grid_alpha: Optional[float] = 0.3,
    bars_alpha: Optional[float] = 0.85,
    reference_pct: Optional[float] = 80,
    reference_linewidth: float = 2,
    reference_color: str = 'crimson',
    reference_alpha: Optional[float] = 0.8,
    show_reference_lines: bool = True,
    scaled_cumulative: bool = False,
    annotation_style: Literal['outside', 'inside', 'edge'] = 'outside',
    show_confidence_interval: bool = False,
    confidence_level: float = 0.95,
    bar_edge_color: str = 'white',
    bar_edge_width: float = 0.8,
    rounded_bars: bool = True,
    sorting: Literal['frequency', 'alphabetical', 'custom'] = 'frequency',
    custom_order: Optional[list] = None,
    show_statistics: bool = True,
    modern_styling: bool = True,
    line_style: Literal['solid', 'dashed', 'dotted'] = 'solid',
    marker_style: str = 'o',
    gradient_bars: bool = False,
    show_percentages_on_bars: bool = True,
    show_legend: bool = True,
    legend_position: str = 'upper right',
    use_sns_palette_colors: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Create an enhanced Pareto chart using Seaborn with modern styling and professional appearance.
    
    A Pareto chart is a bar chart where the bars are ordered by frequency/value in descending order,
    with a cumulative percentage line overlaid. This enhanced version includes modern styling,
    statistical features, and improved visual customization.
    
    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        Input data for the Pareto chart
    value_counts : bool, default False
        Whether to apply value_counts to the data
    scale : Optional[int], default 2
        Scale factor for figure sizing (1-9)
    title : Optional[str], default 'Pareto Chart'
        Chart title
    x_label : Optional[str], default None
        X-axis label
    y1_label : Optional[str], default None
        Primary y-axis label
    y2_label : Optional[str], default None
        Secondary y-axis label
    palette : Optional[str], default 'husl'
        Seaborn color palette name ('husl', 'viridis', 'Set1', 'plasma', etc.)
    palette_type : Literal['qualitative', 'sequential', 'diverging'], default 'qualitative'
        Type of color palette to use
    color1 : Optional[str], default 'steelblue'
        Primary color for bars (used when palette is None)
    color2 : Optional[str], default 'coral'
        Secondary color for cumulative line
    theme : Optional[str], default 'whitegrid'
        Seaborn theme ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
    context : Literal['paper', 'notebook', 'talk', 'poster'], default 'notebook'
        Seaborn context for scaling elements
    pct_decimals : Optional[int], default 1
        Decimal places for percentage labels
    label_rotate : Optional[float], default 45
        Rotation angle for x-axis labels
    figsize : Optional[tuple], default None
        Figure size (width, height)
    fig_margin : Optional[float], default 1.15
        Margin multiplier for y-axis limits
    show_grid : Optional[bool], default True
        Whether to show grid
    grid_alpha : Optional[float], default 0.3
        Grid transparency
    bars_alpha : Optional[float], default 0.85
        Transparency for bars
    reference_pct : Optional[float], default 80
        Reference percentage for horizontal line
    reference_linewidth : float, default 2
        Line width for reference lines
    reference_color : str, default 'crimson'
        Color for reference lines
    reference_alpha : Optional[float], default 0.8
        Transparency for reference lines
    show_reference_lines : bool, default True
        Whether to show reference lines
    scaled_cumulative : bool, default False
        Whether to scale cumulative line to match bar heights
    annotation_style : Literal['outside', 'inside', 'edge'], default 'outside'
        Position of value annotations on bars
    show_confidence_interval : bool, default False
        Whether to show confidence interval for cumulative line
    confidence_level : float, default 0.95
        Confidence level for intervals
    bar_edge_color : str, default 'white'
        Color of bar edges
    bar_edge_width : float, default 0.8
        Width of bar edges
    rounded_bars : bool, default True
        Whether to use rounded bar corners (visual effect)
    sorting : Literal['frequency', 'alphabetical', 'custom'], default 'frequency'
        How to sort the categories
    custom_order : Optional[list], default None
        Custom order for categories (used when sorting='custom')
    show_statistics : bool, default True
        Whether to show statistical summary in legend
    modern_styling : bool, default True
        Whether to apply modern styling enhancements
    line_style : Literal['solid', 'dashed', 'dotted'], default 'solid'
        Style of the cumulative line
    marker_style : str, default 'o'
        Marker style for cumulative line points
    gradient_bars : bool, default False
        Whether to apply gradient effect to bars
    show_percentages_on_bars : bool, default True
        Whether to show individual percentages on bars
    show_legend : bool, default True
        Whether to show legend
    legend_position : str, default 'upper right'
        Position of the legend
    use_sns_palette_colors : bool, default True
        Whether to use seaborn palette colors for bars
    
    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Figure and tuple of primary and secondary axes
    """
    
    # Set seaborn theme and context
    if theme:
        sns.set_style(theme)
    if context:
        sns.set_context(context)
    
    # Convert to series if DataFrame
    if isinstance(data, pd.DataFrame):
        data = to_series(data)  # Assuming this function exists
    
    # Validate data parameter
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"Input data must be a pandas Series or DataFrame. Got {type(data)} instead."
        )
    
    # Validate and process scale parameter
    if not (1 <= scale <= 9):
        raise ValueError(f"Invalid 'scale' value. Must be between 1 and 9, got {scale}.")
    
    scale = round(scale)
    
    # Validate reference percentage
    if reference_pct is not None and not (0 < reference_pct <= 100):
        raise ValueError(f"reference_pct must be between 0 and 100, got {reference_pct}")
    
    # Validate reference linewidth
    if reference_linewidth < 0:
        raise ValueError(f"reference_linewidth must be non-negative, got {reference_linewidth}")
    
    # Count nulls before processing
    nulls = data.isna().sum()
    
    # Get frequency distribution table
    fdt = get_fdt(data, value_counts=value_counts, plain_relatives=False)  # Assuming this function exists
    
    # Apply sorting
    if sorting == 'alphabetical':
        fdt = fdt.sort_index()
        # Recalculate cumulative percentages after sorting
        fdt.iloc[:, -1] = (fdt.iloc[:, 0].cumsum() / fdt.iloc[:, 0].sum()) * 100
    elif sorting == 'custom' and custom_order:
        available_categories = set(fdt.index)
        valid_order = [cat for cat in custom_order if cat in available_categories]
        if valid_order:
            fdt = fdt.reindex(valid_order)
            # Recalculate cumulative percentages after reordering
            fdt.iloc[:, -1] = (fdt.iloc[:, 0].cumsum() / fdt.iloc[:, 0].sum()) * 100
    # 'frequency' is the default and doesn't need special handling
    
    # Calculate figure dimensions
    if figsize is None:
        multiplier = 1.33333334 ** scale
        w_base, h_base = 4.8, 2.4  # Slightly larger base for modern look
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
    
    # Calculate font sizes based on figure width and context
    context_multipliers = {'paper': 0.8, 'notebook': 1.0, 'talk': 1.2, 'poster': 1.4}
    ctx_mult = context_multipliers.get(context, 1.0)
    
    bar_label_size = width * ctx_mult
    axis_label_size = width * 1.25 * ctx_mult
    title_size = width * 1.6 * ctx_mult
    
    # Calculate line properties
    markersize = width * 0.35 * ctx_mult
    linewidth = width * 0.12 * ctx_mult
    
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    
    # Apply modern styling
    if modern_styling:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        # Remove top and right spines for cleaner look
        sns.despine(ax=ax, top=True, right=False)
    
    # Prepare data for plotting
    categories = fdt.index
    frequencies = fdt.iloc[:, 0]
    cumulative_percentages = fdt.iloc[:, -1]
    
    # Set up colors
    if use_sns_palette_colors and palette:
        if palette_type == 'qualitative':
            colors = sns.color_palette(palette, len(categories))
        elif palette_type == 'sequential':
            colors = sns.color_palette(palette, len(categories))
        elif palette_type == 'diverging':
            colors = sns.color_palette(palette, len(categories))
        else:
            colors = sns.color_palette(palette, len(categories))
    else:
        colors = color1

    # Create the bar plot with enhanced styling
    if use_sns_palette_colors and palette:
        bars = sns.barplot(
            x=categories,
            y=frequencies,
            hue=categories,  # Add this line - assign x variable to hue
            palette=colors,
            alpha=bars_alpha,
            ax=ax,
            edgecolor=bar_edge_color,
            linewidth=bar_edge_width,
            saturation=0.9,
            legend=False  # Add this to prevent redundant legend
        )
    else:
        bars = sns.barplot(
            x=categories,
            y=frequencies,
            color=color1,
            alpha=bars_alpha,
            ax=ax,
            edgecolor=bar_edge_color,
            linewidth=bar_edge_width,
            saturation=0.9
        )
    
    # Apply gradient effect if requested
    if gradient_bars:
        for i, bar in enumerate(bars.patches):
            # Create gradient effect by varying alpha
            gradient_alpha = 0.6 + (0.4 * (len(bars.patches) - i) / len(bars.patches))
            bar.set_alpha(gradient_alpha)
    
    # Add value annotations on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        
        # Determine annotation position based on style
        if annotation_style == 'outside':
            y_pos = height + (frequencies.max() * 0.02)
            va = 'bottom'
        elif annotation_style == 'inside':
            y_pos = height * 0.5
            va = 'center'
        else:  # edge
            y_pos = height + (frequencies.max() * 0.005)
            va = 'bottom'
        
        # Add frequency annotation
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{int(height)}',
                ha='center', va=va,
                fontsize=bar_label_size * 0.9,
                fontweight='bold',
                color=color1 if annotation_style == 'outside' else 'white')
        
        # Add percentage on bars if requested
        if show_percentages_on_bars:
            pct = (height / frequencies.sum()) * 100
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height * 0.85 if annotation_style == 'outside' else height * 0.15,
                   f'{pct:.1f}%',
                   ha='center', va='center',
                   fontsize=bar_label_size * 0.7,
                   color='white' if annotation_style == 'outside' else color2,
                   fontweight='medium')
    
    # Create secondary y-axis for cumulative percentage
    ax2 = ax.twinx()
    
    # Prepare line style
    line_styles = {'solid': '-', 'dashed': '--', 'dotted': ':'}
    ls = line_styles.get(line_style, '-')
    
    if scaled_cumulative:
        # Scaling mode - scale cumulative percentages to match bar heights
        total_sum = frequencies.sum()
        scaled_values = (cumulative_percentages / 100) * total_sum
        
        # Plot cumulative line on primary axis
        line_data = pd.DataFrame({
            'x': range(len(categories)),
            'y': scaled_values
        })
        
        # Main line
        sns.lineplot(
            data=line_data,
            x='x',
            y='y',
            color=color2,
            marker=marker_style,
            markersize=markersize,
            linewidth=linewidth,
            markeredgecolor='white',
            markeredgewidth=0.3,
            linestyle=ls,
            ax=ax,
            label='Cumulative %'
        )
        
        # Add confidence interval if requested
        if show_confidence_interval:
            # Calculate confidence interval (simplified approach)
            ci_width = scaled_values.std() * 1.96 / np.sqrt(len(scaled_values))
            ax.fill_between(range(len(categories)), 
                           scaled_values - ci_width, 
                           scaled_values + ci_width,
                           alpha=0.2, color=color2)
        
        # Adjust main axis limits
        max_freq = frequencies.max()
        max_scaled = scaled_values.max()
        ax.set_ylim(0, max(max_freq, max_scaled) * fig_margin)
        
        # Configure ax2 to match primary axis scale
        ax2.set_ylim(0, max(max_freq, max_scaled) * fig_margin)
        
        # Create custom ticks for ax2
        ax2_ticks = []
        ax2_labels = []
        for pct in [0, 20, 40, 60, 80, 100]:
            scaled_tick = (pct / 100) * total_sum
            if scaled_tick <= max(max_freq, max_scaled) * fig_margin:
                ax2_ticks.append(scaled_tick)
                ax2_labels.append(f'{pct}%')
        
        ax2.set_yticks(ax2_ticks)
        ax2.set_yticklabels(ax2_labels)
        
        # Add percentage labels with improved positioning
        for i, (cat, pct, scaled_val) in enumerate(zip(categories, cumulative_percentages, scaled_values)):
            distance = 0.06 if i == 0 else 0.02
            ax.text(i, scaled_val + (max(max_freq, max_scaled) * distance),
                   f'{pct:.{pct_decimals}f}%',
                   ha='center', va='bottom',
                   color=color2,
                   fontsize=bar_label_size * 0.8,
                   fontweight='medium')
        
        # Reference lines in scaled mode
        if show_reference_lines and reference_pct is not None:
            reference_scaled_height = (reference_pct / 100) * total_sum
            
            # Horizontal reference line
            ax.axhline(y=reference_scaled_height, color=reference_color, linestyle='--',
                      alpha=reference_alpha, linewidth=reference_linewidth)
            
            ax.text(0.02, reference_scaled_height + (max(max_freq, max_scaled) * 0.02),
                   f'{reference_pct}%',
                   transform=ax.get_yaxis_transform(),
                   color=reference_color, fontsize=bar_label_size*0.8,
                   fontweight='bold')
            
            # Vertical reference line
            cumulative_values = cumulative_percentages.values
            x_reference_percent = None
            for i, cum_pct in enumerate(cumulative_values):
                if cum_pct >= reference_pct:
                    if i == 0:
                        x_reference_percent = 0
                    else:
                        prev_pct = cumulative_values[i-1]
                        curr_pct = cumulative_values[i]
                        x_reference_percent = (i-1) + (reference_pct - prev_pct) / (curr_pct - prev_pct)
                    break
            
            if x_reference_percent is not None:
                ax.axvline(x=x_reference_percent, color=reference_color, linestyle='--',
                          alpha=reference_alpha, linewidth=reference_linewidth)
                
                ax.text(x_reference_percent + 0.1,
                       reference_scaled_height - (max(max_freq, max_scaled) * 0.12),
                       f'{reference_pct}% rule',
                       rotation=90, color=reference_color, fontsize=bar_label_size*0.7,
                       ha='left', va='center', fontweight='bold')
    
    else:
        # Native scaling mode
        ax2.set_ylim(0, 100 * fig_margin)
        
        # Plot cumulative line on secondary axis
        line_data = pd.DataFrame({
            'x': range(len(categories)),
            'y': cumulative_percentages
        })
        
        # Main line
        sns.lineplot(
            data=line_data,
            x='x',
            y='y',
            color=color2,
            marker=marker_style,
            markersize=markersize,
            linewidth=linewidth,
            markeredgecolor='white',
            markeredgewidth=0.3,
            linestyle=ls,
            ax=ax2,
            label='Cumulative %'
        )
        
        # Add confidence interval if requested
        if show_confidence_interval:
            ci_width = cumulative_percentages.std() * 1.96 / np.sqrt(len(cumulative_percentages))
            ax2.fill_between(range(len(categories)), 
                           cumulative_percentages - ci_width, 
                           cumulative_percentages + ci_width,
                           alpha=0.2, color=color2)
        
        ax2.yaxis.set_major_formatter(PercentFormatter())
        
        # Add percentage labels with improved styling
        for i, (cat, pct) in enumerate(zip(categories, cumulative_percentages)):
            ax2.text(i, pct - 8,
                    f'{pct:.{pct_decimals}f}%',
                    ha='center', va='top',
                    color=color2,
                    fontsize=bar_label_size * 0.8,
                    fontweight='medium')
        
        # Reference lines in native mode
        if show_reference_lines and reference_pct is not None:
            ax2.axhline(y=reference_pct, color=reference_color, linestyle='--',
                       alpha=reference_alpha, linewidth=reference_linewidth)
            
            ax2.text(0.02, reference_pct + 4, f'{reference_pct}%',
                    transform=ax2.get_yaxis_transform(),
                    color=reference_color, fontsize=bar_label_size*0.8,
                    fontweight='bold')
            
            # Vertical reference line
            cumulative_values = cumulative_percentages.values
            x_reference_percent = None
            for i, cum_pct in enumerate(cumulative_values):
                if cum_pct >= reference_pct:
                    if i == 0:
                        x_reference_percent = 0
                    else:
                        prev_pct = cumulative_values[i-1]
                        curr_pct = cumulative_values[i]
                        x_reference_percent = (i-1) + (reference_pct - prev_pct) / (curr_pct - prev_pct)
                    break
            
            if x_reference_percent is not None:
                ax2.axvline(x=x_reference_percent, color=reference_color, linestyle='--',
                           alpha=reference_alpha, linewidth=reference_linewidth)
                
                ax2.text(x_reference_percent + 0.1, reference_pct - 35,
                         f'{reference_pct}% rule',
                         rotation=90, color=reference_color, fontsize=bar_label_size*0.7,
                         ha='left', va='center', fontweight='bold')
    
    # Configure tick parameters with modern styling
    ax.tick_params(axis='y', colors=color1, labelsize=bar_label_size * 0.9)
    ax.tick_params(axis='x', rotation=label_rotate, labelsize=bar_label_size * 0.9)
    ax2.tick_params(axis='y', colors=color2, labelsize=bar_label_size * 0.9)
    
    # Set y-axis limits for primary axis (only in native mode)
    if not scaled_cumulative:
        max_freq = frequencies.max()
        ax.set_ylim(0, max_freq * fig_margin)
    
    # Add enhanced grid
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Set default labels if not provided
    if not x_label:
        x_label = fdt.index.name or 'Categories'
    
    if not y1_label:
        y1_label = fdt.columns[0] if len(fdt.columns) > 0 else 'Frequency'
    
    if not y2_label:
        y2_label = fdt.columns[-1] if len(fdt.columns) > 0 else 'Cumulative %'
    
    # Apply title and labels with improved styling
    fig.suptitle(title, fontsize=title_size, fontweight='bold', y=0.98)
    
    # Enhanced subtitle with statistics
    if show_statistics:
        total_items = frequencies.sum()
        n_categories = len(categories)
        top_3_pct = cumulative_percentages.iloc[min(2, len(cumulative_percentages)-1)]
        
        subtitle = f"Total: {total_items:,} | Categories: {n_categories} | Top 3: {top_3_pct:.1f}% | Nulls: {nulls}"
        ax.set_title(subtitle, fontsize=axis_label_size*0.7, color='gray', pad=10)
    else:
        ax.set_title(f"Nulls: {nulls}", fontsize=axis_label_size*0.8, color=color1, pad=10)
    
    ax.set_xlabel(x_label, fontsize=axis_label_size, fontweight='medium')
    ax.set_ylabel(y1_label, fontsize=axis_label_size, color=color1, fontweight='medium')
    ax2.set_ylabel(y2_label, fontsize=axis_label_size, color=color2, fontweight='medium')
    
    # Add legend if requested
    if show_legend:
        # Create custom legend entries
        legend_elements = []
        
        if use_sns_palette_colors and palette:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=bars_alpha, 
                                               edgecolor=bar_edge_color, label='Frequency'))
        else:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color1, alpha=bars_alpha, 
                                               edgecolor=bar_edge_color, label='Frequency'))
        
        legend_elements.append(plt.Line2D([0], [0], color=color2, marker=marker_style, 
                                        markersize=markersize*0.7, label='Cumulative %', linestyle=ls))
        
        if show_reference_lines and reference_pct is not None:
            legend_elements.append(plt.Line2D([0], [0], color=reference_color, linestyle='--', 
                                            alpha=reference_alpha, label=f'{reference_pct}% Rule'))
        
        ax.legend(handles=legend_elements, loc=legend_position, frameon=True, 
                 fancybox=True, shadow=True, fontsize=bar_label_size*0.8)
    
    # Final modern styling touches
    if modern_styling:
        # Adjust layout
        plt.tight_layout()
        
        # Add subtle shadow to bars
        for bar in bars.patches:
            bar.set_edgecolor(bar_edge_color)
            bar.set_linewidth(bar_edge_width)
    
    return fig, (ax, ax2)





if __name__ == "__main__":

    df = pd.DataFrame({'A': [1, 2, pd.NA, pd.NA, 1],
                       'B': [4.0, pd.NA, pd.NA, 6.1, 4.0],
                       'C': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                       'D': ['x', 'y', pd.NA, 'z', 'x'],
                       'E': ['x', 'y', pd.NA, 'z', 'x']})
    
    print(df)

    df2 = clean_df(df)
    
    print(df2)
        


