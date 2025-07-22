"""
jm_pandas
"""

## TO-DO
# paretto chart calc cumulative % or pass as argument ..... make an fdt function..

## fdt!
# Considerar la opción de adicionar los nulls-nans-pd.NAs opcionalmente a la fdt
# EN REALIDAD el tema de los nans lo tengo que ver en el to_series_with_count()

## Pareto - proportional sizes
#   title - qué - size - etc
#   labels, legends, numbers, line size  - size!!

## Pie
#   Versión con TODO los labels EXTERNO

## OJO con la doble función de formateo de datos que tengo... OJO
# porque debería ajustar tanto esta que tengo acá com la de jm_rchprt o DEJAR SOLO UNA!!!???
# NO SE si conviene hacer dos porque en el caso de series tengo que considerar que NO es bueno mezclar n decimal con 0 decimals en una MISMA Series
# caso del cumulative relative frequency

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

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # for pareto chart and ?
import seaborn as sns
## Claude - Qwen


## Custom types for non-included typing annotations - Grok
IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', pd.Timestamp]


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
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError(f"Decimals must be a non-negative integer. Not '{decimals}")
    
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


def to_series(
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    index: Optional[Union[pd.Index, Sequence[Union[str, int, float, 'datetime.datetime']], np.ndarray]] = None,
    name: Optional[str] = None
) -> pd.Series:
    """
    Converts input data into a pandas Series, optionally returning value counts.

    This function accepts various data types and converts them into a pandas Series.
    If `count=True`, it returns the frequency count of the values in the resulting Series.

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]):
            The input data to convert. Supported types include:
            - pd.Series: returned as-is or counted if `count=True`.
            - np.ndarray: flattened and converted to a Series.
            - dict: keys become the index, values are used for data.
            - list or set: converted directly to a Series.
            - pd.DataFrame:
                - 1 column: converted directly to a Series.
                - 2 columns: first column becomes the index, second becomes the values.

        count (bool or int, optional): Whether to return value counts instead of raw data.
            If True or 1, returns frequencies of each value. Default is False.

    Returns:
        pd.Series: A pandas Series representing the input data. If `count=True`, returns
            the value counts of the data.

    Raises:
        TypeError: If `data` is not one of the supported types.
        ValueError: If `count` is not a boolean or integer 0/1.
        ValueError: If DataFrame has more than 2 columns.

    Examples:
        >>> import pandas as pd
        >>> to_serie_with_count([1, 2, 2, 3])
        0    1
        1    2
        2    2
        3    3
        dtype: int64

        >>> to_serie_with_count([1, 2, 2, 3], count=True)
        2    2
        1    1
        3    1
        dtype: int64

        >>> df = pd.DataFrame({'Category': ['A', 'B', 'A'], 'Value': [10, 20, 30]})
        >>> to_serie_with_count(df)
        Category
        A    10
        B    20
        A    30
        Name: Value, dtype: int64
    """
    
    # Validate parameters - FUTURE
    
    if isinstance(data, pd.Series):                 # If series is already a Series no conversion needed
        series = data                                  
    elif isinstance(data, np.ndarray):              # If data is a NumPy array   
        series = pd.Series(data.flatten())
    elif isinstance(data, (dict, list)):
        series = pd.Series(data)
    elif isinstance(data, (set)):
        series = pd.Series(tuple(data))
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

                      
# Create a complete frecuency distribution table fron a categorical data
def get_fdt(
        data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
        value_counts: Optional[bool] = False,
        dropna: Optional[bool] = True,
        na_position: Optional[str] = 'last',
        pcts: Optional[bool] = True,
        plain_relatives: Optional[bool] = True,
        fmt_values: Optional[bool] = False,
        sort: Optional[str] = 'desc',
        na_aside: Optional[bool] = True
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
        pcts (bool, optional): Whether to include percentage columns.
            If False, only absolute and cumulative frequencies are returned.
            Default is True.
        plain_relatives (bool, optional): Whether to return relative and cumulative relative values.
            If False, only frequency and percentage columns are included.
            Default is True.
        fmt_values (bool, optional): Whether to format numeric values using `_fmt_value_for_pd`.
            Useful for improving readability in reports. Default is False.
        sort (str, optional): Sort order for the output:
            - 'asc': Sort values ascending.
            - 'desc': Sort values descending (default).
            - 'ix_asc': Sort by index ascending.
            - 'ix_desc': Sort by index descending.
            - None: No sorting.
            Default is 'desc'.
        na_aside (bool, optional): Whether to separate NaN values from calculations but keep them in the output.
            If True, NaNs are added at the end and not included in cumulative or relative calculations.
            Default is True.

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
        sr = sr.dropna()

    if value_counts:
        sr = sr.value_counts(dropna=dropna, sort=False)

    match sort:
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
            raise ValueError(f"Valid values for sort: 'asc', 'desc', 'ix_asc', 'ix_desc', or None. Got '{sort}'")

    try:                            # To manage when there aren't NaNs
        nan_value = sr[np.nan]
        sr_without_nan = sr.drop(np.nan)
    except:
        pass
    else:                           # if NaNs: 1. na_position, 2 na_count
        match na_position:          # 1. locate the NaNs values
            case 'first':
                sr = pd.concat([pd.Series({np.nan: nan_value}), sr_without_nan])
            case 'last':
                sr = pd.concat([sr_without_nan, pd.Series({np.nan: nan_value})])
            case 'value' | None:
                pass
            case _:
                raise ValueError(f"Valid values for na_position: 'first', 'last', 'value' or None. Got '{na_position}'")
        
        if na_aside:                # 2. define if NaNs count for relative and cumulative values.
            sr = sr_without_nan     # series without nulls on which the relative values will be calculated
            # Column that will then be concatenated to the end of the DF if the na_aside option is true
            nan_row_df = pd.DataFrame(data = [nan_value], columns=[columns[0]], index=['Nulls'])      # Only 'Frequency' column, others empty

    # Central rutine: Cumulative and relative frequencies
    fdt = pd.DataFrame(sr)
    fdt.columns = [columns[0]]
    fdt[columns[1]] = fdt['Frequency'].cumsum()
    fdt[columns[2]] = fdt['Frequency'] / fdt['Frequency'].sum()
    fdt[columns[3]] = fdt['Relative Frequency'].cumsum()
    fdt[columns[4]] = fdt['Relative Frequency'] * 100
    fdt[columns[5]] = fdt['Cumulative Relative Freq.'] * 100

    if na_aside and not dropna:      # We add nan_columns at the end
        fdt = pd.concat([fdt, nan_row_df])

    if not pcts:                    # Don't return percentage columns
        fdt = fdt[columns[0:4]]
    
    if not plain_relatives:         # Don't return relative and plain cumulative
        fdt = fdt[[columns[0], columns[4], columns[5]]]

    if fmt_values:
        fdt = fdt.map(_fmt_value_for_pd)
        
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


def is_mostly_numeric(serie, threshold):
    ''' Checks if at least 'threshold'% of the values ​​can be numeric'''
    converted = pd.to_numeric(serie, errors='coerce')
    numeric_ratio = converted.notna().sum() / len(serie)
    return numeric_ratio >= threshold


def petty_decimals_and_str(serie):
    for ix, value in serie.items():
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


def get_colorblind_palette_list():
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


def get_colors_list(palette: str, n: Optional[int] = 10) -> list[str]:
    '''
    Return a valid matplotlib palette list
    - 'colorbind', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'set3', 'set2'
    - 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu'
    - 'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
    - 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r'
    - 'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'winter', 'winter_r'",
    '''

    if palette == 'colorblind':
        colors_list = get_colorblind_palette_list()
    elif palette == 'set2':
        colors_list = plt.cm.Set2(np.linspace(0, 1, n))
    elif palette == 'set3':
        colors_list = plt.cm.Set3(np.linspace(0, 1, n))
    else:
        cmap = plt.get_cmap(palette, n)              # Use palette colormap
        colors_list = [cmap(i) for i in range(n)]    # Get colors from the colormap

    return colors_list


def _validate_numeric_series(
        data: Union[pd.Series, pd.DataFrame],
        positive: Optional[bool] = True
) -> Union[None, Exception]:

    # Validate data parameter a pandas object
    if not isinstance(data, (pd.Series, pd.DataFrame)):     # pd.Series or pd.Datafram
        raise TypeError(
            f"Input data must be a pandas Series or DataFrame. Got {type(data)} instead."
        )
              
    if positive:
        if not all(                                             # Only positve numeric values                 
            isinstance(val, (int, float, np.integer, np.floating)) and val > 0 for val in data.values
        ):
            raise ValueError(f"All values in 'data' must be positive numeric values.")
        pass
    else:                                                       # Just only numeric values
        if not all(isinstance(val, (int, float, np.integer, np.floating)) for val in data.values):
            raise ValueError(f"All values in 'data' must be numeric values.")
        pass


def plt_pie(
    data: Union[pd.Series, pd.DataFrame],
    value_counts: Optional[bool] = False,
    sort: Optional[bool] = True,
    nans: Optional[bool] = False,
    scale: Optional[int] = 1,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    label_place: Optional[str] = 'ext',
    palette: Optional[list] = 'colorblind',
    startangle: Optional[float] = -40,
    pct_decimals: Optional[int] = 1,
    label_rotate: Optional[float] = 0,
    legend_loc: Optional[str] = 'best',
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

    # Convert to serie in case of np.ndarray, dict, list, set, pd.DataFrame
    sr = to_series(data)

    if value_counts:
        sr = sr.value_counts(sort=sort, dropna=not nans)

    _validate_numeric_series(sr)

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

    # Configure wedge properties for donut  or pie chart
    wedgeprops = {}
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.54, 'edgecolor': 'white', 'linewidth': 1}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 0.5}

    # Define colors
    color_palette = get_colors_list(palette, len(sr))

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
            
    # Build title
    if not title:
        title = f"Pie/Donut Chart - ({sr.name})"
    ax.set_title(title, fontdict={'size': title_size, 'weight': 'bold'})

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
        color_palette = get_colors_list(palette, fdt.shape[0])
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
        


