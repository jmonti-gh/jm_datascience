"""
jm_pandas
"""

## TO-DO
# paretto chart calc cumulative % or pass as argument ..... make an fdt function..

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
from typing import Union, Optional, Any

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # for pareto chart and ?
## Claude - Qwen

# An auxiliar function to change num format - OJO se puede hacer más amplia como jm_utils.jm_rchprt.fmt...
def _fmt_value_for_pd(value, width=8, decimals=2, miles=',') -> str:
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
    # Paralmeter Value validation <- vamos a tener que analizar este tema por si es un list , etc,,
    # if not isinstance(value, (int, float, np.integer, np.floating)) or pd.api.types.is_any_real_numeric_dtype(value)

    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"Width must be a positive integer. Not '{width}'")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError(f"Decimals must be a non-negative integer. Not '{decimals}")
    
    if miles not in [',', '_', None]:
        raise ValueError(f"Miles must be either ',', '_', or None. Not '{miles}")
    
    try:
        num = float(value)                                  # Convert to float if possible
        if num % 1 == 0:                                    # it its a total integer number
            decimals = 0
        if miles:
            return f"{num:>{width}{miles}.{decimals}f}"     # Ancho fijo, x decimales, alineado a la derecha
        else:
            return f"{num:>{width}.{decimals}f}"
        
    except (ValueError, TypeError):
        return str(value).rjust(width)                      # Alinea también strings, para mantener la grilla


def to_serie_with_count(
    data: Union[pd.Series, np.ndarray, dict, list, pd.DataFrame],
    must_count: Optional[bool] = False
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
    
    # Validate count parameter
    if not isinstance(must_count, (bool, int)):
        return TypeError(f"* count must be bool or int 0 or 1. Not '{type(data)}'.")
    if isinstance(must_count, int) and must_count not in (0, 1):
        return ValueError(f"* count as int must be 0 or 1. Not '{must_count}'.")
    
    if isinstance(data, pd.Series):                 # If series is already a Series no conversion needed
        serie = data                                  
    elif isinstance(data, np.ndarray):              # If data is a NumPy array   
        serie = pd.Series(data.flatten())
    elif isinstance(data, (dict, list)):
        serie = pd.Series(data)
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:                      # Also len(data.columns == 1)
            serie = data.iloc[:, 0]
        elif data.shape[1] == 2:                    # Index: first col, Data: 2nd Col
            serie = data.set_index(data.columns[0])[data.columns[1]]
        else:
            raise ValueError("DataFrame must have 1 oer 2 columns. Categories and values for 2 columns cases.")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. "
                    "Supported types: pd.Series, np.ndarray, pd.DataFrame, dict, list, and pd.DataFrame")

    if must_count:
        return serie.value_counts()
    else:
        return serie


# Create a complete frecuency distribution table fron a categorical data
def get_fdt(
        data: Union[pd.Series, np.ndarray, dict, list, pd.DataFrame],
        must_count: Optional[bool] = False,
        pcts: Optional[bool] = True,
        sort: Optional[str] = None,
        plain_relatives: Optional[bool] = False,
        fmt_values: Optional[bool] = False,
) -> pd.DataFrame:
    '''
    OJO, continuar con estos detalles
    sort: None, 'asc', 'desc', 'ix_asc', 'ix_des' para como queremos que sea vea el orden por valores o por indice
    
    
    '''


    fdt = pd.DataFrame(to_serie_with_count(data, must_count=must_count))
    fdt.columns = ['Frequency']
    fdt['Cumulative Frequency'] = fdt['Frequency'].cumsum()
    fdt['Relative Freq. [%]'] = fdt['Frequency'] / fdt['Frequency'].sum() * 100
    fdt['Cumulative Freq. [%]'] = fdt['Relative Freq. [%]'].cumsum()

    if fmt_values:
        fdt = fdt.map(_fmt_value_for_pd)

    return fdt




# state_fdt_us = pd.DataFrame(df['State'].value_counts())
# state_fdt_us['Relative Freq. US-only [%]'] = state_fdt_us['count'] / state_fdt_us['count'].sum() * 100
# state_fdt_us['Cumulative Freq. US-only [%]'] = state_fdt_us['Relative Freq. US-only [%]'].cumsum()
# state_fdt_us


def describeplus(data, decimals=2, miles=',') -> pd.DataFrame:
    ''' Descriptive sats of data'''

    serie = to_serie_with_count(data)          # Convert data to a pandas Series
    
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



## CHARTs Functions

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
    - 'colorbind' <- daltonic, 'viridis', 'plasma', 'inferno', 'magma', 'cividis' <- daltonic, set3, set2
    - 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 
    'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 
    'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 
    'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
    'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
    'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
    'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 
    'gist_gray_r', 'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
    'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r', 'hot', 'hot_r', 
    'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'managua', 'managua_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 
    'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 
    'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
    'twilight_shifted', 'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'winter', 'winter_r'",
    '''
    if n < 6:       # To get a softer range of colors if n is too small           
        n = 6

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


def _validate_categorical_parameters(
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
    scale: Optional[int] = 2,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    label_place: Optional[str] = 'ext',
    palette: Optional[list] = 'colorblind',
    startangle: Optional[float] = -40,
    pct_decimals: Optional[int] = 2,
    label_rotate: Optional[float] = 0,
    legend_loc: Optional[str] = 'best',
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates pie or donut charts for categorical data visualization with customizable labels.

    This function creates a pie or donut chart from categorical data using matplotlib.
    It supports internal or external labels, custom palettes, and formatting options.

    Parameters:
        data (pd.Series, pd.DataFrame, dict, or list): Input data. If a list is provided,
            frequencies are counted automatically.
        scale (int): Chart scaling factor (1 to 6). Default is 2.
        title (str or None): Optional title for the chart.
        kind (str): Type of chart: 'pie' or 'donut'. Default is 'pie'.
        label_place (str): Placement of labels: 'ext' (external) or 'int' (internal).
        palette (list of str or None): List of color hex codes for categories.
        startangle (float): Starting angle in degrees for the first wedge.
        pct_decimals (int): Number of decimal places for percentage labels. Default is 2.
        label_rotate (float): Rotation angle for internal labels. Default is 0.
        legend_loc (str): Position of the legend ('best', 'upper right', etc.). Default is 'best'.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The Figure and Axes objects
            for further customization.

    Raises:
        ValueError: If data contains invalid values, if unsupported types are passed,
            if more than 9 categories are used, or if parameters like scale are out of range.
        TypeError: If data is not one of the accepted types.

    Examples:
        >>> import pandas as pd
        >>> data = pd.Series([25, 30, 20, 15, 10], index=['A', 'B', 'C', 'D', 'E'])
        >>> fig, ax = plt_pie(data, kind='donut', title='Donut Distribution')
        >>> plt.show()

        >>> data_list = ['A', 'B', 'A', 'C', 'B', 'B']
        >>> fig, ax = plt_pie(data_list, kind='pie', label_place='int')
        >>> plt.show()
    """
    # Convert to serie en case of DF
    if isinstance(data, pd.DataFrame):
        data = to_serie_with_count(data)

    _validate_categorical_parameters(data)
    
    # Validate kind parameter
    if kind.lower() not in ['pie', 'donut']:
        raise ValueError(f"Invalid 'kind' parameter: '{kind}'. Must be 'pie' or 'donut'.")
    
    # Validate maximum categories
    if len(data) > 9:
        raise ValueError(f"Data contains {len(data)} categories. "
                        "Maximum allowed is 9 categories.")
    
    # Build graphs size, and fonts size from scale, and validate scale from 1 to 6.
    if scale < 1 or scale > 9:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '6', not '{scale}'.")
    else:
        scale = round(scale)

    multiplier, w_base, h_base  = 1.33333334 ** scale, 4.45, 2.25
    width, high= w_base * multiplier, h_base * multiplier
    label_size = width * 1.25
    title_size = label_size * 1.25

    # Base fig definitions
    figsize = (width, high)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

    # Configure wedge properties for donut  or pie chart
    wedgeprops = {}
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.54, 'edgecolor': 'white', 'linewidth': 1}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 0.5}

    # Define colors
    color_palette = get_colors_list(palette, len(data))

    if label_place == 'ext':

        wedges, texts = ax.pie(data, wedgeprops=wedgeprops, colors=color_palette, startangle=startangle)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        # Build the labels. Annotations and legend in same label (External)
        labels = [
            f"{data.loc[data == value].index[0]}\n{value}\n({round(value / data.sum() * 100, pct_decimals)} %)"
            for value in data.values
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
            
    elif label_place == 'int':
        # I have to change the fontsize for internal annotations
        label_size = label_size * 0.8
        legend_size = label_size * 1.1

        # autopct for internal annotations. A funtion to show both: absolute an pcts.
        format_string = f'%.{pct_decimals}f%%'

        def make_autopct(values, fmt_str):
            value_iterator = iter(values)
            
            def my_autopct(pct):
                absolute_value = next(value_iterator)
                percentage_string = fmt_str % pct
                return f"{absolute_value}\n({percentage_string})"  
            
            return my_autopct
        
        autopct_function = make_autopct(data.values, format_string)
        
        ax.pie(x=data,
            colors=color_palette,
            startangle=startangle,
            autopct=autopct_function,
            wedgeprops=wedgeprops,
            textprops={'size': label_size,
                        'color': 'w',
                        'rotation': label_rotate,
                        'weight': 'bold'})
        
        ax.legend(data.index,
                loc=legend_loc,
                bbox_to_anchor=(1, 0, 0.2, 1),
                prop={'size': legend_size})

    else:
        raise ValueError(f"Invalid labe_place parameter. Must be 'ext' or 'int', not '{label_place}'.")
            
    # Build title
    if not title:
        title = f"Pie/Donut Chart - ({data.name})"
    ax.set_title(title, fontdict={'size': title_size, 'weight': 'bold'})

    return fig, ax


def plt_paretto(
    data: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = 2,
    title: Optional[str] = None,
    palette: Optional[list] = 'colorblind',
    color1: Optional[str] = 'midnightblue',
    color2: Optional[str] = 'darkorange',
    line_size = 4,
    kind: Optional[str] = 'pie',
    label_place: Optional[str] = 'ext',
    startangle: Optional[float] = -40,
    pct_decimals: Optional[int] = 2,
    label_rotate: Optional[float] = 0,
    legend_loc: Optional[str] = 'best'
) -> tuple[plt.Figure, plt.Axes]:

    # Convert to serie en case of DF
    if isinstance(data, pd.DataFrame):
        data = to_serie_with_count(data)

    _validate_categorical_parameters(data)
    
    # # Validate kind parameter
    # if kind.lower() not in ['pie', 'donut']:
    #     raise ValueError(f"Invalid 'kind' parameter: '{kind}'. Must be 'pie' or 'donut'.")
    
    # # Validate maximum categories
    # if len(data) > 9:
    #     raise ValueError(f"Data contains {len(data)} categories. "
    #                     "Maximum allowed is 9 categories.")
    
    # Build graphs size, and fonts size from scale, and validate scale from 1 to 6.
    if scale < 1 or scale > 9:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '6', not '{scale}'.")
    else:
        scale = round(scale)

    multiplier, w_base, h_base  = 1.33333334 ** scale, 4.45, 2.25
    width, high= w_base * multiplier, h_base * multiplier
    label_size = width * 1.25
    title_size = label_size * 1.25

    # define aesthetics for plot - color1 and 2 plus line_size

    # Base fig definitions - create basic bar plot
    fig, ax = plt.subplots(figsize=(width, high), subplot_kw=dict(aspect="equal"))
    bplot = ax.bar(data.index, data.values, color=color1)

    # Add bar labels
    ax.bar_label(bplot,
                fontweight='bold',
                color=color1,
                padding=4)

    # add cumulative percentage line to plot
    ax2 = ax.twinx()        # create another y-axis sharing a common x-axis
    percentage_lim = 100
    ax2.set_ylim(0, percentage_lim)     # make the secondary y scale from 0 to 100

    ax2.plot(state_fdt_us.index,
            state_fdt_us['Cumulative Freq. US-only [%]'],
            color=color2,
            marker="D",
            ms=line_size)

    ax2.yaxis.set_major_formatter(PercentFormatter())

    # Add maeker labels (in percentage) 
    formatted_weights = [f'{x:.0f}%' for x in state_fdt_us['Cumulative Freq. US-only [%]']]  
    for i, txt in enumerate(formatted_weights):
            ax2.annotate(txt,
                        (state_fdt_us.index[i], state_fdt_us['Cumulative Freq. US-only [%]'].iloc[i] - 6),
                        color='orange')    

    # specify axis colors and x-axis rotation
    ax.tick_params(axis='y', colors=color1)
    ax.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', colors=color2)

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_margins.html#sphx-glr-gallery-subplots-axes-and-figures-axes-margins-py
    # ax.margins(y=0.1)
    # ax2.use_sticky_edges = False          # DO NOT work
    # ax2.margins(0.3)                      # DO NOT work

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylim.html#matplotlib.axes.Axes.set_ylim
    ax.set_ylim(0, state_fdt_us['count'].iloc[0] * 1.2 )
    ax2.set_ylim(0, percentage_lim * 1.1)

    #display Pareto chart
    plt.show()





if __name__ == "__main__":

    df = pd.DataFrame({'A': [1, 2, pd.NA, pd.NA, 1],
                       'B': [4.0, pd.NA, pd.NA, 6.1, 4.0],
                       'C': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                       'D': ['x', 'y', pd.NA, 'z', 'x'],
                       'E': ['x', 'y', pd.NA, 'z', 'x']})
    
    print(df)

    df2 = clean_df(df)
    
    print(df2)
        


