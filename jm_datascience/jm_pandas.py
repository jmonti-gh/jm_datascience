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


## Standard Libs
from typing import Union, Optional, Tuple, Dict, Any
import warnings

# Third-Party Libs
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


def to_serie(data: Union[pd.Series, np.ndarray, dict, list, pd.DataFrame]) -> pd.Series:
    '''
    Convert df of one or two columns, or numpy-array, or list, or dict to pd.Serie.
    
    Param -> data: df of one or two columns, or numpy-array, or list, or dict to pd.Serie.
    Returns: pd.Series
    '''

    if isinstance(data, pd.Series):                 # If serie is already a Series
        pdserie = data                              # No conversion needed      
    elif isinstance(data, np.ndarray):              # If data is a NumPy array   
        pdserie = pd.Series(data.flatten())
    elif isinstance(data, dict):
        pdserie = pd.Series(data)
    elif isinstance(data, list):                    # If data is a list
        pdserie = pd.Series(data)
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:                      # Also len(data.columns == 1)
            pdserie = data                          # Also pdserie = serie.iloc[:, 0]
        else:
            raise ValueError("DataFrame must exactly have 1 column: Categories -> index")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. "
                    "Supported types: pd.Series, np.ndarray, pd.DataFrame, dict, list")

    return pdserie


def to_categorical_serie(data: Union[pd.Series, np.ndarray, dict, list, pd.DataFrame]) -> pd.Series:
    '''
    Build a categorical (base of Freq. Dist. Table) of input
    '''
    if isinstance(data, pd.Series):                 # If serie is already a Series
        cat_serie = data                            # No conversion needed      
    elif isinstance(data, np.ndarray):              # If data is a NumPy array   
        cat_serie = pd.Series(data.flatten()).value_counts()
    elif isinstance(data, dict):
        cat_serie = pd.Series(data)
    elif isinstance(data, list):                    # If data is a list
        cat_serie = pd.Series(data).value_counts()
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:                      # Also len(data.columns == 1)
            cat_serie = data.value_counts()         # Also cat_serie = serie.iloc[:, 0]
        elif data.shape[1] == 2:                    # Index: first col, Data: 2nd Col
            cat_serie = data.set_index(data.columns[0])[data.columns[1]]
        else:
            raise ValueError("DataFrame must have 1 oer 2 columns. Categories and values for 2 columns cases.")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. "
                    "Supported types: pd.Series, np.ndarray, pd.DataFrame, dict, list")

    return cat_serie


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

def get_colorblind_palette():
    """
    Retorna una lista de colores (hexadecimales) amigables para personas
    con daltonismo, equivalentes a sns.color_palette('colorblind').
    """
    return [
        '#0173B2',  # Azul
        '#DE8F05',  # Naranja
        '#029E73',  # Verde
        '#D55E00',  # Bermellón
        '#CC78BC',  # Violeta rojizo
        '#CA9161',  # Marrón amarillento
        '#FBAFE4',  # Rosa
        '#949494',  # Gris
        '#ECE133',  # Amarillo
        '#56B4E9'   # Azul cielo
    ]


def get_colors(palette):
    '''
    Return a valid matplotlib palette list
    - 'colorbind', 'viridis', 'plasma', 'inferno', 'magma', 'cividis' <- daltonic
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
    if palette == 'colorblind':
        colors_palette = get_colorblind_palette()
    else:
        cmap = plt.get_cmap(palette, 10)                 # Use palette colormap
        colors_palette = [cmap(i) for i in range(10)]    # Get colors from the colormap

    return colors_palette


def is_valid_for_pie(data):
    if len(data) > 9:
        raise ValueError(f"Data contains {len(data)} categories. "
                        "Maximum allowed is 9 categories.")


def plt_pie(
        data,
        scale = 2,
        kind = 'pie',
        title = '',
        palette = 'colorblind',
        figsize = (),
        pct_decimals = 2,
        label_size = 0,
        title_size = 0
):
    ''' Donut chart w/ external legend/label absolute/pct values '''

    is_valid_for_pie(data)
    
    # Build graphs size, and fonts size from scale
    if scale < 1 or scale > 6:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '6', not '{scale}'.")
    else:
        scale = round(scale)

    multiplier, w_base, h_base  = 1.33333334 ** scale, 4.45, 2.25
    width, high= w_base * multiplier, h_base * multiplier
    label_size = width * 1.25
    title_size = label_size * 1.25

    # Base fig definitions
    if not figsize:
        figsize = (width, high)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

    # Configure wedge properties for donut  or pie chart
    wedgeprops = {}
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.54, 'edgecolor': 'white', 'linewidth': 1}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 0.5}

    # Define colors
    color_palette = get_colors(palette)

    wedges, texts = ax.pie(data, wedgeprops=wedgeprops, colors=color_palette, startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

    # Build the labels. 1st. sum to pcts calc. 
    labels = [
        f"{data.loc[data == value].index[0]}\n{value}\n({round(value / data.sum(), pct_decimals)} %)"
        for value in data.values
    ]
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, fontsize=label_size, **kw)

    # Build title
    if not title:
        title = f"Donut Chart - ({data.name})"
   
    ax.set_title(title, fontdict={'size': title_size})

    return fig, ax


def plt_pie_1(
        data,
        figsize = (6, 6),
        palette = 'colorblind',
        startangle = 0,
        rotate = 0,
        label_size = 15,
        title = 'Pie Chart',
        title_size = 20,
        autopct = '%.2f%%',
        legend_loc = 'best',
        legend_size = 15,

    ):
    """
    Create a pie chart from a specified column in a DataFrame.
    
    Parameters:
    - data: df[col], pd.Series, list, tuple, or array-like object to create the pie chart from.
    - title (str): Title of the pie chart.
    - figsize (float, float): Size of the figure.
    - autopct (str_format): Format for displaying percentages.
    """

    # Validate maximum categories
    if len(data) > 9:
        raise ValueError(f"Data contains {len(data)} categories. "
                        "Maximum allowed is 9 categories.")

    fig, ax = plt.subplots(figsize=figsize)

    # Colors - colors_palette
    if palette == 'colorblind':
        colors_palette = get_colorblind_palette()
    else:
        cmap = plt.get_cmap(palette, len(data))                 # Use palette colormap
        colors_palette = [cmap(i) for i in range(len(data))]    # Get colors from the colormap

    # autopct
    autopct_function = None

    def make_autopct(values, format_string):                    # A closure: a function that return a new function
        value_iterator = iter(values)                           # 1. Mk an iterator of the original values      
        
        def my_autopct(pct):
            absolute_value = next(value_iterator)               # 2. Get next value from iterator
            percentage_string = format_string % pct             # 3. Format percentage using autopct. El operador '%' aplica el formato al valor 'pct'.
            return f"{absolute_value} ({percentage_string})"     
        
        return my_autopct
    
    autopct_function = make_autopct(data.values, autopct)
    
    ax.pie(x=data,
           colors=colors_palette,
           startangle=startangle,
           autopct=autopct_function,
           textprops={'size': label_size,
                      'color': 'w',
                      'rotation': rotate,
                      'weight': 'bold'})

    ax.set_title(title, fontdict={'size': title_size, 'weight': 'bold'})
    
    ax.legend(data.index,
              loc=legend_loc,
              bbox_to_anchor=(1, 0, 0.2, 1),
              prop={'size': legend_size})

    return fig, ax


# Aux Funct. to Create external horizontal labels with connection lines.
def _create_external_labels(
    ax: plt.Axes, 
    wedges: list, 
    categories: list, 
    values: np.ndarray, 
    percentages: np.ndarray,
    label_size: int
) -> None:
    """
    Create external horizontal labels with connection lines.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object.
    wedges : list
        List of wedge objects from pie chart.
    categories : list
        List of category names.
    values : np.ndarray
        Array of frequency values.
    percentages : np.ndarray
        Array of percentage values.
    """
    # Calculate label positions
    label_distance = 1.3
    connection_distance = 1.15
    
    for i, (wedge, category, value, percentage) in enumerate(zip(wedges, categories, values, percentages)):
        # Get the angle of the wedge center
        angle = (wedge.theta1 + wedge.theta2) / 2
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate connection point on wedge edge
        connection_x = connection_distance * np.cos(angle_rad)
        connection_y = connection_distance * np.sin(angle_rad)
        
        # Calculate label position
        label_x = label_distance * np.cos(angle_rad)
        label_y = label_distance * np.sin(angle_rad)
        
        # Determine horizontal alignment based on position
        ha = 'left' if label_x > 0 else 'right'
        
        # Adjust label position for better horizontal alignment
        if ha == 'left':
            label_x += 0.1
        else:
            label_x -= 0.1
        
        # Create label text
        label_text = f'{category}\n{int(value)} ({percentage}%)'
        
        # Add connection line
        ax.annotate('', xy=(connection_x, connection_y), xytext=(label_x, label_y),
                   arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.7))
        
        # Add label text
        ax.text(label_x, label_y, label_text, 
               horizontalalignment=ha, verticalalignment='center',
               fontsize=label_size, fontweight='normal',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='lightgray', alpha=0.8))


def plt_pie_2(
    data: Union[pd.Series, pd.DataFrame, dict, list],
    kind: str = 'pie',
    title: Optional[str] = None,
    title_size: Optional[int] = 20,
    label_size: Optional[int] = 15,
    decimals: Optional[int] = 2,
    figsize: Tuple[int, int] = (9, 7),
    colors: Optional[list] = None,
    startangle: float = 0,
    explode: Optional[list] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a pie or donut chart with external horizontal labels.
    
    This function generates publication-ready categorical data visualizations with
    external labels showing category names, frequencies, and percentages. Labels
    are connected to their corresponding segments via connection lines.
    
    Parameters:
    - data : Union[pd.Series, pd.DataFrame, dict, list] - Input categorical data. Can be:
        - pd.Series with category names as index and values as frequencies
        - pd.DataFrame with two columns (categories and values)
        - dict with categories as keys and frequencies as values
        - list of category names (frequencies will be counted)
    
    - kind : str, default 'pie' - Type of chart to create. Options:
        - 'pie': Traditional pie chart
        - 'donut': Donut chart with hollow center
    
    - title : str, optional
        Chart title. If None, no title will be displayed.
    
    - figsize : Tuple[int, int],
        Figure size in inches (width, height).
    
    - colors : list, optional
        List of colors for chart segments. If None, uses matplotlib default colors.
    
    - startangle : float, default 90
        Angle at which the first wedge starts (in degrees).
    
    - explode : list, optional
        List of floats specifying how much to separate each wedge from center.
        Must have same length as number of categories.
    
    **kwargs
        Additional keyword arguments passed to matplotlib's pie function.
    
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects for further customization.
    
    Raises
    ------
    ValueError
        If data contains more than 8 categories, if 'kind' parameter is invalid,
        if data is empty, or if explode list length doesn't match data length.
    
    TypeError
        If data type is not supported.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([25, 30, 20, 15, 10], 
    ...                  index=['A', 'B', 'C', 'D', 'E'])
    >>> fig, ax = create_categorical_chart(data, kind='pie', 
    ...                                   title='Sales by Category')
    >>> plt.show()
    
    >>> # Create donut chart with custom colors
    >>> colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    >>> fig, ax = create_categorical_chart(data, kind='donut', 
    ...                                   colors=colors)
    >>> plt.show()
    """
    
    # Validate and normalize input data, not empty, positive numbers, and sort for better visualization
    processed_data = to_categorical_serie(data)
    
    if len(processed_data) == 0:
        raise ValueError("Input data is empty.")
    
    if not all(isinstance(val, (np.integer, np.floating)) and val > 0 for val in processed_data.values):
        raise ValueError("All data values must be positive numbers.")
    
    processed_data = processed_data.sort_values(ascending=False)
    
    # Validate kind parameter
    if kind.lower() not in ['pie', 'donut']:
        raise ValueError(f"Invalid 'kind' parameter: '{kind}'. Must be 'pie' or 'donut'.")
    
    # Validate maximum categories
    if len(processed_data) > 9:
        raise ValueError(f"Data contains {len(processed_data)} categories. "
                        "Maximum allowed is 9 categories.")
    
    # Validate explode parameter
    if explode is not None and len(explode) != len(processed_data):
        raise ValueError(f"Length of 'explode' ({len(explode)}) must match "
                        f"number of categories ({len(processed_data)}).")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    categories = processed_data.index.tolist()
    values = processed_data.values
    total = values.sum()
    percentages = (values / total * 100).round(decimals)
    
    # Set up colors
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    elif len(colors) < len(categories):
        warnings.warn(f"Not enough colors provided ({len(colors)}). "
                     f"Need {len(categories)} colors. Using default colors.")
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # Configure wedge properties for donut chart
    wedgeprops = {}
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.5, 'edgecolor': 'white', 'linewidth': 2}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 1}

    
    # Create the pie/donut chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # We'll create custom external labels
        colors=colors,
        startangle=startangle,
        explode=explode,
        wedgeprops=wedgeprops,
        autopct='',  # We'll create custom percentage labels
        pctdistance=0.85,
        labeldistance=1.1,
        **kwargs
    )
    
    # Create external labels with connection lines
    _create_external_labels(ax, wedges, categories, values, percentages, label_size)
    
    # Set title
    if not title:
        title = f"{kind.title()} Chart"
        
    ax.set_title(
        title,
        # loc='left',
        fontdict={'size': title_size, 'weight': 'bold'},
        x=0.0,                                      # Extremo derecho (1.0 = 100% del ancho del eje)
        y=1.02,                                     # Ligeramente por encima del área del gráfico
    )
    
    # Ensure equal aspect ratio for circular chart
    ax.axis('equal')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
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
        


