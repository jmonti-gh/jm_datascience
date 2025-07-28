## Parameters and Variables - Grok
##  - Listing and characteristics of function parameters and variable names

#-----------------------------------------------------------------------------------------------------------------------------------------------------#
# Generals (valid for parameters and variables):
#-----------------------------------------------------------------------------------------------------------------------------------------------------#
#   - n_*: Indicates the number of... what follows. E.g., n_decimals: number of decimal places; n_nulls: number of nulls.
#   *_sep_*: sep indicates 'separator'. E.g., thousands_sep: thousands separator.
#   - IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]. | Element type for Series or DF index.
#       - # IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]
#   - *pcts*: Reference percentages. E.g., include_pcts: Indicates whether to include percentages in the output.
#   - na_*: Reference NaNs (np.nan, pd.NA). E.g., na_position
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------------------------------------#
# Parameters by Functions
#   The policy is to use the same parameter name for the same parameter in different functions. These parameters must match
#   the native parameters of Pandas (and Numpy) functions as closely as possible.
#-----------------------------------------------------------------------------------------------------------------------------------------------------#
# _fmt_value_for_pd()
#   - value: Indicates a value, which can be a number, string, etc. It is used in functions that require a single value.
#   - width: Indicates the width of something, such as a bar in a chart.
#   - n_decimals:
#   - thousands_sep:
#  Los cuatro anteriores revisar porque por ahí mejor usar la función de jm_utils metida acá como interna
#
# _validate_numeric_series()
#   - pd_data: Union[pd.Series, pd.DataFrame],                  | Only pandas data: Pandas Series or DataFrame
#   - positive: Optional[bool] = True                           | Indicates whether to validate that values are positive.
#
# to_series()
#   - data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]     | Data to be converted to a Series or DataFrame.
#        - Tuples are not included, as a list of args functions as such and can be confusing.
#        - The 'sets' are previously converted to lists.
#        - A two-column data frame is converted to a series by taking the first column as the index and the second as the values.
#   - index: Optional[Union[pd.Index, Sequence[IndexElement]]] = None       | Series or DataFrame valid index (e.g., pd.Index, list, np.ndarray, etc.).
#        - If not provided, the index is automatically generated.
#   - name: Optional[str] = None                                            | Series or DataFrame name (and others like columns, etc. in other functions).
#
# get_fdt()
#   - data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]     | Data to be converted to a Series or DataFrame.
#        - Tuples are not included, as a list of args functions as such and can be confusing.
#        - The 'sets' are previously converted to lists.
#   - value_counts: Optional[bool] = False,                     | Indicates whether to use the native values or aggregated ones by categories.
#   - dropna: Optional[bool] = True,                            | Indicates whether to drop NaN values.
#   - na_position: Optional[str] = 'last',                      | Position of NaN values in the Series or DataFrame. Can be: 'first', 'last' or 'value'.
#   - include_pcts: Optional[bool] = True,                      | Indicates whether to include percentages in the output.
#   - include_plain_relatives: Optional[bool] = True,           | Indicates whether to include plain relatives (without percentages) in the output.
#   - fmt_values: Optional[bool] = False,                       | Indicates whether to format values.
#   - order: Optional[str] = 'desc',                            | Sort order for the Series or DataFrame. Can be: 'asc', 'desc', 'ix_asc', 'ix_desc', or None.
#   - na_aside: Optional[bool] = True                           | NaN values aside in the Series or DataFrame calculations.
#   - index_name: Optional[str] = None                          | Set custom index name
#
# get_colors_list(
#   - palette: str,                         | 
#   - n_items: Optional[int] = 10           |
#
#   - palette: str                                              | Color palette. Can be: 'colorblind', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'set2', 'set3', or any valid matplotlib colormap.
#   - n_items: Optional[int] = 10                               | Number of items/elements of a sequence/colletcion.
#
#   - pd_data: Union[pd.Series, pd.DataFrame],                  | Only pandas data: Pandas Series or DataFrame
#   - positive: Optional[bool] = True                           | Indicates whether to validate that all values are positive.
#
#     value_counts: Optional[bool] = False,                     | Indicates whether to use the native values or aggregated ones by categories. E.g., for pie charts.
#     sort: Optional[bool] = True,
#     dropna: Optional[bool] = True,
#     scale: Optional[int] = 1,
#     figsize: Optional[tuple[float, float]] = None,
#     title: Optional[str] = None,
#     kind: Optional[str] = 'pie',
#     label_place: Optional[str] = 'ext',
#     palette: Optional[list] = 'colorblind',
#     startangle: Optional[float] = -40,
#     pct_decimals: Optional[int] = 1,
#     label_rotate: Optional[float] = 0,
#     legend_loc: Optional[str] = 'best',
# #
#
# Common parameters for categorical charts:
#   - data: Union[pd.Series, pd.DataFrame], | One or two col DF. Case two cols 1se col is index (categories) and 2nd values
#   - value_counts: Optional[bool] = False, | You can plot native values or aggregated ones by categories
#   - scale: Optional[int] = 1,             | All sizes, widths, etc. are scaled from this number (from 1 to 9)
#   - ...