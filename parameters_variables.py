## Parameters and Variables - Grok
##  - Listing and characteristics of function parameters and variable names

#--------------------------------------------------------------------------------------------------------------------------------#
# Generals (valid for parameters and variables):
#--------------------------------------------------------------------------------------------------------------------------------#
#   - n_*: Indicates the number of... what follows. E.g., n_decimals: number of decimal places; n_nulls: number of nulls.
#   *_sep_*: sep indicates 'separator'. E.g., thousands_sep: thousands separator.
#
#--------------------------------------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------------------------------------#
# Parameters:
#   The policy is to use the same parameter name for the same parameter in different functions. These parameters must match
#   the native parameters of Pandas (and Numpy) functions as closely as possible.
#--------------------------------------------------------------------------------------------------------------------------------#
#   - value:
#   - width:
#   - n_decimals:
#   - thousands_sep:
#  Los cuatro anteriores revisar porque por ahí mejor usar la función de jm_utils metida acá como interna
#
#   - data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]
#   - index: Optional[Unionpd.Index]
#   name: Optional[str] = None
#
# 
# Common parameters for categorical charts:
#   - data: Union[pd.Series, pd.DataFrame], | One or two col DF. Case two cols 1se col is index (categories) and 2nd values
#   - value_counts: Optional[bool] = False, | You can plot native values or aggregated ones by categories
#   - scale: Optional[int] = 1,             | All sizes, widths, etc. are scaled from this number (from 1 to 9)
#   - ...