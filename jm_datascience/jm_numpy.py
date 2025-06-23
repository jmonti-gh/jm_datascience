"""
jm_numpy
"""

__version__ = "0.1.0"
__description__ = "Utilities I use frequently - Several modules"
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"


import numpy as np
## Claude

def array_stats(arr):
    """Estadísticas básicas de un array numpy"""
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'shape': arr.shape
    }

def normalize_array(arr):
    """Normaliza un array numpy"""
    return (arr - np.mean(arr)) / np.std(arr)


if __name__ == "__main__":
    pass


        


