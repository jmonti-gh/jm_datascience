"""
jm_datascience - Utilities for DS tasks

Utilities for pandas, numpy, matplotlib, etc.
"""

# Metadatos del paquete principal
__version__ = "0.1.0"
__description__ = "Utilities I use frequently - Several modules"
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"

# # Importar módulos principales
# from . import jm_pdaccessor
# from . import jm_numpy
# from . import jm_pandas

# Definir qué se exporta con "from jm_utils import *"
__all__ = [
    'jm_pdaccessor',
    'jm_numpy', 
    'jm_pandas'
]

# Información del paquete para acceso programático
def get_version():
    """Retorna la versión del paquete"""
    return __version__

def get_info():
    """Retorna información completa del paquete"""
    return {
        'name': 'jm_datascience',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'status': __status__
    }