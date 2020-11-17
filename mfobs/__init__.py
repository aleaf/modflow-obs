
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from mfobs.heads import get_head_obs
from mfobs.obs import get_spatial_differences, get_temporal_differences
from mfobs.swflows import get_flux_obs
