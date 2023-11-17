
from . import _version
__version__ = _version.get_versions()['version']

from mfobs.heads import get_head_obs
from mfobs.obs import get_spatial_differences, get_temporal_differences, get_base_obs
