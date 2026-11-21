"""Deprecated; the heads module has been renamed to :mod:`mfobs.wells`.
"""
import warnings
from mfobs.wells import *  # noqa: F401,F403
from mfobs.wells import get_head_obs, get_well_obs  # noqa: F401

warnings.warn('mfobs.heads has been renamed to mfobs.wells',
              FutureWarning, stacklevel=2)
