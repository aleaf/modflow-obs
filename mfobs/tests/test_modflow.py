"""Tests for the modflow module.
"""
import numpy as np
import pandas as pd
from affine import Affine
from mfobs.modflow import get_ij, get_perioddata


def test_get_ij():
    
    # upper left corner
    xul, yul = 434955., 1342785.
    spacing = 500.
    transform = Affine(spacing, 0.0, xul,
                       0.0, -spacing, yul)
    
    # get the i, j location for a point just past the first cell center
    x = xul + spacing/2 + 1
    y = yul - spacing/2 - 1
    i, j = get_ij(transform, x, y)
    assert (i, j) == (0, 0)
    
    # test the upper left corner
    i, j = get_ij(transform, xul, yul)
    assert (i, j) == (0, 0)
    
    # test a lower left corner
    expected_i, expected_j = 1000, 1000
    i, j = get_ij(transform, 
                  xul + spacing * expected_i, 
                  yul - spacing * expected_j
                  )
    assert (i, j) == (expected_i, expected_j)  
    
    
def test_get_perioddata(test_data_path):
    tdis_file = test_data_path / 'shellmound/mfsim.tdis'
    sto_file = test_data_path / 'shellmound/shellmound.sto'
    results = get_perioddata(tdis_file, sto_file)
    elapsed_time = np.cumsum(results['perlen'].values) -1
    elapsed_time[0] = 1
    assert results['start_datetime'].dtype == np.object
    assert results['end_datetime'].dtype == np.object
    expected_end_datetimes = pd.Timestamp(results['start_datetime'][0]) + \
        pd.to_timedelta(elapsed_time -1, unit='d')
    assert all(results['end_datetime'] == expected_end_datetimes.strftime('%Y-%m-%d'))
