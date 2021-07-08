"""Tests for the modflow module.
"""
import numpy as np
import pandas as pd
from affine import Affine
import pytest
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
    

@pytest.mark.parametrize('tdis_file,sto_file,start_date,end_date,time_units,expected_time_units',
                         (('badriver/br_trans.dis', None, '2000-01-01', None, None, 'days'),
                          ('badriver/br_trans.dis', None, '2000-01-01', '2020-01-01', 'days', 'days'),
                          ('shellmound/mfsim.tdis', 'shellmound/shellmound.sto', '2000-01-01', None, None, 'days'),
                          ('shellmound/mfsim.tdis', 'shellmound/shellmound.sto', None, None, None, 'days'),
                         ))    
def test_get_perioddata(tdis_file, sto_file, start_date, end_date, 
                        time_units, expected_time_units, test_data_path):
    tdis_file = test_data_path / tdis_file
    if sto_file is not None:
        sto_file = test_data_path / sto_file
    results = get_perioddata(tdis_file, sto_file, start_datetime=start_date, end_datetime=end_date, 
                             model_time_units=time_units)
    elapsed_time = np.cumsum(results['perlen'].values) -1
    elapsed_time[0] = 1
    assert results['start_datetime'].dtype == np.object
    assert results['end_datetime'].dtype == np.object
    expected_end_datetimes = pd.Timestamp(results['start_datetime'][0]) + \
        pd.to_timedelta(elapsed_time -1, unit='d')
    assert all(results['end_datetime'] == expected_end_datetimes.strftime('%Y-%m-%d'))
    assert pd.api.types.is_integer_dtype(results['nstp'])
    
    # explicitly check for time units of days
    # (either entered or read)
    ndays = (pd.Timestamp(results['end_datetime'].values[-1]) - 
             pd.Timestamp(results['start_datetime'].values[-1])).days
    assert ndays == results['perlen'].values[-1] - 1
    
