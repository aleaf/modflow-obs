"""Tests for the utilities module.
"""
import pandas as pd
from mfobs.modflow import get_perioddata
from mfobs.utils import set_period_start_end_dates


def test_set_period_start_end_dates(test_data_path):
    perioddata_file = test_data_path / 'stress_period_data.csv'
    perioddata = pd.read_csv(perioddata_file)
    set_period_start_end_dates(perioddata)
    starts = pd.to_datetime(perioddata['start_datetime'])
    perlen = perioddata['perlen'].values
    expected_end_dates = starts + pd.to_timedelta(perlen - 1, unit='d')
    # we want to keep the end dates as strings
    assert isinstance(perioddata['end_datetime'].iloc[0], str)
    # end dates should be the last day of the period
    assert (pd.to_datetime(perioddata['end_datetime']) - \
            expected_end_dates).sum().days == 0
    
    # run the function again with correct end dates to verify that they don't change
    set_period_start_end_dates(perioddata)
    assert (pd.to_datetime(perioddata['end_datetime']) - \
            expected_end_dates).sum().days == 0
    
    # test with timesteps
    perioddata = get_perioddata(test_data_path / 'mf2005/br_trans.dis', start_datetime='2000-01-01',
                                end_datetime='2001-02-01', include_timesteps=True)
    set_period_start_end_dates(perioddata)
    # end date times should match the start datetimes 
    # (1 day timesteps that start/end on the same day for the purposes of pandas slicing)
    pd.testing.assert_series_equal(pd.to_datetime(perioddata['start_datetime']), 
                                   pd.to_datetime(perioddata['end_datetime']), 
                                   check_names=False)
