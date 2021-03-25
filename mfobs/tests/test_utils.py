"""Tests for the utilities module.
"""
import pandas as pd
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