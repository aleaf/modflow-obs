"""Tests for the utilities module.
"""
import numpy as np
import pandas as pd
from mfobs.modflow import get_perioddata
from mfobs.utils import set_period_start_end_dates
import pytest


def assert_idempotent(perioddata):
    """Running set_period_start_end_dates again shouldn't change the result."""
    rerun = perioddata.copy()
    set_period_start_end_dates(rerun)
    pd.testing.assert_frame_equal(rerun, perioddata)


@pytest.fixture
def br_trans_timesteps(test_data_path):
    """MODFLOW-2005 perioddata with an initial 1-day steady-state period
    followed by 397 1-day timesteps."""
    return get_perioddata(test_data_path / 'mf2005/br_trans.dis', start_datetime='2000-01-01',
                          end_datetime='2001-02-01', include_timesteps=True)


# Scenario tests

def test_set_period_start_end_dates_stress_period_csv(test_data_path):
    """Time column with an initial steady-state period and multi-timestep periods;
    end dates should be the last day of each period."""
    perioddata = pd.read_csv(test_data_path / 'stress_period_data.csv')
    set_period_start_end_dates(perioddata)
    # we want to keep the end dates as strings
    assert isinstance(perioddata['end_datetime'].iloc[0], str)
    starts = pd.to_datetime(perioddata['start_datetime'])
    expected_end_datetimes = starts + pd.to_timedelta(perioddata['perlen'] - 1, unit='D')
    pd.testing.assert_series_equal(pd.to_datetime(perioddata['end_datetime']),
                                   expected_end_datetimes, check_names=False)
    assert_idempotent(perioddata)


def test_set_period_start_end_dates_daily_timesteps(br_trans_timesteps):
    """1-day timesteps start and end on the same day (for the purposes of pandas slicing)."""
    perioddata = br_trans_timesteps
    set_period_start_end_dates(perioddata)
    pd.testing.assert_series_equal(pd.to_datetime(perioddata['start_datetime']),
                                   pd.to_datetime(perioddata['end_datetime']),
                                   check_names=False)
    assert_idempotent(perioddata)


def test_set_period_start_end_dates_truncated_timesteps(br_trans_timesteps):
    """No initial steady-state period, for example if some number of initial
    periods are truncated so as not to generate observations for those.
    The first period's length is its time."""
    perioddata = br_trans_timesteps.iloc[1:].copy()
    perioddata['time'] += 100
    set_period_start_end_dates(perioddata)
    perlen = np.diff(perioddata['time'].values, prepend=0)
    assert perlen[0] == 102
    expected_end_datetimes = pd.to_datetime(perioddata['start_datetime']) +\
        pd.to_timedelta(perlen - 1, unit='D')
    pd.testing.assert_series_equal(pd.to_datetime(perioddata['end_datetime']),
                                   expected_end_datetimes, check_names=False)
    assert_idempotent(perioddata)


def test_set_period_start_end_dates_no_time_column_monthly():
    """A time column is constructed from monthly start and (inclusive) end dates."""
    perioddata = pd.DataFrame({
        'start_datetime': pd.date_range('2020-01-01', '2020-12-31', freq='MS'),
        'end_datetime': pd.date_range('2020-01-01', '2020-12-31', freq='ME')
    })
    expected_end_datetimes = perioddata['end_datetime'].dt.strftime('%Y-%m-%d')
    set_period_start_end_dates(perioddata)
    pd.testing.assert_series_equal(perioddata['end_datetime'], expected_end_datetimes)
    # 2020 is a leap year; inclusive end dates mean the last day is counted
    assert perioddata['time'].values[-1] == 366
    assert perioddata['time'].diff().values[-1] == 31
    assert_idempotent(perioddata)


# Expected results for specific rules

@pytest.mark.parametrize('input_perioddata,expected', (
    # Single day stress periods, transient or steady-state
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-01-02'],
             'end_datetime': ['2020-01-01', '2020-01-01', '2020-01-02'],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-01-02'],
             'end_datetime': ['2020-01-01', '2020-01-01', '2020-01-02'],
             'steady': [True, False, False],
             'time': [1., 2., 3.]}),
        id='single-day periods'),
    # End date overlapping the next start is moved to the day before the next start
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2020-01-01', '2020-02-01', '2020-02-29'],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-02-29'],
             'steady': [True, False, False],
             'time': [1., 32., 61.]}),
        id='end overlaps next start'),
    # Partially overlapping transient periods (example in the docstring)
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-03-01', '2020-03-15'],
             'end_datetime': ['2020-04-01', '2020-04-15']}),
        pd.DataFrame(
            {'start_datetime': ['2020-03-01', '2020-03-15'],
             'end_datetime': ['2020-03-14', '2020-04-15'],
             'time': [14., 46.]}),
        id='partially overlapping periods'),
    # Gaps between transient periods are removed
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-02-01', '2020-05-01', '2020-08-31']}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-02-29', '2020-05-31', '2020-08-31'],
             'time': [60., 152., 244.]}),
        id='transient gaps removed'),
    # Already consistent dates are unchanged (and leap day 2/29 is retained)
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-02-29', '2020-05-31', '2020-08-31']}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-02-29', '2020-05-31', '2020-08-31'],
             'time': [60., 152., 244.]}),
        id='consistent dates unchanged'),
    # Without a time column, steady-state lengths are from their own dates,
    # so a gap following a steady-state period is preserved
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-04-01'],
             'end_datetime': ['2020-02-01', '2020-03-31', '2020-04-30'],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-04-01'],
             'end_datetime': ['2020-02-01', '2020-03-31', '2020-04-30'],
             'steady': [True, False, False],
             'time': [32., 63., 93.]}),
        id='no time, steady length from own dates'),
    # Transient period followed by a steady-state period:
    # length from its own dates; end date not aligned with the steady-state period
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-01-31', '2020-03-01', '2020-06-30'],
             'steady': [False, True, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-01-31', '2020-03-01', '2020-06-30'],
             'steady': [False, True, False],
             'time': [31., 32., 62.]}),
        id='no time, transient before steady'),
    # Time column is definitive; second period is only 30 days,
    # because the length 1 steady-state period doesn't count
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-04-01'],
             'time': [1, 31, 60],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-01-31'],
             'end_datetime': ['2020-01-01', '2020-01-30', '2020-02-28'],
             'time': [1, 31, 60],
             'steady': [True, False, False]}),
        id='time adjusts transient starts and ends'),
    # Transient end dates inconsistent with time are ignored
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2020-01-01', '2020-01-15', '2020-02-29'],
             'time': [1, 32, 60],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-02-28'],
             'time': [1, 32, 60],
             'steady': [True, False, False]}),
        id='time overrides transient end dates'),
    # Gaps following steady-state periods are preserved;
    # the transient start following a steady-state period is kept
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-01'],
             'end_datetime': ['2020-02-01', '2020-05-01', '2020-08-31'],
             'steady': [True, False, False],
             'time': [32., 152., 244.]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-03-01', '2020-06-29'],
             'end_datetime': ['2020-02-01', '2020-06-28', '2020-09-28'],
             'steady': [True, False, False],
             'time': [32., 152., 244.]}),
        id='time, steady gap preserved'),
    # Steady-state end dates are kept as-is when supplied, even if inconsistent with time
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2019-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2019-12-31', '2020-01-31', '2020-02-29'],
             'time': [1, 32, 61],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2019-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2019-12-31', '2020-01-31', '2020-02-29'],
             'time': [1, 32, 61],
             'steady': [True, False, False]}),
        id='time, steady end kept'),
    # No end_datetime column; end dates (including steady-state) computed from time
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'time': [1, 32, 61],
             'steady': [True, False, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'time': [1, 32, 61],
             'steady': [True, False, False],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-02-29']}),
        id='time, no end_datetime column'),
    # Index named 'time' is copied to a 'time' column (index is retained)
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'steady': [True, False, False]},
            index=pd.Index([1., 32., 61.], name='time')),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'steady': [True, False, False],
             'time': [1., 32., 61.],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-02-29']},
            index=pd.Index([1., 32., 61.], name='time')),
        id='time index'),
    # Rows are sorted by time (original index labels are kept)
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-02-01', '2020-01-01', '2020-01-01'],
             'time': [61, 1, 32],
             'steady': [False, True, False]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'time': [1, 32, 61],
             'steady': [True, False, False],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-02-29']},
            index=[1, 2, 0]),
        id='sorted by time'),
    # Sub-daily lengths: end dates are 1 second before the next start,
    # and the whole table is formatted with times
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01'] * 3,
             'time': [0.5, 1.0, 2.25]}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01 00:00:00', '2020-01-01 12:00:00', '2020-01-02 00:00:00'],
             'time': [0.5, 1.0, 2.25],
             'end_datetime': ['2020-01-01 11:59:59', '2020-01-01 23:59:59', '2020-01-03 05:59:59']}),
        id='sub-daily lengths'),
    # Input dates that aren't at midnight are also treated as sub-daily,
    # even if the lengths are whole days
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01 06:00:00', '2020-01-02 06:00:00'],
             'end_datetime': ['2020-01-01 12:00:00', '2020-01-03 05:59:59']}),
        pd.DataFrame(
            {'start_datetime': ['2020-01-01 06:00:00', '2020-01-02 06:00:00'],
             'end_datetime': ['2020-01-02 05:59:59', '2020-01-03 05:59:59'],
             'time': [1., 2.]}),
        id='sub-daily input dates'),
))
def test_set_period_start_end_dates_cases(input_perioddata, expected):
    set_period_start_end_dates(input_perioddata)
    pd.testing.assert_frame_equal(input_perioddata, expected)
    assert_idempotent(input_perioddata)


@pytest.mark.parametrize('input_perioddata,match', (
    # Zero-length transient periods (identical start dates) aren't allowed
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01', '2020-01-01', '2020-02-01'],
             'end_datetime': ['2020-01-01', '2020-01-31', '2020-02-29'],
             'steady': [False, False, False]}),
        'Overlapping', id='identical transient starts'),
    # Zero-length periods in the time column (repeated times or first time <= 0)
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01'] * 3,
             'time': [10, 10, 20]}),
        'Zero-length', id='zero-length time'),
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01'] * 2,
             'time': [10, 10]}),
        'Zero-length', id='zero-length last period in time'),
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01'] * 3,
             'time': [1, 1, 32],
             'steady': [True, True, False]}),
        'Zero-length', id='zero-length steady-state period in time'),
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-01-01'] * 2,
             'time': [0, 31]}),
        'Zero-length', id='first time zero'),
    # Transient start following a steady-state period
    # before the previous transient start
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-03-01', '2020-04-01', '2020-02-01'],
             'end_datetime': ['2020-03-31', '2020-04-01', '2020-02-28'],
             'steady': [False, True, False]}),
        'Overlapping', id='transient start after steady before previous start'),
    # Neither a time nor an end_datetime column
    pytest.param(
        pd.DataFrame({'start_datetime': ['2020-01-01']}),
        "must have a 'time' column", id='no time or end_datetime'),
    # End date before the start date
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2020-02-01', '2020-03-01'],
             'end_datetime': ['2020-01-01', '2020-03-31'],
             'steady': [True, False]}),
        'before start_datetime', id='end before start'),
    # 2021-02-29 doesn't exist
    pytest.param(
        pd.DataFrame(
            {'start_datetime': ['2021-01-01', '2021-03-01', '2021-06-01'],
             'end_datetime': ['2021-02-29', '2021-05-31', '2021-08-31']}),
        None, id='invalid leap day'),
))
def test_set_period_start_end_dates_raises(input_perioddata, match):
    with pytest.raises(ValueError, match=match):
        set_period_start_end_dates(input_perioddata)
