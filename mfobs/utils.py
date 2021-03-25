"""
Miscellaneous utilities
"""
import pandas as pd


def fill_nats(df, perioddata):
    """Fill in NaT (not a time) values with
    corresponding date for that stress period.

    Parameters
    ----------
    df : DataFrame
        Observation data. Must have 'datetime' column
        with date and 'per' column with stress period.
    perioddata : DataFrame
        Perioddata table produced by modflow-setup. Must have
        'per' column and 'start_datetime' column.

    Returns
    -------
    Operates on perioddata in-place.
    """
    period_start_datetimes = pd.to_datetime(perioddata['start_datetime'])
    start_datetimes = dict(zip(perioddata['per'], period_start_datetimes))
    datetime = [start_datetimes[per] if pd.isnull(dt) else dt
                for per, dt in zip(df['per'], df['datetime'])]
    df['datetime'] = datetime


def set_period_start_end_dates(perioddata):
    """Ensure that the start and end dates for consecutive periods are consistent 
    but don't overlap. For example given two consecutive periods start dates of 
    3/1/2020 and 4/1/2020, we want the end dates should be 3/31/2020 and 4/30/2020,
    so that a selection in pandas with :meth:`pandas.DataFrame.loc` doesn't include
    values from the next period.

    Parameters
    ----------
    perioddata : DataFrame
        Perioddata table produced by modflow-setup. Must have
        'per', 'start_datetime', and 'perlen' columns.

    Returns
    -------
    Operates on perioddata in-place.

    Notes
    -----
    Assumes time units of days.
    """
    perioddata.sort_values(by='per', inplace=True)
    start_datetimes = pd.to_datetime(perioddata.start_datetime.values)
    perlen = perioddata['perlen'].values
    new_end_datetimes = start_datetimes + pd.to_timedelta(perlen - 1, unit='d')
    perioddata['end_datetime'] = new_end_datetimes.strftime('%Y-%m-%d')
