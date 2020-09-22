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
    """Ensure that the start and end dates for each period don't overlap.
    For example given two consecutive periods start dates of 3/1/2020 and 4/1/2020,
    the end dates should be 3/31/2020 and 4/30/2020. If the end dates were 4/1/2020
    and 5/1/2020, the periods would overlap with the first day of the next period,
    due to the behavior of :meth:`pandas.DataFrame.loc`

    Parameters
    ----------
    perioddata : DataFrame
        Perioddata table produced by modflow-setup. Must have
        'per' column and 'start_datetime' column.

    Returns
    -------
    Operates on perioddata in-place.

    Notes
    -----
    Assumes time units of days.
    """
    perioddata.sort_values(by='per', inplace=True)
    start_datetimes = pd.to_datetime(perioddata.start_datetime.values)
    end_datetimes = pd.to_datetime(perioddata.end_datetime.values)
    new_end_datetimes = []
    for i, start in enumerate(start_datetimes[:-1]):
        end = end_datetimes[i]
        next_start = start_datetimes[i+1]
        if end >= start and end < next_start:
            pass
        elif next_start - pd.Timedelta(1, unit='d') >= start:
            end = next_start - pd.Timedelta(1, unit='d')
        else:
            pass
        new_end_datetimes.append(end)
    new_end_datetimes.append(end_datetimes[-1])
