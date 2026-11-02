"""
Miscellaneous utilities
"""
import inspect
import pprint
import numpy as np
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
    """Fix potentially malformed perioddata
    
     - Ensure that the start and end dates for consecutive transient periods or timesteps are consistent 
       but don't overlap. For example given consecutive periods start dates of 
       3/1/2020, 4/1/2020, 5/1/2020, we want the end dates of the first two periods to be 
       3/31/2020 and 4/30/2020, so that a selection in pandas with :meth:`pandas.DataFrame.loc` doesn't include
       values from the next period. In other words, the end dates are inclusive. An end date of 4/30/2020
       includes all of that day up to midnight, but not into 5/1/2020.
     - Overlapping start_datetimes, defined as transient start_datetimes that are less than or equal to the
       previous start_datetime (e.g., two transient periods both starting on 1/1/2020), raise an error,
       regardless of whether an end_datetime column is supplied.
     - Gaps between transient periods are closed by moving the earlier end date to the day before the next start.
     - If successive transient stress periods overlap
       (e.g., 3/1/2020 -> 4/1/2020 and 3/15/2020 -> 4/15/2020), set the overlapping end
       dates to a day before the next start date.
     - If an (elapsed) 'time' column is supplied, treat that as definitive, and adjust transient start_
       and end_datetimes accordingly. Values in the time column are assumed to be model (MODFLOW)
       time at the end of the current stress period or timestep. MODFLOW time includes time lengths assigned
       to steady-state stress periods. If perioddata has an index named 'time' instead of a 'time' column,
       the index is copied to a 'time' column and treated as such.

       - Period lengths are the differences between successive times (the first period's length is its time).
         No period (transient or steady-state) can have zero length; repeated times, or a first time <= 0,
         raise an error before any other processing.
       - In determining start_datetime(s), preceding steady-state periods are not considered;
         the input start_datetime for any transient period following a steady-state period is always used.
       - Other transient start_datetimes are computed as the previous transient start plus the
         previous transient period's length.
       - Only the first input start date, steady-state period start dates, and transient start dates following
         steady-state periods are kept. Input end dates for transient periods are ignored.
       - Transient end_datetimes are determined from the start_datetime + period or timestep length - 1 day.

     - Otherwise, if no 'time' column is supplied, a time column is constructed from the start and end dates.

       - Period or timestep lengths are determined as the difference between successive start_datetimes;
         time is the cumulative sum of period lengths. The length of the last timestep or period, and of
         any transient period followed by a steady-state period, is its end_datetime - start_datetime + 1 day
         (consistent with inclusive end dates).
       - Steady-state period lengths are determined by their own dates (end_datetime - start_datetime + 1 day),
         regardless of how their start or end dates relate to adjacent transient periods, so they are always
         at least 1 day. For example, a 1-day steady-state period starting on 1/1/2020
         followed by a transient period starting on 1/1/2020 is allowed.

     - Steady-state stress periods are treated independently; their start_ and end_datetimes
       are left as-is unless no end_datetime column is provided, in which case end_datetimes are computed
       from start_datetime and time. End dates for transient periods preceding a steady-state period are not adjusted to
       align with the subsequent steady-state period.
     - Sub-daily periods or timesteps: if any period or timestep length is not a whole number of days,
       or any input start_ or end_datetime is not at midnight, the whole table is treated as sub-daily.
       In that case, end_datetimes are set to one second before the next start (instead of one day before),
       and the last end_datetime is its start_datetime + length - 1 second.

    Parameters
    ----------
    perioddata : DataFrame
        Perioddata table produced by modflow-setup or modflow-obs.

        - Must have 'start_datetime' and either a 'time' column (of elapsed model times at the end of each period,
          in days), or an 'end_datetime' column of period end dates. An index named 'time' is treated as
          a 'time' column.
        - If 'time' is present, 'end_datetime' is optional; if supplied, it is ignored for transient periods,
          and transient end_datetimes are computed from start_datetime and time.
        - The rows in perioddata can represent model stress periods composed of one or more timesteps, or
          individual timesteps.

    Returns
    -------
    Operates on perioddata in-place.

        - start_ and end_datetimes are formatted as '%Y-%m-%d' strings if all periods or timesteps
          are whole days, or as '%Y-%m-%d %H:%M:%S' strings otherwise (the format is the same for the whole table).
        - Rows are sorted by time.
        - An integer time column stays integer; a time column computed from dates is float.
        - Running the function twice gives the same result as running it once.

    Notes
    -----
    Assumes time units of days.
    """
    # 1. Validate input and normalize the time column
    if 'time' not in perioddata.columns and perioddata.index.name == 'time':
        perioddata['time'] = perioddata.index.values
    has_time = 'time' in perioddata.columns
    has_end = 'end_datetime' in perioddata.columns
    if not has_time and not has_end:
        raise ValueError("perioddata must have a 'time' column "
                         "or an 'end_datetime' column.")
    if has_time:
        _check_period_lengths(perioddata['time'].values)
    perioddata['start_datetime'] = pd.to_datetime(perioddata['start_datetime'])
    if has_end:
        perioddata['end_datetime'] = pd.to_datetime(perioddata['end_datetime'])

    # 2. Sub-daily if any input date isn't at midnight
    # (lengths that aren't whole days are checked in step 5)
    sub_daily = _has_time_of_day(perioddata['start_datetime'])
    if has_end:
        sub_daily = sub_daily or _has_time_of_day(perioddata['end_datetime'])

    # 3. Without a time column, construct one from the input dates
    # (start_datetimes are treated as correct)
    if not has_time:
        eps = _end_offset(sub_daily)
        perioddata['time'] = _time_from_dates(
            perioddata['start_datetime'].values,
            perioddata['end_datetime'].values,
            _get_steady(perioddata), eps)

    # 4. Sort by time
    # (an index also named 'time' would make sort_values ambiguous)
    index_name = perioddata.index.name
    if index_name == 'time':
        perioddata.index.name = None
    perioddata.sort_values(by='time', kind='stable', inplace=True)
    perioddata.index.name = index_name

    # 5. Period lengths from time (the first period's length is its time)
    time = perioddata['time'].values.astype(float)
    lengths = np.diff(time, prepend=0.)
    sub_daily = sub_daily or not np.allclose(lengths, np.round(lengths))
    eps = _end_offset(sub_daily)
    length_deltas = pd.to_timedelta(lengths, unit='D')
    if sub_daily:
        # avoid floating point noise in timestamps
        length_deltas = length_deltas.round('s')
    steady = _get_steady(perioddata)
    input_starts = pd.DatetimeIndex(perioddata['start_datetime'].values)
    input_ends = pd.DatetimeIndex(perioddata['end_datetime'].values) \
        if has_end else None

    # 6. Start datetimes: keep the first input start, steady-state starts,
    # and transient starts following a steady-state period;
    # otherwise previous transient start + previous transient length
    new_starts = []
    for i in range(len(perioddata)):
        if i == 0 or steady[i] or steady[i-1]:
            new_starts.append(input_starts[i])
        else:
            new_starts.append(new_starts[-1] + length_deltas[i-1])
    new_starts = pd.DatetimeIndex(new_starts)

    # 7. End datetimes (inclusive): start + length - 1 day (or 1 second if sub-daily);
    # steady-state end dates are kept if supplied
    new_ends = new_starts + length_deltas - eps
    if has_end:
        keep_end = steady & ~pd.isnull(input_ends)
        new_ends = pd.DatetimeIndex(np.where(keep_end, input_ends, new_ends))

    # 8. Transient start datetimes must be later than the previous transient start
    transient = np.flatnonzero(~steady)
    overlapping = transient[1:][np.diff(new_starts[transient].asi8) <= 0]
    if len(overlapping) > 0:
        bad = perioddata.iloc[np.sort(np.concatenate([overlapping - 1, overlapping]))]
        raise ValueError("Overlapping stress periods or timesteps "
                         "(transient start_datetimes must be later than the "
                         f"previous transient start_datetime); check perioddata input.\n{bad}")

    # 9. Write the results as strings
    fmt = '%Y-%m-%d %H:%M:%S' if sub_daily else '%Y-%m-%d'
    perioddata['start_datetime'] = new_starts.strftime(fmt).values
    perioddata['end_datetime'] = new_ends.strftime(fmt).values


def _get_steady(perioddata):
    """Boolean array of steady-state periods (all False if no 'steady' column)."""
    if 'steady' not in perioddata.columns:
        return np.zeros(len(perioddata), dtype=bool)
    return perioddata['steady'].fillna(False).astype(bool).values


def _check_period_lengths(time):
    """Raise an error if any period (or timestep) implied by the time column
    has zero or negative length, i.e. if the first time is <= 0
    or any time is repeated."""
    time = np.sort(np.asarray(time, dtype=float))
    lengths = np.diff(time, prepend=0.)
    bad = lengths <= 0
    if np.any(bad):
        raise ValueError("Zero-length (or negative-length) periods in time column; "
                         "times must be > 0 and unique.\n"
                         f"Offending times: {time[bad].tolist()}")


def _has_time_of_day(datetimes):
    """True if any (non-null) datetime isn't at midnight."""
    datetimes = pd.DatetimeIndex(datetimes).dropna()
    return bool(np.any(datetimes != datetimes.normalize()))


def _end_offset(sub_daily):
    """Offset between an inclusive end datetime and the next start."""
    return pd.Timedelta(1, unit='s') if sub_daily else pd.Timedelta(1, unit='D')


def _time_from_dates(starts, ends, steady, eps):
    """Construct model times (in days) from start and end datetimes.

    Transient period lengths are the difference between successive start_datetimes.
    The last period, steady-state periods, and transient periods followed by a
    steady-state period are given lengths based on their own dates
    (end - start + eps, consistent with inclusive end dates).
    """
    starts = pd.DatetimeIndex(starts)
    ends = pd.DatetimeIndex(ends)
    one_day = pd.Timedelta(1, unit='D')
    n = len(starts)
    lengths = []
    for i in range(n):
        own_dates = steady[i] or i == n - 1 or steady[i+1]
        if own_dates:
            length = (ends[i] - starts[i] + eps) / one_day
            if not length > 0:
                raise ValueError("end_datetime is before start_datetime "
                                 f"in row {i}: {starts[i]}, {ends[i]}")
        else:
            length = (starts[i+1] - starts[i]) / one_day
        lengths.append(length)
    return np.cumsum(lengths)


def get_input_arguments(kwargs, function, verbose=False, warn=False, exclude=None):
    """Return subset of keyword arguments in kwargs dict
    that are valid parameters to a function or method.

    Parameters
    ----------
    kwargs : dict (parameter names, values)
    function : function of class method
    warn : bool;
        If true, print supplied argument that are not in the function's signature
    exclude : sequence


    Returns
    -------
    input_kwargs : dict
    """
    np.set_printoptions(threshold=20, edgeitems=1)

    # translate the names of some variables
    # to valid flopy arguments
    # (not sure if this is the best place for this)
    translations = {'continue': 'continue_'
                    }

    if verbose:
        print('\narguments to {}:'.format(function.__qualname__))
    params = inspect.signature(function)
    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)
    input_kwargs = {}
    not_arguments = {}
    for k, v in kwargs.items():
        k_original = k
        k = translations.get(k, k)
        if k in params.parameters and not {k, k_original}.intersection(exclude):
            input_kwargs[k] = v
            if verbose:
                print_item(k, v)
        else:
            not_arguments[k] = v
    if verbose and warn:
        print('\nother arguments:')
        for k, v in not_arguments.items():
            #print('{}: {}'.format(k, v))
            print_item(k, v)
    if verbose:
        print('\n')
    return input_kwargs


def print_item(k, v):
    print('{}: '.format(k), end='')
    if isinstance(v, dict):
        if len(v) > 1:
            print('{{{}: {}\n ...\n}}'.format(*next(iter(v.items()))))
        else:
            print(v)
    elif isinstance(v, list):
        if len(v) > 3:
            print('[{} ... {}]'.format(v[0], v[-1]))
        else:
            pprint.pprint(v, compact=True)
    elif isinstance(v, pd.DataFrame):
        print(v.head())
    elif isinstance(v, np.ndarray):
        txt = 'array: {}, {}'.format(v.shape, v.dtype)
        try:
            txt += ', min: {:g}, mean: {:g}, max: {:g}'.format(v.min(), v.mean(), v.max())
        except:
            pass
        print(txt)
    else:
        print(v)
        