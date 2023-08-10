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
    """Ensure that the start and end dates for consecutive periods are consistent 
    but don't overlap. For example given two consecutive periods start dates of 
    3/1/2020 and 4/1/2020, we want the end dates should be 3/31/2020 and 4/30/2020,
    so that a selection in pandas with :meth:`pandas.DataFrame.loc` doesn't include
    values from the next period.

    Parameters
    ----------
    perioddata : DataFrame
        Perioddata table produced by modflow-setup. Must have
        'start_datetime' and either a 'time' column 
        (of elapsed times at the end of each period, in days), 
        or an 'end_datetime' column of period end dates.

    Returns
    -------
    Operates on perioddata in-place.

    Notes
    -----
    Assumes time units of days.
    """
    orig_perioddata = perioddata.copy()
    if 'time' not in perioddata.columns and perioddata.index.name != 'time':
        timedeltas = perioddata['end_datetime'] - perioddata['start_datetime'].min()
        perioddata['time'] = timedeltas.dt.days + 1
    if perioddata.index.name == 'time':
        perioddata.sort_index(inplace=True)
    else:
        perioddata.sort_values(by='time', inplace=True)
    start_datetimes = pd.to_datetime(perioddata['start_datetime'].values)
    perlen = np.array(perioddata['time'].tolist()[:1] + perioddata['time'].diff().tolist()[1:])
    new_end_datetimes = start_datetimes + pd.to_timedelta(perlen - 1, unit='d')
    # fix any invalid end_datetimes resulting from the assumption above
    # (that perlen is between successive start dates)
    # there may be gaps between the periods, 
    # for example in a successive steady-state simulation
    invalid_end_datetimes = new_end_datetimes[:-1] > perioddata['start_datetime'][1:]
    if np.any(invalid_end_datetimes):
        next_start_datetimes = perioddata['start_datetime'][1:][invalid_end_datetimes]
        corr_end_datetimes = pd.to_datetime(next_start_datetimes) - pd.Timedelta(1, 'D')
        # make boolean vector of full length
        # (including last end datetime, which can't be invalid)
        invalid_end_datetimes = invalid_end_datetimes.tolist() + [False]
        # cast new_end_datetimes to a Series so that we can assign values via a slice
        new_end_datetimes = pd.Series(new_end_datetimes)
        new_end_datetimes[invalid_end_datetimes] = corr_end_datetimes.tolist()
        # recast to DateTimeIndex for consistnecy with no invalid_end_datetimes case
        new_end_datetimes = pd.DatetimeIndex(new_end_datetimes)
    perioddata['start_datetime'] = start_datetimes.strftime('%Y-%m-%d')
    perioddata['end_datetime'] = new_end_datetimes.strftime('%Y-%m-%d')


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
        