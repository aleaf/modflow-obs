import datetime as dt
import numpy as np
import pandas as pd
from mfobs.checks import check_obsnme_suffix


def read_statvar_file(statvar_file):
    """Read a PRMS statvar file into a pandas dataframe.

    Parameters
    ----------
    statvar_file : str or pathlike
    
    Returns
    -------
    df : pandas DataFrame (n days x n sites)
        With a datetime index (one row per day)
        and columns: *site1, site2, ...*
        
        Column names are formulated as variable-segment, for example,
        *seg_outflow-1527* for stream segment outflow at segment 1527.
    
    """
    with open(statvar_file) as src:
        n_sites = int(next(iter(src)).strip())
        site_names = []
        for i in range(n_sites):
            variable, segment = next(iter(src)).strip().split()
            segment = int(segment)
            site_names.append(f"{segment}-{variable}")
        parse = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')
        df = pd.read_csv(src, header=None,
                  delim_whitespace=True, 
                  parse_dates={'datetime': list(range(1, 7))}, date_parser=parse,
                  index_col='datetime')
    df.columns = ['time'] + site_names
    return df
    
    
def get_prms_statvar_obs(perioddata,
                         statvar_file,
                         statvar_sitenames=None,
                         obsnme_date_suffix=True,
                         obsnme_suffix_format='%Y%m%d',
                         abs=False):
    """Read raw PRMS statvar observation output from text file table with
    times along the row axis and observation sites along the column axis. Reshape
    (stack) results to be n times x n sites rows, with a single observation value
    in each row. 
    
    If there is more than one time in a stress period, retain only
    the last time (so that there is one observation per stress period for each site.
    

    Parameters
    ----------
    perioddata : DataFrame
        DataFrame with start/end dates for stress periods. Must have columns
        'per' (stress period number), 'time' (modflow time, in days), 
        'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str
        Path to MODFLOW-6 observation csv output (shape: n times rows x n obs columns).
    statvar_sitenames : dict
        Dictionary of the site names (values) associated with each HRU or stream segment 
        (keys) referenced in the statvar file.
    obsnme_date_suffix : bool
        If true, give observations a date-based suffix. Otherwise, assign a 
        elapsed time-based suffix. In either case, the format of the suffix
        is controlled by obsnme_suffix_format.
        by default True
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes. Observation names are created following the format of
        <obsprefix>_<date or elapsed time suffix>. By default, ``'%Y%m'``,
        which would yield ``'202001'`` for a Jan, 2020 observation 
        (obsnme_date_suffix=True). If obsnme_date_suffix=False, obsnme_suffix_format
        should be a decimal format in the "new-style" string format
        (e.g. '{:03d}', which would yield ``'001'`` for an elapsed time of 1 day.)
    abs : bool, optional
        Option to convert simulated values to absolute values

    Returns
    -------
    results : DataFrame
        DataFrame with one head observation per row, with the following columns:

        =================== =============================================================
        per                 zero-based model stress period
        site_no             unique identifier for each site
        variable            PRMS variable name
        obsprefix           prefix of observation name (site identifier)
        sim_value           simulated values
        datetime            pandas datetimes, based on stress period start date
        obsnme              observation name based on format of <obsprefix>_'%Y%m'

        =================== =============================================================

        Example observation names:

        site1000_202001, for a Jan. 2020 observation at site1000 (obsnme_date_suffix=True)
        
        site1000_00001, for a day 1 observation at site1000 (obsnme_date_suffix=False)


    """
    # validation checks
    check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
                        function_name='read_statvar_file')
    
    if perioddata.index.name == 'per':
        perioddata = perioddata.sort_index()
    else:
        perioddata = perioddata.sort_values(by='per')
    perioddata = perioddata.copy()
    perioddata['start_datetime'] = pd.to_datetime(perioddata['start_datetime'])
    perioddata['end_datetime'] = pd.to_datetime(perioddata['end_datetime'])
        
    if 'perlen' not in perioddata.columns:
        perioddata['perlen'] = perioddata['time'].diff().fillna(0).tolist()
    print('reading model output from {}...'.format(statvar_file))
    model_output = read_statvar_file(statvar_file)

    # convert all observation names to lower case
    model_output.columns = model_output.columns.str.lower()

    # add stress period information to model output
    # update the last time in perioddata to the last statvar time
    if perioddata['end_datetime'].iloc[-1] < model_output.index[-1]:
        perioddata['end_datetime'].iloc[-1] = model_output.index[-1]
    model_output['per'] = 0
    for i, r in perioddata.iterrows():
        # no PRMS output should be applied to steady-state periods
        if r['steady']:
            continue
        model_output.loc[r['start_datetime']:r['end_datetime'], 'per'] = r.per

    # reshape the model output from (nper rows, nsites columns) to nper x nsites rows
    periods = dict(zip(model_output.index, model_output['per']))
    times = dict(zip(model_output.index, model_output['time']))
    simval_col = 'sim_obsval'
    stacked = model_output.drop(['time', 'per'], axis=1).stack(level=0).reset_index()
    stacked.columns = ['datetime', 'obsprefix', simval_col]
    stacked['variable'] = [s.split('-')[1] for s in stacked['obsprefix']]
    if statvar_sitenames is None:
        statvar_sitenames = {}

    hrus = [int(s.split('-')[0]) for s in stacked['obsprefix']]
    sitenames = [statvar_sitenames.get(hru, hru) for hru in hrus]
    sitenames = [s.lower() if isinstance(s, str) else s for s in sitenames]
    stacked['site_no'] = sitenames
    variables = [s.split('-')[1] for s in stacked['obsprefix']]
    stacked['variable'] = variables
    stacked['obsprefix'] = [f"{sitename}-{variable}".lower() 
                        for sitename, variable in zip(sitenames, variables)]
    stacked['per'] = [periods[ts] for ts in stacked['datetime']]
    stacked['time'] = [times[ts] for ts in stacked['datetime']]

    # optionally convert simulated values to absolute values
    if abs:
        stacked[simval_col] = stacked[simval_col].abs()

    # assign obsnames using the prefixes (location identifiers) and month
    obsnames = []
    for prefix, per, dt in zip(stacked.obsprefix, stacked.per, stacked.datetime):
        if obsnme_date_suffix and not pd.isnull(dt):
            name = f"{prefix}_{dt.strftime(obsnme_suffix_format)}"
        elif not obsnme_date_suffix:
            suffix = f"{per:{obsnme_suffix_format.strip('{:}')}}"
            name = f"{prefix}_{suffix}"
        else:
            name = prefix
        obsnames.append(name)
    stacked['obsnme'] = obsnames
    if stacked['obsnme'].duplicated().any():
        duplicates = stacked.loc[stacked['obsnme']. \
            duplicated(keep=False)].sort_values(by='obsnme')
        msg = ("mfobs.prms.get_prms_statvar_obs:"
               "Duplicate observation names. If obsnme_date_suffix=True, "
               "you may need a more specific obsnme_suffix_format, e.g. '%Y%m%d\n"
               ""
               )
        raise ValueError(msg)
    stacked.index = stacked['obsnme']
    sort_cols = [c for c in ['obsprefix', 'per', 'layer'] if c in stacked.columns]
    stacked.sort_values(by=sort_cols, inplace=True)
    results = stacked[['datetime', 'site_no', 'variable', 'obsprefix', 'obsnme',
                        'sim_obsval']].copy()
    return results