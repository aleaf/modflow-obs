"""
Functions for working with lake observations
"""
import pandas as pd
from mfobs.fileio import write_insfile
from mfobs.modflow import read_mf6_lake_obs


def get_lake_stage_obs(lake_obs_files, perioddata, observed_values_file,
                       lake_site_numbers,
                       outfile=None,
                       variable_name='stage',
                       observed_values_site_id_col='obsprefix',
                       observed_values_obsval_col='measured',
                       write_ins=True):
    # read in the observed values and site locations
    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file)
    else:
        observed = observed_values_file
    observed.index = observed['obsnme']

    dfs = []
    for name, f in lake_obs_files.items():
        df = read_mf6_lake_obs(f, perioddata)
        df.reset_index(inplace=True)  # put index into datetime column

        # add obsnames
        steady = perioddata.steady.values
        prefix = '{}_lk'.format(lake_site_numbers[name])
        obsnme = []
        group = []
        for per, dt in zip(df.kper, df.datetime):
            if steady[per]:
                obsnme.append(prefix + '_ss')
            else:
                obsnme.append('{}_{}'.format(prefix, dt.strftime('%Y%m')))
        df['obsnme'] = obsnme
        df.index = df.obsnme
        df['obsprefix'] = prefix
        df['name'] = name
        sim_values_column = 'sim_' + variable_name
        obs_values_column = 'obs_' + variable_name
        df[obs_values_column] = observed[observed_values_obsval_col]
        df['group'] = observed['group']
        # rename columns for consistency with other obs
        renames = {'stage': sim_values_column,
                   'kper': 'per'}
        df.rename(columns=renames, inplace=True)
        # drop values that don't have an observation
        df.dropna(subset=[obs_values_column], axis=0, inplace=True)
        dfs.append(df)
    df = pd.concat(dfs)

    # write output
    if outfile is not None:
        df.to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(df, outfile + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=sim_values_column, index=False)
    return df


def get_lake_in_out_gw_fluxes(cell_budget_file, perioddata, precision='double',
                              lakenames=None,
                              outfile=None):
    """Read Lake/groundwater flux information from MODFLOW-6 cell-by-cell budget output;
    sum in and out fluxes by lake, for each time recorded.

    Parameters
    ----------
    cell_budget_file : str
        Path to MODFLOW-6 binary cell budget output.
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    precision : str, {'double', 'single'}
        Precision of data in cell_budget_fil. MODFLOW-6 output is written in double precision
        (by default 'double')
    lakenames : dict, optional
        Dictionary of lake name for each lake number, by default None
    outfile : str, optional
        Output file for saving table of summed lake fluxes, by default 'lake_gw_fluxes.csv'

    Returns
    -------
    df : DataFrame
        Table of summed lake fluxes.

    """
    # open the cell budget file
    cbbobj = bf.CellBudgetFile(cell_budget_file, precision=precision)

    # get the times corresponding to each record
    kstpkper = cbbobj.get_kstpkper()
    times = cbbobj.get_times()

    # index perioddata by modflow time
    perioddata.index = perioddata.time

    # for each record summarize the in and out fluxes by lake
    sums = []
    for i, (kstp, kper) in enumerate(kstpkper):

        # results come out as a list of recarrays
        # cast the first item to a DataFrame
        results = cbbobj.get_data(text='LAK', kstpkper=(kstp, kper))
        df = pd.DataFrame(results[0])

        # categorize as in or out
        df['in_out'] = 'in'  # gw inflow to lake
        df.loc[df.q > 0, 'in_out'] = 'out'  # lake leakage to gw

        # groupby lake and by in/out; results in 2 levels of columns
        sums_by_lake = df.groupby(['node2', 'in_out']).sum().unstack(level=-1)
        # flatten the columns
        cols = ['_'.join(col).strip() for col in sums_by_lake.columns.values]
        sums_by_lake.columns = cols

        # check for only in or only out possibility
        if 'q_in' not in cols:
            sums_by_lake['q_in'] = 0
        if 'q_out' not in cols:
            sums_by_lake['q_out'] = 0

        # just keep the fluxes
        sums_by_lake = sums_by_lake[['q_in', 'q_out']].copy()
        sums_by_lake.rename(columns={'q_in': 'gw_in',
                                     'q_out': 'gw_out',
                                     }, inplace=True)
        sums_by_lake['gw_net'] = sums_by_lake['gw_in'] + sums_by_lake['gw_out']
        sums_by_lake.index.name = 'lakeno'
        if lakenames is not None:
            sums_by_lake['lake_name'] = [lakenames.get(no, '') for no in sums_by_lake.index]

        # add time information
        sums_by_lake['kstp'] = kstp
        sums_by_lake['kper'] = kper
        sums_by_lake['mf_time'] = times[i]
        sums_by_lake['start_datetime'] = perioddata.loc[times[i], 'start_datetime']
        sums_by_lake['end_datetime'] = perioddata.loc[times[i], 'end_datetime']

        sums.append(sums_by_lake)

    df = pd.concat(sums)
    df.reset_index(inplace=True)
    df.index = df['start_datetime']
    df.sort_values(by=['lakeno', 'mf_time'], inplace=True)

    if outfile is not None:
        df.to_csv(outfile, index=False)
    return df

