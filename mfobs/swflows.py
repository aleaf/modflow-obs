"""
Functions for working with streamflows
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from mfobs.fileio import write_insfile
from mfobs.modflow import get_mf6_single_variable_obs
from mfobs.utils import fill_nats, set_period_start_end_dates


def get_flux_obs(perioddata,
                 model_output_file='meras3_1L.sfr.obs.output.csv',
                 observed_values_file='../tables/flux_obs.csv',
                 observed_values_metadata_file=None,
                 variable_name='flux',
                 observed_values_site_id_col='obsprefix',
                 observed_values_datetime_col='datetime',
                 obsnme_date_suffix_format='%Y%m',
                 observed_values_obsval_col='obsval',
                 observed_values_group_column='group',
                 observed_values_unc_column='uncertainty',
                 aggregrate_observed_values_by='mean',
                 drop_groups=None,
                 label_period_as_steady_state=None, steady_state_period_start=None,
                 steady_state_period_end=None,
                 outfile=None,
                 write_ins=False):
    """[summary]

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str, optional
        [description], by default 'meras3_1L.sfr.obs.output.csv'
    observed_values_file : str, optional
        [description], by default '../tables/flow_obs_by_stress_period.csv'
    observed_values_column : str, optional
        Column in obs_values_file with measured flux values
    variable_name : str, optional
        [description], by default 'measured'
    outfile : str, optional
        [description], by default 'processed_flux_obs.dat'
    write_ins : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    outpath = Path('.')
    if outfile is not None:
        outpath = Path(outfile).parent

    obs_values_column = 'obs_' + variable_name  # output column with observed values
    sim_values_column = 'sim_' + variable_name  # output column with simulated equivalents to observed values

    perioddata = perioddata.copy()
    set_period_start_end_dates(perioddata)
    perioddata.index = perioddata.per

    results = get_mf6_single_variable_obs(perioddata, model_output_file=model_output_file,
                                          variable_name=variable_name,
                                          obsnme_date_suffix_format=obsnme_date_suffix_format,
                                          label_period_as_steady_state=label_period_as_steady_state)

    # rename columns to their defaults
    renames = {observed_values_site_id_col: 'obsprefix',
               observed_values_datetime_col: 'datetime',
               observed_values_group_column: 'group',
               observed_values_unc_column: 'uncertainty'
               }

    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file,
                               dtype={observed_values_site_id_col: object})
    else:
        observed = observed_values_file
    observed.rename(columns=renames, inplace=True)
    #observed.index = observed['obsnme']

    # read in the observed values metadata
    if observed_values_metadata_file is not None:
        if not isinstance(observed_values_metadata_file, pd.DataFrame):
            metadata = pd.read_csv(observed_values_metadata_file,
                               dtype={observed_values_site_id_col: object})
        else:
            metadata = observed_values_metadata_file
        metadata.rename(columns=renames, inplace=True)

        # join the metadata to the observed data
        metadata.index = metadata['obsprefix'].values
        observed.index = observed['obsprefix'].values
        join_cols = [c for c in ['screen_top', 'screen_botm', 'x', 'y', 'layer']
                     if c in metadata.columns]
        observed = observed.join(metadata[join_cols])

    # convert obs names and prefixes to lower case
    observed['obsprefix'] = observed['obsprefix'].str.lower()

    # cast datetimes to pandas datetimes
    observed['datetime'] = pd.to_datetime(observed['datetime'])
    observed['steady'] = False  # flag for steady-state observations

    # drop model results that aren't in the obs information file
    # these are probably observations that aren't in the model time period
    # (and therefore weren't included in the parent model calibration;
    # but modflow-setup would include them in the MODFLOW observation input)
    # also drop sites that are in the obs information file, but not in the model results
    # these include sites outside of the model (i.e. in the inset when looking at the parent)
    no_info_sites = set(results.obsprefix).symmetric_difference(observed.obsprefix)
    # dump these out to a csv
    print('Dropping {} sites with no information'.format(len(no_info_sites)))
    dropped_obs_outfile = outpath / 'dropped_head_observation_sites.csv'
    results.loc[results.obsprefix.isin(no_info_sites)].to_csv(dropped_obs_outfile,
                                                              index=False)
    results = results.loc[~results.obsprefix.isin(no_info_sites)].copy()
    observed = observed.loc[~observed.obsprefix.isin(no_info_sites)].copy()

    # for each model stress period, get the simulated values
    # and the observed equivalents
    observed.index = pd.to_datetime(observed.datetime)
    periods = results.groupby('per')
    observed_simulated_combined = []
    for per, data in periods:

        # get the equivalent observed values
        start, end = perioddata.loc[per, ['start_datetime', 'end_datetime']]
        suffix = pd.Timestamp(start).strftime(obsnme_date_suffix_format)

        # steady-state observations can represent a period
        # other than the "modflow time" in the perioddata table
        if per == label_period_as_steady_state:
            suffix = 'ss'
            if steady_state_period_start is not None:
                start = steady_state_period_start
            if steady_state_period_end is not None:
                end = steady_state_period_end
        observed_in_period = observed.loc[start:end].reset_index(drop=True)
        if len(observed_in_period) == 0:
            warnings.warn(('Stress period {}: No observations between start and '
                           'end dates of {} and {}!'.format(per, start, end)))
            continue
        observed_in_period.sort_values(by=['obsprefix', 'datetime'], inplace=True)
        if 'n' not in observed_in_period.columns:
            observed_in_period['n'] = 1
        by_site = observed_in_period.groupby('obsprefix')
        observed_in_period_rs = getattr(by_site, aggregrate_observed_values_by)()
        observed_in_period_rs['n'] = by_site.n.sum()
        observed_in_period_rs['datetime'] = pd.Timestamp(start)
        observed_in_period_rs.reset_index(inplace=True)  # put obsprefix back

        missing_cols = set(observed_in_period.columns).difference(observed_in_period_rs.columns)
        for col in missing_cols:
            observed_in_period_rs[col] = by_site[col].first().values
        observed_in_period_rs = observed_in_period_rs[observed_in_period.columns]
        obsnames = ['{}_{}'.format(prefix.lower(), suffix)
                    for prefix in observed_in_period_rs.obsprefix]
        observed_in_period_rs['obsnme'] = obsnames
        observed_in_period_rs.index = observed_in_period_rs['obsnme']

        # get the simulated equivalents
        any_simulated_obs = data.obsnme.isin(observed_in_period_rs.obsnme).any()
        if not any_simulated_obs:
            continue
        sim_values = []
        #for obsnme, layer in zip(observed_in_period_rs.obsnme, observed_in_period_rs.layer):
        #    obsnme_results = data.loc[obsnme]
        #    # if a DataFrame (with simulated values for multiple layers) is returned
        #    if len(obsnme_results.shape) == 2:
        #        layer = obsnme_results.iloc[np.argmin(obsnme_results.layer - layer)]['layer']
        #        sim_value = obsnme_results.iloc[layer][sim_values_column]
        #    # Series (row) in results DataFrame with single simulated value
        #    else:
        #        sim_value = obsnme_results[sim_values_column]
        #    sim_values.append(sim_value)
        observed_in_period_rs[sim_values_column] = data.reindex(observed_in_period_rs.index)[sim_values_column]

        # add stress period and observed values
        observed_in_period_rs['per'] = per
        observed_in_period_rs[obs_values_column] = observed_in_period_rs[observed_values_obsval_col]
        observed_simulated_combined.append(observed_in_period_rs)

    # Combined DataFrame of observed heads and simulated equivalents
    obsdata = pd.concat(observed_simulated_combined)

    # raise an error if there are duplicates- reindexing below will fail if this is the case
    if obsdata.index.duplicated().any():
        msg = ('The following observations have duplicate names. There should only be'
               'one observation per site, for each time period implied by the '
               'obsnme_date_suffix_format parameter.\n{}'
               .format(obsdata.loc[obsdata.duplicated()]))
        raise ValueError(msg)

    # drop any observations in specified groups
    # (e.g. lake stages that should be compared with lake package output)
    if drop_groups is not None and 'group' in obsdata.columns:
        obsdata = obsdata.loc[~obsdata.group.isin(drop_groups)].copy()

    # nans are where sites don't have observation values for that period
    # or sites that are in other model (inset or parent)
    obsdata.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # reorder the columns
    columns = ['datetime', 'per', 'obsprefix', 'obsnme', obs_values_column, sim_values_column,
               'group', 'uncertainty']
    columns = [c for c in columns if c in obsdata.columns]
    obsdata = obsdata[columns].copy()
    if 'layer' in columns:
        obsdata['layer'] = obsdata['layer'].astype(int)

    # fill NaT (not a time) datetimes
    fill_nats(obsdata, perioddata)

    obsdata.sort_values(by=['obsprefix', 'per'], inplace=True)
    if outfile is not None:
        obsdata.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(obsdata, str(outfile) + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=sim_values_column, index=False)
    return obsdata
