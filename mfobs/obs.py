"""
General functions for creating observation input for PEST, 
including base observations representing field measurements and simulated equivalents, 
and derivative observations that are created from the base observations.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from mfobs.checks import check_obsnme_suffix
from mfobs.fileio import write_insfile, get_insfile_observations
from mfobs.sep import ih_method
from mfobs.utils import fill_nats, set_period_start_end_dates


def get_base_obs(perioddata,
            model_output,
            observed_values_file,
            variable_name=None,
            simulated_values_column='sim_obsval',
            observed_values_metadata_file=None,
            observed_values_site_id_col='obsprefix',
            observed_values_datetime_col='datetime',
            obsnme_date_suffix=True,
            obsnme_suffix_format='%Y%m',
            observed_values_obsval_col='obsval',
            observed_values_group_column='obgnme',
            observed_values_unc_column='uncertainty',
            aggregrate_observed_values_method='mean',
            drop_groups=None,
            label_period_as_steady_state=None, steady_state_period_start=None,
            steady_state_period_end=None, forecast_sites=None,
            forecast_start_date=None, forecast_end_date=None,
            forecasts_only=False, forecast_sites_only=False,
            outfile=None, verbose=True,
            write_ins=False):
    """Get a set of base observations from a tables of model output, observed 
    values, and model time discretizaton.

    Parameters
    ----------
    perioddata : str, pathlike or DataFrame
        Path to csv file or pandas DataFrame with start/end dates for stress periods or timesteps. 
        Must have a `start_datetime` and either a `time` column 
        (of elapsed times at the end of each period, in days), 
        or an `end_datetime` column of period end dates.
        
        =================== =======================================================
        start_datetime      start date for each stress period or timestep
        end_datetime        (optional) end date for each stress period or timestep
        time                (optional) elapsed simulation time at the end of each 
                            period, in days
        =================== =======================================================

    model_output : DataFrame
        DataFrame with one head observation per row, with the following columns:

        ======================= =============================================================
        site_no                 unique identifier for each site
        variable                variable name for the simulated values
        obsprefix               prefix of observation name (<site_no>-<variable>)
        simulated_values_column column with simulated values
        datetime                pandas datetimes, based on stress period end date
        layer                   zero-based model layer
        obsnme                  observation name based on format of <obsprefix>_'%Y%m'
        ======================= =============================================================
        
        Usually, a helper function such as :func:`mfobs.modflow.get_mf6_single_variable_obs`
        or other custom code will be used to produce model_output from the 
        raw model output files.
        
        Observation prefixes in the `obsprefix` column are assumed to be formated as
        **<site_no>-<variable>**, where `site_no` is a unique identifier for a site
        contained in `observed_values_file` and `variable` matches the `variable_name` argument
        to this function (and is the variable name for the observed values in `observed_values_file`).
        
        All observation prefixes are assumed to be case-insensitive (obsprefixes in the model_output 
        and obseved_values_files tables are all converted to lower case prior to matching).
        
    observed_values_file : str, pathlike or DataFrame
        Path to csv file or pandas DataFrame with observed values.
        Must have the following columns (column names in brackets <> are supplied
        with other arguments to this function):

        ============================== ================================================================================
        <observed_values_obsval_col>   Column with observed values
        <observed_values_site_id_col>  Column with unique identifier for each site
        <observed_values_datetime_col> Column with date/time for each observation
        <observed_values_unc_column>   Column with estimated uncertainty values for each observation
        ============================== ================================================================================
    variable_name : str, optional
        Name for the type of values being processed, 
        e.g. 'head', 'downstream-flow' or 'stage'. Required if the `obsprefix`
        values in the `model_output` table are formatted as **<site number>-<variable>**;
        if supplied, the observation names created for the observations in 
        `observed_values_file` will also have **<site number>-<variable>** prefixes,
        and the observed values and simulated equivalents can be aligned.
        
        By default, None, in which case the `obsprefix` values in the `model_output`
        table are assumed to be the site numbers; prefixes for observations
        in `observed_values_file` will also then be site numbers.
    simulated_values_column : str, optional
        Column in model output with simulated values
    observed_values_metadata_file : str, pathlike or DataFrame, optional
        Table with site information for the observed values.

    observed_values_obsval_col : str, optional
        Column in obs_values_file with measured values
    observed_values_site_id_col : str
        Column with unique identifier for each site,
        by default, 'obsprefix'
    observed_values_datetime_col : str
        Column with date/time for each observation,
        by default, 'datetime'
    observed_values_unc_column : str
        Column with estimated uncertainty values for each observation,
        by default, 'uncertainty'
    aggregrate_observed_values_method : str
        Method for aggregating observed values to time periods bracked by 
        aggregate_start_dates and aggregate_end_dates.
        Can be any method used to aggregate values on a pandas groupby object
        (e.g. 'mean', 'last', etc.)
    obsnme_date_suffix : bool
        If true, give observations a date-based suffix. Otherwise, assign a 
        stress period- or timestep-based suffix. In either case, the format of the suffix
        is controlled by obsnme_suffix_format.
        by default True
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes. Observation names are created following the format of
        <obsprefix>_<date or stress period suffix>. By default, ``'%Y%m'``,
        which would yield ``'202001'`` for a Jan, 2020 observation 
        (obsnme_date_suffix=True). If obsnme_date_suffix=False, obsnme_suffix_format
        should be a decimal format in the "new-style" string format
        (e.g. '{:03d}', which would yield ``'001'`` for stress period 1.)
    label_period_as_steady_state : int, optional
        Zero-based model stress period where observations will be
        assigned the suffix 'ss' instead of a date suffix.
        By default, None, in which case all model output is assigned
        a date suffix based on the start date of the stress period.
        Passed to :func:`~mfobs.modflow.get_mf6_single_variable_obs`.
    steady_state_period_start : str, optional
        Start date for the period representing steady-state conditions.
        Observations between ``steady_state_period_start`` and ``steady_state_period_end``
        will be averaged to create additional observations with the suffix 'ss'.
        The steady-state averages will be matched to model output from the
        stress period specified by ``label_period_as_steady_state``.
        By default None, in which case all observations are used 
        (if ``observed_values_datetime_col is None``) 
        or no steady-state observations 
        are created (``observed_values_datetime_col`` exists).
    steady_state_period_end : str, optional
        End date for the period representing steady-state conditions.
        By default None, in which case all observations are used 
        (if ``observed_values_datetime_col is None``) 
        or no steady-state observations 
        are created (``observed_values_datetime_col`` exists).
    forecast_sites : str or sequence, optional
        At these sites, observations will be created for each simulated value,
        regardless is there is an observed equivalent. Can be supplied
        as a sequence of site numbers (`site_id`s) or ``'all'`` to
        include all sites. By default, None (no forecast sites).
    forecast_start_date : str, optional
        Start date for forecast period. When forecast_sites is not
        ``None``, forecast observations will be generated for each
        time between `forecast_start_date` and `forecast_end_date`.
        By default, None (generate forecasts for any time with missing values).
    forecast_end_date : str, optional
        End date for forecast period. When forecast_sites is not
        ``None``, forecast observations will be generated for each
        time between `forecast_start_date` and `forecast_end_date`.
        By default, None (generate forecasts for any time with missing values).
    forecasts_only : bool, optional
        Use this option to only output forecast observations 
        (those without an observed equivalent), subject to the parameters of
        `forecast_sites`, `forecast_start_date`, and `forecast_end_date`.
    forecast_sites_only : bool, optional
        Option to only output observations at sites specified
        with `forecast_sites` (has no effect if ``forecast_sites='all'``). If
        ``forecasts_only=False``, the output will include forecast and non-forecast
        observations (for a continuous time-series).
    verbose : bool
        Controls the verbosity of screen outputs. If True, warnings will be printed
        when there are no observations within a stress period and forecast_sites is None, 
        (meaning no observation results will be processed for that stress period), or when
        supplied observations fall outside of the model timeframe. Within a workflow,
        verbose can be set equal to write_ins for initial debugging, and the subsequently
        set False during history matching runs.
        by default, True
    outfile : str, optional
        [description], by default 'processed_flux_obs.dat'
    write_ins : bool, optional
        [description], by default False

    Returns
    -------
    base_obs : DataFrame
        Table of base observations with the following columns:
        
        =================== =============================================================
        datetime            pandas datetimes, based on stress period end date
        per                 MODFLOW stress period
        site_no             unique site identifier
        obsprefix           prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>
        obsval              observed values
        sim_obsval          simulated equivalents to observed values
        obgnme              observation group name

        =================== =============================================================
        
    """
    # validation checks
    check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
                        function_name='get_base_obs')
    
    outpath = Path('.')
    if outfile is not None:
        outpath = Path(outfile).parent

    #obs_values_column = 'obsval'
    output_sim_values_column = 'sim_obsval'
    output_obs_values_column = 'obsval'
    #if variable_name is None:
    #    obs_values_column = 'obsval'  #'obs_value'
    #    #sim_values_column = 'sim_value'
    #else:
    #    obs_values_column = 'obs_' + variable_name  # output column with observed values
    #    #sim_values_column = 'sim_' + variable_name  # output column with simulated equivalents to observed values

    # read in/set up the perioddata table
    if not isinstance(perioddata, pd.DataFrame):
        perioddata = pd.read_csv(perioddata)
    else:
        perioddata = perioddata.copy()        
    set_period_start_end_dates(perioddata)
    #perioddata.index = perioddata.per

    # model results
    results = model_output.copy()
    results['obsprefix'] = results['obsprefix'].str.lower()

    # rename columns to their defaults
    renames = {observed_values_site_id_col: 'site_no',
               observed_values_datetime_col: 'datetime',
               observed_values_group_column: 'obgnme',
               observed_values_unc_column: 'uncertainty'
               }

    # read in/set up the observed values table
    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file,
                               dtype={observed_values_site_id_col: object})
    else:
        observed = observed_values_file
    observed.rename(columns=renames, inplace=True)
    if 'obsprefix' not in observed.columns:
        if variable_name is not None:
            observed['obsprefix'] = [f"{sn}-{variable_name}" 
                                     for sn in observed['site_no']]
            if forecast_sites is not None and forecast_sites != 'all':
                forecast_sites = [f"{sn}-{variable_name}" 
                                     for sn in forecast_sites]
        else:
            observed['obsprefix'] = observed['site_no'].astype(str)
    else:
        observed['obsprefix'] = observed['obsprefix'].astype(str)

    # read in the observed values metadata
    if observed_values_metadata_file is not None:
        if not isinstance(observed_values_metadata_file, pd.DataFrame):
            metadata = pd.read_csv(observed_values_metadata_file,
                               dtype={observed_values_site_id_col: object})
        else:
            metadata = observed_values_metadata_file
        metadata.rename(columns=renames, inplace=True)
        if 'obsprefix' not in metadata.columns:
            metadata['obsprefix'] = metadata[observed_values_site_id_col]

        # join the metadata to the observed data
        metadata.index = metadata['obsprefix'].values
        observed.index = observed['obsprefix'].values
        join_cols = [c for c in ['site_no', 'screen_top', 'screen_botm', 'x', 'y', 'layer']
                     if c in metadata.columns]
        observed = observed.join(metadata[join_cols])

    # convert obs names and prefixes to lower case
    observed['obsprefix'] = observed['obsprefix'].str.lower()

    # make a dictionary of site metadata for possible use later
    temp = observed.copy()
    temp.index = temp['obsprefix'].str.lower()
    site_info_dict = temp.to_dict()
    for col in 'datetime', 'obsval', observed_values_obsval_col:
        if col in site_info_dict:
            del site_info_dict[col]
        
    del temp

    # cast datetimes to pandas datetimes
    # (observed data may not have a datetime col. if model is steady-state)
    if observed_values_datetime_col is not None and 'datetime' in observed.columns:
        try:
            observed['datetime'] = pd.to_datetime(observed['datetime'])
        except ValueError:
            observed['datetime'] = pd.to_datetime(observed['datetime'], format='mixed')
        # not necessarily True
        observed['steady'] = False  # flag for steady-state observations
    elif label_period_as_steady_state is not None:
        observed['datetime'] = pd.to_datetime(
            perioddata['start_datetime'][label_period_as_steady_state])
        observed['steady'] = True
    else:
        if len(perioddata) == 1:
            observed['datetime'] = pd.to_datetime(perioddata['start_datetime'][0])
            observed['steady'] = True
        else:
            raise ValueError("Model has more than one stress period "
                             "but observed data have no 'datetime' column.")

    # drop model results that aren't in the obs information file
    # these are probably observations that aren't in the model time period
    # (and therefore weren't included in the parent model calibration;
    # but modflow-setup would include them in the MODFLOW observation input)
    # also drop sites that are in the obs information file, but not in the model results
    # these include sites outside of the model (i.e. in the inset when looking at the parent)
    no_info_sites = set(results.obsprefix).symmetric_difference(observed.obsprefix)
    if forecast_sites == 'all':
        # forecast observations at all simulated sites
        # (only drop sites that aren't simulated)
        no_info_sites = set(observed.obsprefix).difference(results.obsprefix)
    elif forecast_sites is not None:
        # remove selected forecast sites from 'no_info' sites to drop
        forecast_sites = {s.lower() for s in forecast_sites}
        no_info_sites = no_info_sites.difference(forecast_sites)
        
    # dump these out to a csv
    print('Dropping {} sites with no information'.format(len(no_info_sites)))
    dropped_obs_outfile = outpath / 'dropped_head_observation_sites.csv'
    results.loc[results.obsprefix.isin(no_info_sites)].to_csv(dropped_obs_outfile,
                                                              index=False)
    results = results.loc[~results.obsprefix.isin(no_info_sites)].copy()
    observed = observed.loc[~observed.obsprefix.isin(no_info_sites)].copy()
    if len(results) == 0:
        raise ValueError("No matches between observation prefixes in the model results"
                         "and observed values! Check that the obsprefixes in the "
                         "observed_values_file match those in the model_output DataFrame.")

    # for each model stress period or timestep, get the simulated values
    # and the observed equivalents
    observed.index = pd.to_datetime(observed.datetime)
    observed.sort_index(inplace=True)
    results.index = pd.to_datetime(results.datetime)
    results.sort_index(inplace=True)
    
    # integer column for stress period- or timestep-based obsnme suffixes
    # timestep-based observations
    if 'timestep' in perioddata.columns:
        perioddata['unique_timestep'] = list(range(len(perioddata)))
        per_column = 'unique_timestep'
    # stress period-based observations
    else:
        per_column = 'per'
        
    observed_simulated_combined = []

    # if no datetime column is supplied with observations,
    # only make steady state obs
    if observed_values_datetime_col is None and \
        label_period_as_steady_state is not None:
            idx = label_period_as_steady_state
            perioddata = perioddata[idx:idx+1]
            
    for i, r in perioddata.iterrows():

        # get the equivalent observed values
        #start, end = perioddata.loc[per, ['start_datetime', 'end_datetime']]
        start, end = r['start_datetime'], r['end_datetime']
        
        # date-based suffix
        if obsnme_date_suffix:  
            suffix = pd.Timestamp(end).strftime(obsnme_suffix_format)
        # stress or timestep-based period-based suffix
        else:  
            suffix = f"{r[per_column]:{obsnme_suffix_format.strip('{:}')}}"

        # steady-state observations can represent a period
        # other than the "modflow time" in the perioddata table
        if r['per'] == label_period_as_steady_state:
            suffix = 'ss'
            if steady_state_period_start is not None:
                start = steady_state_period_start
            if steady_state_period_end is not None:
                end = steady_state_period_end
        # don't process observations for a steady-state period unless 
        # it is explicitly labeled as such and given a representative date range
        elif r['steady']:
            continue
        observed_in_period_rs = aggregrate_to_period(
            observed, start, end, 
            aggregrate_observed_values_method=aggregrate_observed_values_method,
            obsnme_suffix=suffix)
        if verbose and (forecast_sites is None) and (len(observed_in_period_rs) == 0):
            if per_column == 'per':
                warnings.warn(('Stress period {}: No observations between start and '
                                'end dates of {} and {}!'.format(r['per'], start, end)))
            continue

        # get the simulated equivalents
        # first populate obsnmes
        sim_in_period_rs = aggregrate_to_period(
            results, start, end, 
            aggregrate_observed_values_method=aggregrate_observed_values_method,
            obsnme_suffix=suffix)
        
        if verbose and (sim_in_period_rs is None) or (len(sim_in_period_rs) == 0):
            if per_column == 'per':
                warnings.warn(('Stress period {}: No simulated equivalents between start and '
                                'end dates of {} and {}!'.format(r['per'], start, end)))
            continue
        if forecast_sites is not None:
            observed_in_period_rs = observed_in_period_rs.reindex(sim_in_period_rs['obsnme'].values, axis=0)
            obsprefix = observed_in_period_rs.index.str.split('_', expand=True).levels[0]
            observed_in_period_rs['obsprefix'] = obsprefix
            observed_in_period_rs['obsnme'] = observed_in_period_rs.index
            observed_in_period_rs['datetime'] = sim_in_period_rs['datetime'].values[0]
        any_simulated_obs = sim_in_period_rs.obsnme.isin(observed_in_period_rs.obsnme).any()
        
        if verbose and not any_simulated_obs and (per_column == 'per'):
            warnings.warn(('Stress period {}: No observation/simulated equivalent pairs for start and '
                            'end dates of {} and {}!'.format(r['per'], start, end)))
            continue
                
        observed_in_period_rs[output_sim_values_column] = sim_in_period_rs.reindex(observed_in_period_rs.index)[simulated_values_column]

        # add stress period and observed values
        observed_in_period_rs['per'] = r['per']
        if per_column == 'unique_timestep':
            observed_in_period_rs['timestep'] = r['timestep']
            observed_in_period_rs['unique_timestep'] = r['unique_timestep']
        observed_in_period_rs[output_obs_values_column] = observed_in_period_rs[observed_values_obsval_col]
        observed_simulated_combined.append(observed_in_period_rs)

    # Combined DataFrame of observed heads and simulated equivalents
    if len(observed_simulated_combined) == 0:
        msg = ("No overlap between model results "
               f"({results['datetime'].min():%Y-%m-%d} to {results['datetime'].max():%Y-%m-%d})"
               " and observed values "
               f"({observed['datetime'].min():%Y-%m-%d} to {observed['datetime'].max():%Y-%m-%d})!"
               )
        raise ValueError(msg)
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
    if drop_groups is not None and 'obgnme' in obsdata.columns:
        obsdata = obsdata.loc[~obsdata.obgnme.isin(drop_groups)].copy()
        
    if 'obgnme' not in obsdata.columns:
        obsdata['obgnme'] = variable_name
    
    # fill forecast obs with site info from observed dataframe
    if forecast_sites is not None:
        for k, v in site_info_dict.items():
            obsdata[k] = [v.get(p, current) 
                          for p, current in zip(obsdata['obsprefix'], obsdata[k])]
        obsdata['obsnme'] = obsdata.index
    else:  
        # nans are where sites don't have observation values for that period
        # or sites that are in other model (inset or parent)
        obsdata.dropna(subset=[output_obs_values_column], axis=0, inplace=True)

    # label forecasts in own group
    if forecast_sites is not None:
        is_forecast = obsdata[output_obs_values_column].isna()
        obsdata.loc[is_forecast, 'obgnme'] += '-forecast'
    
        # cull forecasts to specified date window
        # and specific sites (if specified)
        keep_forecasts = is_forecast.copy()  #np.array([True] * len(head_obs))
        if forecast_start_date is not None:
            keep_forecasts = (obsdata['datetime'] >= forecast_start_date)
        if forecast_end_date is not None:
            keep_forecasts &= (obsdata['datetime'] <= forecast_end_date)
        #drop = drop & is_forecast
        #head_obs = head_obs.loc[~drop].copy()
        #is_forecast = head_obs[output_obs_values_column].isna()
        if forecast_sites != 'all':
            keep_forecasts &= obsdata['obsprefix'].isin(forecast_sites)
        # option to only include forecast obs
        # (those without observed equivalents)
        if forecasts_only:
            keep = keep_forecasts
        else:
            keep = keep_forecasts | ~is_forecast
        # option to only include output from designated forecast sites
        if forecast_sites_only:
            keep = keep & obsdata['obsprefix'].isin(forecast_sites)
        obsdata = obsdata.loc[keep].copy()
        
    # add standard obsval and obgmne columns
    columns = ['datetime', 'per', 'site_no', 'obsprefix', 'obsnme', 
               output_obs_values_column, output_sim_values_column,
               'uncertainty', 'obgnme']
    if output_obs_values_column != 'obsval':
        obsdata['obsval'] = obsdata[output_obs_values_column]
        columns.insert(-1, 'obsval')


    # reorder the columns
    columns = [c for c in columns if c in obsdata.columns]
    base_obs = obsdata[columns].copy()
    if 'layer' in columns:
        base_obs['layer'] = base_obs['layer'].astype(int)

    # fill NaT (not a time) datetimes
    fill_nats(base_obs, perioddata)

    base_obs.sort_values(by=['obsprefix', 'per'], inplace=True)
    if outfile is not None:
        base_obs.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(base_obs):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(base_obs, str(outfile) + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=output_sim_values_column, index=False)
    return base_obs


def aggregrate_to_period(data, start, end, obsnme_suffix,
                         aggregrate_observed_values_method='mean',
                         ):

    data_in_period = data.loc[start:end].reset_index(drop=True)
        
    if (len(data_in_period) == 0):
        cols = list(data.columns) + ['obsnme']
        data_in_period_rs = pd.DataFrame(columns=cols)
        data_in_period_rs.index = data_in_period_rs['obsnme']
        return data_in_period_rs
    data_in_period.sort_values(by=['obsprefix', 'datetime'], inplace=True)
    if 'n' not in data_in_period.columns:
        data_in_period['n'] = 1    
    by_site = data_in_period.groupby('obsprefix')
    data_in_period_rs = by_site.first()
    data_cols = [c for c, dtype in data_in_period.dtypes.items() 
                    if 'float' in dtype.name]
    for c in data_cols:
        data_in_period_rs[c] = getattr(by_site[c], aggregrate_observed_values_method)()
    data_in_period_rs['n'] = by_site.n.sum()
    data_in_period_rs['datetime'] = pd.Timestamp(end)
    data_in_period_rs.reset_index(inplace=True)  # put obsprefix back
    
    missing_cols = set(data_in_period.columns).difference(data_in_period_rs.columns)
    for col in missing_cols:
        data_in_period_rs[col] = by_site[col].first().values
    data_in_period_rs = data_in_period_rs[data_in_period.columns]
    obsnames = ['{}_{}'.format(prefix.lower(), obsnme_suffix)
                for prefix in data_in_period_rs.obsprefix]
    data_in_period_rs['obsnme'] = obsnames
    data_in_period_rs.index = data_in_period_rs['obsnme']
    return data_in_period_rs


def get_spatial_differences(base_data, perioddata,
                            difference_sites,
                            obs_values_col='obs_head',
                            sim_values_col='sim_head',
                            use_gradients=False,
                            sep='-d-',
                            write_ins=False, outfile=None):
    r"""Takes the base_data dataframe output by :func:`mfobs.obs.get_obs` and creates
    spatial difference observations. Optionally writes an output csvfile
    and a PEST instruction file.

    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_base_obs`. Must have the following columns:
        
        =================== ========================================================================
        datetime            pandas datetimes, based on stress period end date
        per                 MODFLOW stress period
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>
        obsval              observed values
        sim_obsval          simulated equivalents to observed values
        obgnme              observation group name
        screen_top          (optional) well open interval top; required if ``use_gradients=True``
        screen_botm         (optional) well open interval bottom; required if ``use_gradients=True``
        =================== ========================================================================
        
        1) where obsprefix is assumed to be formatted as <site_no>-<variable> or simply <site_no>
        
    perioddata : DataFrame
        DataFrame with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    difference_sites : dict
        Dictionary of site numbers (keys) and other site numbers to compare to (values).
        Values can be a string for a single site, a list of strings for multiple sites,
        or a string pattern contained in multiple site numbers;
        observations at the sites represented in the values will be compared to the observation
        at the site represented by the key, at times of coincident measurements. Differences
        are computed by subtracting the values site(s) from the key site, so for example,
        to represent a gain in streamflow as positive, the downstream site should be key site.
    obs_values_col : str
        Column in ``base_data`` with observed values
    sim_values_col : str
        Column in `base_data`` with simulated equivalent values
    use_gradients : bool
        If True, compute vertical hydraulic gradients and use those for the
        observation values, if False, use differences. For this option,
        'screen_top' and 'screen_botm' columns are needed in ``base_data``.
        By default False.
    sep : str
        Separator in spatial difference obsnnames. For example, with
        sites "site1" and "site2" at time "202001", and sep='-d-', the obsnme
        would be "site1-d-site2_202001".
        by default, '-d-'
    outfile : str, optional
        CSV file to write output to. Nan values are filled with -1e30.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    =======
    spatial_differences : DataFrame
        Spatial difference observations. Columns:

        ================= ===================================================================================
        datetime          observation date-time labels
        per               model stress period
        obsprefix         observation name prefix (site identifier)
        obsnme1           name of observation from keys of ``difference_sites``
        <obs_values_col>1 observed value associated with obsnme1
        <sim_values_col>1 simulated equivalent associated with obsnme1
        screen_top1       well screen top (elevation) associated with obsnme1*
        screen_botm1      well screen botm (elevation) associated with obsnme1*
        layer1            model layer associated with obsnme1*
        obsnme2           name of observation from value(s) in ``difference_sites`` (associated with obsnme1)
        <obs_values_col>2 observed value associated with obsnme2
        <sim_values_col>2 simulated equivalent associated with obsnme2
        screen_top2       well screen top (elevation) associated with obsnme2*
        screen_botm2      well screen botm (elevation) associated with obsnme2*
        layer2            model layer associated with obsnme2*
        obs_diff          observed difference between obsnme1 and obsnme2
        sim_diff          simulated equivalent difference between obsnme1 and obsnme2
        dz                distance between well screen midpoints for obsnme1 and obsnme2*
        obs_grad          observed vertical hydraulic gradient between obsnme1 and obsnme2*
        sim_grad          simulated equivalent vertical hydraulic gradient between obsnme1 and obsnme2*
        obgnme            observation group
        obsnme            spatial difference observation name
        obsval            observation value (i.e. for PEST control file)
        sim_obsval        simulated equivalent (i.e. for PEST instruction file)
        type              description of spatial difference observations
        uncertainty       (loosely) error-based uncertainty, assumed to be 2x that of obsnme2
        ================= ===================================================================================

        Notes:

        * * denotes optional columns that may not be present.
        * Columns relating to well open interval are only created if 
          ``base_data`` has 'screen_top' and 'screen_botm' columns.
        * Negative difference or gradient values indicate a gradient towards the key site.

    """
    # copy the base data to prevent unintended side-effects
    base_data = base_data.copy()
    
    # model stress period data:
    perioddata = perioddata.copy()
    # make sure start and end dates don't overlap
    set_period_start_end_dates(perioddata)
    perioddata.index = perioddata.per

    # rename the observed and sim. eq. values columns
    #if variable is not None:
    #    renames = {obs_values_col: f'obs_{variable}',
    #            sim_values_col: f'sim_{variable}',
    #            }
    #    obs_values_col = f'obs_{variable}'
    #    sim_values_col = f'sim_{variable}'
    #else:
    renames = {obs_values_col: 'base_obsval',  # f'obs_{variable}',
            sim_values_col: 'base_sim_obsval',  # f'sim_{variable}',
            }
    obs_values_col = f'base_obsval'
    sim_values_col = f'base_sim_obsval'
    base_data.drop([#f'obs_{variable}', f'sim_{variable}', 
                    'base_obsval', 'bas_sim_obsval'], axis=1, 
                   inplace=True, errors='ignore')
    base_data.rename(columns=renames, inplace=True)
    
    # rename the observed and sim. eq. values columns
    #renames = {obs_values_col: f'obs_{variable}',
    #           sim_values_col: f'sim_{variable}',
    #           }
    #base_data.drop([f'obs_{variable}', f'sim_{variable}'], axis=1, 
    #               inplace=True, errors='ignore')
    #base_data.rename(columns=renames, inplace=True)
    #obs_values_col = f'obs_{variable}'
    #sim_values_col = f'sim_{variable}'
    
    # get a set of unique variables
    # (including '', which signifies no variable in the obsprefix, 
    #  just a site number)
    variables = {s.split('-')[1] if len(s.split('-')) > 1 else '' 
                 for s in base_data['obsprefix']}
    
    # get subset of base_data sites to compare to each key site in difference_sites
    base_data_obsprefixes = set(base_data['obsprefix'])
    groups = base_data.groupby('obsprefix')
    spatial_differences = []
    
    # key site: value/comparison sites can be include variables or not
    # if they don't, and the base_data includes variables, 
    # process each obsprefix with that site number
    for variable in variables:
        for key_obsprefix, patterns in difference_sites.items():
            
            if len(variable) > 0:
                key_obsprefix = f"{key_obsprefix.split('-')[0]}-{variable}"
                
            if key_obsprefix not in base_data_obsprefixes:
                print((f'warning: observation prefix {key_obsprefix} not in base_data. '
                    'Skipping spatial differencing.'))
                continue
            
            compare = []
            if isinstance(patterns, str):
                patterns = [patterns]
            # matching obsprefixes must have the pattern and the variable
            # if obsprefixes are simply formatted as site numbers
            # (no variables)
            # then the only variable will be an empty string,
            # which will match any string
            for pattern in patterns:
                matches = [True if pattern in obsprefix and variable in obsprefix 
                            else False for obsprefix in base_data.obsprefix]
                compare.append(matches)
            compare = np.any(compare, axis=0)
            prefixes = set(base_data.loc[compare, 'obsprefix'])

            # for each site in the subset, compare the values to the keys site
            # index by stress period
            key_values = groups.get_group(key_obsprefix).copy()
            key_values.index = key_values.per

            for obsprefix, site_observations in groups:
                if obsprefix in prefixes:
                    site_obs = site_observations.copy()
                    site_obs.rename(columns={obs_values_col: f"{obs_values_col}2",  # 'obs_head2',
                                            sim_values_col: f"{sim_values_col}2",  # 'sim_head2',
                                            'obsnme': 'obsnme2',
                                            'screen_top': 'screen_top2',
                                            'screen_botm': 'screen_botm2',
                                            'layer': 'layer2'
                                            }, inplace=True)
                    site_obs.index = site_obs.per
                    site_obs['obsnme1'] = key_values['obsnme']
                    site_obs[f"{obs_values_col}1"] = key_values[obs_values_col]
                    site_obs[f"{sim_values_col}1"] = key_values[sim_values_col]
                    if 'screen_top' in key_values.columns:
                        site_obs['screen_top1'] = key_values['screen_top']
                        if use_gradients and 'screen_botm' not in key_values.columns:
                            raise ValueError("use_gradients=True requires both screen_top "
                                             "and screen_botm columns in base_data")
                    if 'screen_botm' in key_values.columns:
                        site_obs['screen_botm1'] = key_values['screen_botm']
                    if 'layer2' in site_obs.columns:
                        site_obs['layer1'] = key_values['layer']
                    # negative values indicate gradient towards key site
                    # (key site head < values site head)
                    site_obs['obs_diff'] = site_obs[f"{obs_values_col}1"] - site_obs[f"{obs_values_col}2"]
                    site_obs['sim_diff'] = site_obs[f"{sim_values_col}1"] - site_obs[f"{sim_values_col}2"]

                    # get a screen midpoint and add gradient
                    screen_midpoint1 = None
                    if use_gradients and {'screen_top1', 'screen_botm1'}.intersection(site_obs.columns):
                        screen_midpoint1 = site_obs[['screen_top1', 'screen_botm1']].mean(axis=1)
                    if use_gradients and {'screen_top2', 'screen_botm2'}.intersection(site_obs.columns):
                        screen_midpoint2 = site_obs[['screen_top2', 'screen_botm2']].mean(axis=1)
                        if screen_midpoint1 is not None:
                            site_obs['dz'] = (screen_midpoint1 - screen_midpoint2)
                            site_obs['obs_grad'] = site_obs['obs_diff'] / site_obs['dz']
                            site_obs['sim_grad'] = site_obs['sim_diff'] / site_obs['dz']
                            
                    # default obgnme prefix
                    if 'obgnme' not in site_obs.columns:
                        if len(variable) == 0:
                            site_obs['obgnme'] = 'sdiff'
                        else:
                            site_obs['obgnme'] = variable
                    # obgnme = <prefix>_sdiff
                    # unless there is no prefix
                    site_obs['obgnme'] = [f'{g}_sdiff' if g != 'sdiff' else g 
                                          for g in site_obs['obgnme']]
                        
                     # whether to use gradients for the obsvals, or just head differences
                    if use_gradients:
                        site_obs['obsval'] = site_obs['obs_grad']
                        site_obs['sim_obsval'] = site_obs['sim_grad']
                        if len(variable) == 0:
                            site_obs['type'] = f'gradient'
                        else:
                            site_obs['type'] = f'{variable} gradient'
                    else:
                        site_obs['obsval'] = site_obs['obs_diff']
                        site_obs['sim_obsval'] = site_obs['sim_diff']
                        if len(variable) == 0:
                            site_obs['type'] = f'spatial difference'
                        else:
                            site_obs['type'] = f'spatial {variable} difference'
                    spatial_differences.append(site_obs)
    
    if len(spatial_differences) == 0:
        raise ValueError('No spatial difference site/variable pairs found! '
                         'Check that the key sites and their compare patterns exist in the base_data.')
        
    spatial_differences = pd.concat(spatial_differences)
    spatial_differences.dropna(subset=['obs_diff', 'sim_diff'], axis=0, inplace=True)

    # name the spatial head difference obs as
    # <obsprefix1><sep><obsprefix2>_<suffix>
    obsnme = []
    obsprefix = []
    for i, r in spatial_differences.iterrows():
        prefix1, suffix1 = r.obsnme1.split('_')
        prefix2, suffix2 = r.obsnme2.split('_')

        assert suffix1 == suffix2, "Observations are at different times! {}, {}".format(r.obsnme1,
                                                                                        r.obsnme2)
        # if the obsprefixes include variables; only retain the the second instance
        prefix = f"{prefix1.split('-')[0]}{sep}{prefix2}"
        obsnme.append('{}_{}'.format(prefix, suffix2))
        obsprefix.append(prefix)
    spatial_differences['obsnme'] = obsnme
    spatial_differences['obsprefix'] = obsprefix

    # clean up columns
    cols = ['datetime', 'per', 'obsprefix', 
            'obsnme1', f"{obs_values_col}1", f"{sim_values_col}1", 'screen_top1', 'screen_botm1', 'layer1',
            'obsnme2', f"{obs_values_col}2", f"{sim_values_col}2", 'screen_top2', 'screen_botm2', 'layer2',
            'obs_diff', 'sim_diff', 'dz', 'obs_grad', 'sim_grad', 
            'obsnme', 'obsval', 'sim_obsval', 'obgnme', 'type'
            ]
    cols = [c for c in cols if c in spatial_differences.columns]
    spatial_differences = spatial_differences[cols]

    spatial_differences.dropna(axis=0, subset=['obsval'], inplace=True)

    # uncertainty column is from base_data;
    # assume that spatial head differences have double the uncertainty
    # (two wells/two measurements per obs)
    if 'uncertainty' in spatial_differences.columns:
        spatial_differences['uncertainty'] *= 2

    # check for duplicates
    assert not spatial_differences['obsnme'].duplicated().any()

    # fill NaT (not a time) datetimes
    fill_nats(spatial_differences, perioddata)

    if outfile is not None:
        spatial_differences.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(spatial_differences):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(spatial_differences, str(outfile) + '.ins',
                          obsnme_column='obsnme',
                          simulated_obsval_column='sim_obsval', index=False)
    return spatial_differences


def get_temporal_differences(base_data, perioddata,
                             obs_values_col='obs_head',
                             sim_values_col='sim_head',
                             variable='head',
                             get_displacements=False,
                             displacement_from=None,
                             obsnme_date_suffix=True,
                             obsnme_suffix_format='%Y%m',
                             exclude_suffix='ss',
                             exclude_obs=None,
                             outfile=None,
                             write_ins=False):
    """Takes the base_data dataframe output by :func:`mfobs.obs.get_obs`,
    creates temporal difference observations. Optionally writes an output csvfile
    and a PEST instruction file.

    Parameters
    ----------
    base_data : DataFrame
        Head observation data with same column structure as
        output from :func:`mfobs.obs.get_obs`
    perioddata : DataFrame
        DataFrame with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    obs_values_col : str
        Column in ``base_data`` with observed values
    sim_values_col : str
        Column in `base_data`` with simulated equivalent values
    variable : str  {'head', 'flux', or other}
        Type of observation being processed. Simulated and observed values
        columns are named in the format 'sim_<variable>' and 'obs_<variable>',
        respectively. If there is no 'obgnme' column in ``base_data``,
        ``variable`` is also used as a default base group name.
    get_displacements : bool
        If True, compute the displacement of each observation from 
        a datum (specified by ``displacement_from``). If False, difference
        each observation with the previous observation.
        by default, False
    displacement_from : str or date-like
        Datum for computing displacements. Must be in a format that can be
        used for time slicing in pandas (e.g. '2010-01-01', which would result
        in displacements from the first observation on or after '2010-01-01' at each site,
        or None, which would result in displacements from the first observation 
        at each site. By default, None
    non-zero weighted observation
    obsnme_date_suffix : bool
        If true, give observations a date-based suffix. Otherwise, assign a 
        stress period-based suffix. In either case, the format of the suffix
        is controlled by obsnme_suffix_format.
        by default True
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes. Observation names are created following the format of
        <obsprefix>_<date or stress period suffix>. By default, ``'%Y%m'``,
        which would yield ``'202001'`` for a Jan, 2020 observation 
        (obsnme_date_suffix=True). If obsnme_date_suffix=False, obsnme_suffix_format
        should be a decimal format in the "new-style" string format
        (e.g. '{:03d}', which would yield ``'001'`` for stress period 1.)
    exclude_suffix : str or list-like
        Option to exclude observations from processing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    exclude_obs : list-like
        Sequence of observation names to exclude from return/written dataset. For example,
        if sequential head differences are also being computed, the first displacement observation
        after the reference observation will be a duplicate of the first sequential head difference
        observation. By default, None (no observations excluded).
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    period_diffs : DataFrame

    Notes
    -----
    Differences are computed by subtracting the previous time from the current,
    so a positive value indicates an increase.
    """
    base_data = base_data.copy()
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    
    # validation checks
    check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
                        function_name='get_head_obs', obsdata=base_data)
    if base_data.columns.duplicated().any():
        raise ValueError('Duplicate column names in base_data:\n'
                         f'{base_data.columns[base_data.columns.duplicated(keep=False)]}')
    
    # rename the observed and sim. eq. values columns
    if variable is not None:
        renames = {obs_values_col: f'obs_{variable}',
                sim_values_col: f'sim_{variable}',
                }
        obs_values_col = f'obs_{variable}'
        sim_values_col = f'sim_{variable}'
    else:
        renames = {obs_values_col: 'base_obsval',  # f'obs_{variable}',
                sim_values_col: 'base_sim_obsval',  # f'sim_{variable}',
                }
        obs_values_col = f'base_obsval'
        sim_values_col = f'base_sim_obsval'
    base_data.drop([f'obs_{variable}', f'sim_{variable}', 
                    'base_obsval', 'bas_sim_obsval'], axis=1, 
                   inplace=True, errors='ignore')
    base_data.rename(columns=renames, inplace=True)

    
    # rename the observed and sim. eq. values columns
    #renames = {obs_values_col: f'obs_{variable}',
    #           sim_values_col: f'sim_{variable}',
    #           }
    #base_data.drop([f'obs_{variable}', f'sim_{variable}'], axis=1, 
    #               inplace=True, errors='ignore')
    #base_data.rename(columns=renames, inplace=True)
    #obs_values_col = f'obs_{variable}'
    #sim_values_col = f'sim_{variable}'
    
    # only compute differences on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.isin(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()

    # group observations by site (prefix)
    sites = base_data.groupby('obsprefix')
    period_diffs = []
    
    # get index position for obsval columns,
    # after adding them if needed
    # (this allows assignment of new values purely by iloc, 
    #  which avoids setting on slices in loop below)
    if 'obsval' in base_data.columns:
        obsval_col_iloc = base_data.columns.get_loc('obsval')
    else:
        base_data['obsval'] = -1e30
        obsval_col_iloc = len(base_data.columns) - 1
    if 'sim_obsval' in base_data.columns:
        sim_obsval_col_iloc = base_data.columns.get_loc('sim_obsval')
    else:
        base_data['sim_obsval'] = -1e30
        sim_obsval_col_iloc = len(base_data.columns) - 1
    
    for site_no, values in sites:
        values = values.sort_values(by=['per']).copy()
        values.index = values['datetime']

        # compute the differences
        if get_displacements:
            values = values.sort_index().loc[displacement_from:].copy()
            
            # some sites may not have any measurements
            # after displacement datum; skip these
            if len(values) <= 1:
                continue
            values['obsval'] = values[obs_values_col] - \
                values[obs_values_col].iloc[0]
            values['sim_obsval'] = values[sim_values_col] - \
                values[sim_values_col].iloc[0]
            # assign np.nan to starting displacements (of 0) 
            # (so they get dropped later on, 
            # consistent with workflow for sequential difference obs)
            values.iloc[0, obsval_col_iloc] = np.nan
            values.iloc[0, sim_obsval_col_iloc] = np.nan
            #values['obsval'].iloc[0] = np.nan
            #values['sim_obsval'].iloc[0] = np.nan
        else:
            values['obsval'] = values[obs_values_col].diff()
            values['sim_obsval'] = values[sim_values_col].diff()

        # name the temporal difference obs as
        # <obsprefix>_<obsname1 suffix>d<obsname2 suffix>
        # where the obsval = obsname2 - obsname1
        obsnme = []
        for i, (idx, r) in enumerate(values.iterrows()):
            obsname2_suffix = ''
            if i > 0:
                if get_displacements:
                    obsname_2_loc = 0
                else:
                    obsname_2_loc = i - 1
                # date-based suffixes
                if obsnme_date_suffix:
                    obsname2_suffix = values.iloc[obsname_2_loc] \
                    ['datetime'].strftime(obsnme_suffix_format)
                # stress period-based suffixes
                else:
                    per = values.iloc[obsname_2_loc]['per']
                    obsname2_suffix = f"{per:{obsnme_suffix_format.strip('{:}')}}"
            obsnme.append('{}d{}'.format(r.obsnme, obsname2_suffix))
        values['obsnme'] = obsnme

        # todo: is there a general uncertainty approach for temporal differences that makes sense?

        period_diffs.append(values)
    
    period_diffs = pd.concat(period_diffs).reset_index(drop=True)
    
    # some checks to make sure the assignments from the above loop are correct
    assert period_diffs.iloc[:, sim_obsval_col_iloc].name == 'sim_obsval'
    assert period_diffs.iloc[:, obsval_col_iloc].name == 'obsval'
    assert not np.any(period_diffs[['obsval', 'sim_obsval']] == -1e30)
    
    period_diffs['datetime'] = pd.to_datetime(period_diffs['datetime'])

    if 'obgnme' not in period_diffs.columns:
        period_diffs['obgnme'] = variable
    
    if get_displacements:
        period_diffs['type'] = f'{variable} displacement'
        period_diffs['obgnme'] = [f'{g}_disp' for g in period_diffs['obgnme']]
    else:
        period_diffs['type'] = f'temporal {variable} difference'
        period_diffs['obgnme'] = [f'{g}_tdiff' for g in period_diffs['obgnme']]

    # drop some columns that aren't really valid; if they exist
    period_diffs.drop(['n'], axis=1, inplace=True, errors='ignore')

    # clean up columns
    cols = ['datetime', 'per', 'obsprefix', 'obsnme',
            f'obs_{variable}', f'sim_{variable}', 'screen_top', 'screen_botm', 'layer',
            'obsval', 'sim_obsval', 'obgnme', 'type']
    cols = [c for c in cols if c in period_diffs.columns]
    period_diffs = period_diffs[cols].copy()

    # drop observations with no difference (first observations at each site)
    period_diffs.dropna(axis=0, subset=['sim_obsval'], inplace=True)
    
    # drop any excluded obs
    if exclude_obs is not None:
        exclude_obs = set(exclude_obs)
        print(f"dropping {len(exclude_obs)} observations specified with exclude_obs")
        period_diffs = period_diffs.loc[~period_diffs['obsnme'].isin(exclude_obs)].copy()

    # fill NaT (not a time) datetimes
    fill_nats(period_diffs, perioddata)

    if outfile is not None:
        period_diffs.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(period_diffs):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(period_diffs, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
    return period_diffs


def get_annual_means(base_data, 
                     obsnme_suffix_format='%Y',
                     exclude_suffix='ss',
                     obgnme_suffix='annual-mean',
                     outfile=None,
                     write_ins=False):
    r"""Create observations of annual mean values for a set of 
    base observations. Means are computed for all columns with 
    floating point data.

    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_base_obs`. Must have the following columns,
        in addition to columns of floating point data to aggregate 
        (which can have any name):
        
        =================== =============================================================
        datetime            pandas datetimes, based on stress period end date
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name

        =================== =============================================================
        
        1) where obsprefix is assumed to be formatted as <site_no>-<variable> or simply <site_no>
        2) Assumed to be formatted <obsprefix>_<`obsnme_suffix_format`>
        
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes.
        By default '%Y', which returns the 4-digit year
    exclude_suffix : str or list-like
        Option to exclude observations from processing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    obgnme_suffix : str
        Create new observation group names by appending this suffix to existing
        obgnmes, for example <existing obgnme>_<obgnme_suffix>
        by default, 'annual-mean'
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    aggregated : DataFrame
        With the following columns (in addition to the data columns in `base_data`, 
        which now contain the aggregated values):
        
        =================== =============================================================
        datetime            year of annual mean, as datetime
        year                year of annual mean, as integer
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name\ :sup:`3`

        =================== =============================================================
        
        1) With format <site_no>-<variable> or simply <site_no> 
        2) With suffix formatted with `obsnme_suffix_format`
        3) With format of <original obgnme>_<obgnme_suffix>
            e.g. heads_annual-mean
        
        
    Notes
    -----
    """
    # validation checks
    #obsnme_date_suffix=True
    #check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
    #                    function_name='get_head_obs', obsdata=base_data)
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    
    # only compute statistics on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.isin(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()
    
    # only include times where there are both an observation and sim. equivalent
    base_data.dropna(subset=['obsval', 'sim_obsval'], inplace=True)
    
    grouped = base_data.groupby([base_data['site_no'], base_data['datetime'].dt.year])
    aggregated = grouped.first()
    data_cols = [c for c, dtype in base_data.dtypes.items() if 'float' in dtype.name]
    for c in data_cols:
        aggregated[c] = grouped[c].mean()
    aggregated['n'] = grouped.count().iloc[:, 0]
    aggregated.index.set_names(['site_no', 'year'], inplace=True)
    aggregated.reset_index(inplace=True)
    aggregated.sort_values(by=['site_no', 'year'], inplace=True)
        
    aggregated['obsnme'] = [f"{prefix}_{dt:{obsnme_suffix_format}}" 
                            for prefix, dt in zip(aggregated['obsprefix'], 
                                                  aggregated['datetime'])]
    aggregated['obgnme'] = [f"{prefix}_{obgnme_suffix}" 
                            for prefix in aggregated['obgnme']]
    cols = ['datetime', 'year', 'site_no', 'obsprefix', 'obsnme'] + data_cols + ['obgnme', 'n']
    aggregated = aggregated[cols]
    
    if outfile is not None:
        aggregated.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(aggregated):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(aggregated, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
    return aggregated
    
    
def get_monthly_means(base_data, 
                      obsnme_suffix_format='%Y%m',
                      exclude_suffix='ss',
                      obgnme_suffix='monthly-mean',
                      outfile=None,
                      write_ins=False):
    r"""Create observations of monthly mean values for a set of 
    base observations. Means are computed for all columns with 
    floating point data.

    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_base_obs`. Must have the following columns,
        in addition to columns of floating point data to aggregate 
        (which can have any name):
        
        =================== =============================================================
        datetime            pandas datetimes, based on stress period end date
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name

        =================== =============================================================
        
        1) where obsprefix is assumed to be formatted as <site_no>-<variable> or simply <site_no>
        2) Assumed to be formatted <obsprefix>_<`obsnme_suffix_format`>
        
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes.
        By default '%Y%m', which returns the year and month as consecutive integers, 
        e.g. 200101 for Jan, 2001.
    exclude_suffix : str or list-like
        Option to exclude observations from processing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    obgnme_suffix : str
        Create new observation group names by appending this suffix to existing
        obgnmes, for example <existing obgnme>_<obgnme_suffix>
        by default, 'annual-mean'
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    aggregated : DataFrame
        With the following columns (in addition to the data columns in `base_data`, 
        which now contain the aggregated values):
        
        =================== =============================================================
        datetime            aggregated dates as datetime objects
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name\ :sup:`3`

        =================== =============================================================
        
        1) With format <site_no>-<variable> or simply <site_no> 
        2) With suffix formatted with `obsnme_suffix_format`
        3) With format of <original obgnme>_<obgnme_suffix>
            e.g. heads_annual-mean
        
        
    Notes
    -----
    """
    # validation checks
    #obsnme_date_suffix=True
    #check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
    #                    function_name='get_head_obs', obsdata=base_data)
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    
    # only compute statistics on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.isin(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()
    
    # only include times where there are both an observation and sim. equivalent
    base_data.dropna(subset=['obsval', 'sim_obsval'], inplace=True)
    
    grouped = base_data.groupby([base_data['site_no'],
                                 base_data['datetime'].dt.year, 
                                 base_data['datetime'].dt.month])
    aggregated = grouped.first()
    data_cols = [c for c, dtype in base_data.dtypes.items() if 'float' in dtype.name]
    for c in data_cols:
        aggregated[c] = grouped[c].mean()
    aggregated['n'] = grouped.count().iloc[:, 0]
    aggregated.index.set_names(['site_no', 'year', 'month'], inplace=True)
    aggregated.reset_index(inplace=True)
    aggregated.sort_values(by=['site_no', 'year', 'month'], inplace=True)
        
    aggregated['obsnme'] = [f"{prefix}_{dt:{obsnme_suffix_format}}" 
                            for prefix, dt in zip(aggregated['obsprefix'], 
                                                  aggregated['datetime'])]
    aggregated['obgnme'] = [f"{prefix}_{obgnme_suffix}" 
                            for prefix in aggregated['obgnme']]
    cols = ['datetime', 'site_no', 'obsprefix', 'obsnme'] + data_cols + ['obgnme', 'n']
    aggregated = aggregated[cols]
    
    if outfile is not None:
        aggregated.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(aggregated):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(aggregated, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
    return aggregated


def get_mean_monthly(base_data, 
                     obsnme_suffix_format='%b',
                     exclude_suffix='ss',
                     obgnme_suffix='mean-monthly',
                     outfile=None,
                     write_ins=False):
    r"""Create observations of mean monthly, or means for months of the year
    (for example, the mean for Jan. across all years). Means are computed for 
    all columns with floating point data.

    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_base_obs`. Must have the following columns,
        in addition to columns of floating point data to aggregate 
        (which can have any name):
        
        =================== =============================================================
        datetime            pandas datetimes, based on stress period end date
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name

        =================== =============================================================
        
        1) where obsprefix is assumed to be formatted as <site_no>-<variable> or simply <site_no>
        2) Assumed to be formatted <obsprefix>_<`obsnme_suffix_format`>
        
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes.
        By default '%b', which returns the abbreviated month name (e.g. 'Jan').
    exclude_suffix : str or list-like
        Option to exclude observations from processing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    obgnme_suffix : str
        Create new observation group names by appending this suffix to existing
        obgnmes, for example <existing obgnme>_<obgnme_suffix>
        by default, 'annual-mean'
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    aggregated : DataFrame
        With the following columns (in addition to the data columns in `base_data`, 
        which now contain the aggregated values):
        
        =================== =============================================================
        datetime            datetimes that represent the mean of the period averaged
        month               month of monthly average values
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name\ :sup:`3`

        =================== =============================================================
        
        1) With format <site_no>-<variable> or simply <site_no> 
        2) With suffix formatted with `obsnme_suffix_format`
        3) With format of <original obgnme>_<obgnme_suffix>
            e.g. heads_annual-mean
        
        
    Notes
    -----
    """
    # validation checks
    #obsnme_date_suffix=True,
    #check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
    #                    function_name='get_head_obs', obsdata=base_data)
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    
    # only compute statistics on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.isin(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()
    
    # only include times where there are both an observation and sim. equivalent
    base_data.dropna(subset=['obsval', 'sim_obsval'], inplace=True)
    
    grouped = base_data.groupby([base_data['site_no'],
                                 base_data['datetime'].dt.month])
    aggregated = grouped.first()
    data_cols = [c for c, dtype in base_data.dtypes.items() if 'float' in dtype.name]
    for c in data_cols:
        aggregated[c] = grouped[c].mean()
    aggregated['n'] = grouped.count().iloc[:, 0]
    aggregated.index.set_names(['site_no', 'month'], inplace=True)
    aggregated.reset_index(inplace=True)
    aggregated.sort_values(by=['site_no', 'month'], inplace=True)
    # change month column to be name instead of number
    aggregated['month'] = [f"{dt:%B}" for dt in aggregated['datetime']]
        
    aggregated['obsnme'] = [f"{prefix}_{dt:{obsnme_suffix_format}}".lower() 
                            for prefix, dt in zip(aggregated['obsprefix'], 
                                                  aggregated['datetime'])]
    aggregated['obgnme'] = [f"{prefix}_{obgnme_suffix}" 
                            for prefix in aggregated['obgnme']]
    cols = ['datetime', 'month', 'site_no', 'obsprefix', 'obsnme'] + data_cols + ['obgnme', 'n']
    aggregated = aggregated[cols]
    
    if outfile is not None:
        aggregated.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(aggregated):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(aggregated, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
    return aggregated
    

def get_log10_observations(base_data, 
                           obsnme_prefix_variable='log10',
                           obsnme_suffix_format='%Y%m%d',
                           exclude_suffix='ss',
                           obgnme_suffix='log10',
                           fill_zeros_with=0,
                           outfile=None,
                           write_ins=False):
    r"""Create observations of log values. Log values are computed for 
    all columns with floating point data.

    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_base_obs`. Must have the following columns,
        in addition to columns of floating point data to aggregate 
        (which can have any name):
        
        =================== =============================================================
        datetime            pandas datetimes, based on stress period end date
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name

        =================== =============================================================
        
        1) where obsprefix is assumed to be formatted as <site_no>-<variable> or simply <site_no>
        2) Assumed to be formatted <obsprefix>_<`obsnme_suffix_format`>
    obsnme_prefix_variable : str
        Variable name indicating the type of observation. Is appended to the site number (site_no)
        to create new observation name prefixes (obsprefixes) in the format of <site_no>-<variable>.
        For example, by default, an observation prefix in base_data of '04026300-flow' or '04026300'
        (both for a site_no of '04026300'), would result in a new observation prefix of
        '04026300-flow-log10'. A log10 flow computed from an original flow observation named '04026300-flow_20221103' 
        or '04026300_20221103' (at site '04026300' on 2022/11/03) would be named '04026300-flow-log10_20221103'.
        by default, 'log10'.
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes.
        By default ''%Y%m%d' (e.g. 20010101 for Jan 1, 2001)
    exclude_suffix : str or list-like
        Option to exclude observations from processing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    obgnme_suffix : str
        Create new observation group names by appending this suffix to existing
        obgnmes, for example <existing obgnme>_<obgnme_suffix>
        by default, 'annual-mean'
    fill_zeros_with : numeric
        Fill value in log data for zero values in base_data.
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    log_base_data : DataFrame
        With the following columns (in addition to the data columns in `base_data`, 
        which now contain the aggregated values):
        
        =================== =============================================================
        datetime            observation dates as datetimes
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name\ :sup:`3`

        =================== =============================================================
        
        1) With format <site_no>-<variable> or simply <site_no> 
        2) With suffix formatted with `obsnme_suffix_format`
        3) With format of <original obgnme>_<obgnme_suffix>
            e.g. heads_annual-mean
        
        
    Notes
    -----
    """
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    
    # only compute statistics on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.isin(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()
    
    log_base_data = base_data.copy()
    data_cols = [c for c, dtype in base_data.dtypes.items() if 'float' in dtype.name]
    for c in data_cols:
        log_base_data[c] = np.log10(base_data[c])
        log_base_data.loc[base_data[c] == 0, c] = fill_zeros_with # handle zero values
    log_base_data['obsprefix'] = [f"{prefix}-{obsnme_prefix_variable}" 
                               for prefix in log_base_data['obsprefix']]
    log_base_data['obsnme'] = [f"{prefix}_{dt:{obsnme_suffix_format}}".lower() 
                            for prefix, dt in zip(log_base_data['obsprefix'], 
                                                  log_base_data['datetime'])]
    log_base_data['obgnme'] = [f"{prefix}_{obgnme_suffix}" 
                            for prefix in log_base_data['obgnme']]
    cols = ['datetime', 'site_no', 'obsprefix', 'obsnme'] + data_cols + ['obgnme']
    log_base_data = log_base_data[cols]
    
    if outfile is not None:
        log_base_data.fillna(-1e30).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(log_base_data):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(log_base_data, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
    return log_base_data
    

def get_baseflow_observations(base_data, 
                              obsnme_prefix_variable='baseflow',
                              obsnme_suffix_format='%Y%m%d',
                              exclude_suffix='ss',
                              obgnme_suffix='baseflow',
                              missing_obs_fill_value=0.,
                              outfile=None,
                              write_ins=False, **kwargs):
    r"""Create observations of base flow using the 
    BFI/Institute of Hydrology hydrograph separation method (:func:`mfobs.sep.ih_method`).
    
    
    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_base_obs`. Must have the following columns,
        in addition to columns of floating point data to aggregate 
        (which can have any name):
        
        =================== =============================================================
        datetime            pandas datetimes, based on stress period end date
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name

        =================== =============================================================
        
        1) where obsprefix is assumed to be formatted as <site_no>-<variable> or simply <site_no>
        2) Assumed to be formatted <obsprefix>_<`obsnme_suffix_format`>
    obsnme_prefix_variable : str
        Variable name indicating the type of observation. Is appended to the site number (site_no)
        to create new observation name prefixes (obsprefixes) in the format of <site_no>-<variable>.
        For example, by default, an observation prefix in base_data of '04026300-flow' or '04026300'
        (both for a site_no of '04026300'), would result in a new observation prefix of
        '04026300-baseflow'. A base flow computed from a total flow observation named '04026300-flow_20221103' 
        or '04026300_20221103' (at site '04026300' on 2022/11/03) would be named '04026300-baseflow_20221103'.
        by default, 'baseflow'.
    obsnme_suffix_format : str, optional
        Format for suffix of obsnmes.
        By default '%Y%m%d' (e.g. 20010101 for Jan 1, 2001)
    exclude_suffix : str or list-like
        Option to exclude observations from processing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    obgnme_suffix : str
        Create new observation group names by appending this suffix to existing
        obgnmes, for example <existing obgnme>_<obgnme_suffix>
        by default, 'baseflow'
    missing_obs_fill_value : float
        Baseflow observations are different from most other observation types in that
        they are produced by hydrograph separation, which can lead to different numbers
        of observations with changing conditions. For example, a given parameter set
        may produce a total flow hydrograph with a different number of ordinates (turning points),
        which in turn can change the interpolation of baseflow values between ordinates, potentially
        producing either 'extra' observations (not in the instruction file) or missing observations.
        Missing observations will be filled with `missing_obs_value`, so that PEST doesn't crash. A
        value of 0. is recommended, as missing observations are often associated with low or no-flow
        periods, especially near the start or end of a timeseries.
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False
    **kwargs : key-word arguments to :func:`mfobs.sep.ih_method`

    Returns
    -------
    bf_base_data : DataFrame
        With the following columns (in addition to the data columns in `base_data`, 
        which now contain the aggregated values):
        
        =================== =============================================================
        datetime            observation dates as datetimes
        site_no             unique site identifier
        obsprefix\ :sup:`1` prefix of observation name
        obsnme              observation name based on format of <obsprefix>_<suffix>\ :sup:`2`
        obgnme              observation group name\ :sup:`3`

        =================== =============================================================
        
        1) With format <site_no>-<variable> or simply <site_no> 
        2) With suffix formatted with `obsnme_suffix_format`
        3) With format of <original obgnme>_<obgnme_suffix>
            e.g. heads_annual-mean
        
        
    Notes
    -----
    """
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    
    # only compute statistics on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.isin(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()
    
    bf_base_data = base_data.copy()
    data_cols = [c for c, dtype in base_data.dtypes.items() if 'float' in dtype.name]
    bf_base_data.index = pd.to_datetime(bf_base_data['datetime'])
    
    dfs = []
    site_groups = bf_base_data.groupby('obsprefix')
    for obsprefix, group in site_groups:
        for c in data_cols:
            # series must be at least 2x the BFI block length (default=5)
            block_length = kwargs.get('block_length', 5)
            if len(group[c]) >= (block_length * 2):            
                results = ih_method(group[c], **kwargs)
                group[c] = results['QB']
        dfs.append(group)
    bf_base_data = pd.concat(dfs)

    # drop nan values
    bf_base_data.dropna(subset=['obsval', 'sim_obsval'], axis=0, inplace=True)
    bf_base_data['obsprefix'] = [f"{site_no}-{obsnme_prefix_variable}" 
                                 for site_no in bf_base_data['site_no']]
    bf_base_data['obsnme'] = [f"{prefix}_{dt:{obsnme_suffix_format}}".lower() 
                              for prefix, dt in zip(bf_base_data['obsprefix'],
                                                    bf_base_data['datetime'])]
    bf_base_data['obgnme'] = [f"{prefix}_{obgnme_suffix}" 
                            for prefix in bf_base_data['obgnme']]
    cols = ['datetime', 'site_no', 'obsprefix', 'obsnme'] + data_cols + ['obgnme']
    bf_base_data = bf_base_data[cols]
    
    if outfile is not None:
        # write the instruction file
        if write_ins:
            write_insfile(bf_base_data, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
        # otherwise, normalize observations to the instruction file
        else:
            insfile_obs = get_insfile_observations(str(outfile) + '.ins')
            
            missing_in_output = set(insfile_obs).difference(bf_base_data['obsnme'])
            extra_in_output = set(bf_base_data['obsnme']).difference(insfile_obs)
            
            # fill missing observations
            if any(missing_in_output):
                bf_base_data.index = bf_base_data['obsnme']
                
                # re-index the observation data to add rows for the missing obs
                reindexed = bf_base_data.reindex(insfile_obs)
                # identify rows containing missing obs
                filled = reindexed['sim_obsval'].isna()
                # refill columns for the missing obs
                reindexed['obsnme'] = reindexed.index
                reindexed.loc[filled, 'datetime'] = [pd.Timestamp(s.split('_')[1].split('-')[0])
                                                     for s in reindexed.loc[filled, 'obsnme']]
                reindexed.loc[filled, 'site_no'] = [s.split('_')[0].split('-')[0]
                                        for s in reindexed.loc[filled, 'obsnme']]
                reindexed.loc[filled, 'obsprefix'] = [s.split('_')[0]
                        for s in reindexed.loc[filled, 'obsnme']]
                reindexed.loc[filled, 'obgnme'] = 'filled-bf'
                reindexed.loc[filled, data_cols] = missing_obs_fill_value
                assert not reindexed.isna().any().any()
                bf_base_data = reindexed
            # drop any extra observations
            if any(extra_in_output):
                bf_base_data = bf_base_data.loc[~bf_base_data.index.isin(extra_in_output)]
            
        bf_base_data.to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(bf_base_data):,} observations to {outfile}')
            
    return bf_base_data