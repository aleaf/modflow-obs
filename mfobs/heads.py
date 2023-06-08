"""
Functions for processing head observations
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from mfobs.checks import check_obsnme_suffix
from mfobs.fileio import load_array, write_insfile
from mfobs.modflow import get_mf6_single_variable_obs, get_transmissivities
from mfobs.obs import aggregrate_to_period
from mfobs.utils import fill_nats, set_period_start_end_dates


def get_head_obs(perioddata, modelgrid_transform, model_output_file,
                 observed_values_file,
                 gwf_obs_input_file,
                 observed_values_metadata_file=None,
                 variable_name='head',
                 observed_values_site_id_col='obsprefix',
                 observed_values_datetime_col='datetime',
                 obsnme_date_suffix=True,
                 obsnme_suffix_format='%Y%m',
                 observed_values_obsval_col='obsval',
                 observed_values_x_col='x',
                 observed_values_y_col='y',
                 observed_values_screen_top_col='screen_top',
                 observed_values_screen_botm_col='screen_botm',
                 observed_values_layer_col=None,
                 observed_values_group_column='obgnme',
                 observed_values_unc_column='uncertainty',
                 aggregrate_observed_values_by='mean',
                 gwf_obs_block=None,
                 drop_groups=None,
                 hk_arrays=None, top_array=None, botm_arrays=None,
                 label_period_as_steady_state=None, steady_state_period_start=None,
                 steady_state_period_end=None, forecast_sites=None,
                 forecast_start_date=None, forecast_end_date=None,
                 forecasts_only=False, forecast_sites_only=False,
                 write_ins=False, outfile=None):
    """Post-processes model output to be read by PEST, and optionally,
    writes a corresponding PEST instruction file. Reads model output
    using get_mf6_single_variable_obs(). General paradigm is to include all model
    layers in the MODFLOW input for each observation, and then post-process the model
    results to a single value by computing a transmissivity-weighted average.

    Observation names to match observed values to their simulated equivalents are constructed
    in the format of <obsprefix>_<date suffix>, where obsprefix is a site identifier taken
    from the ``observed_values_site_id_col`` in ``observed_values_file``. In creating
    observation names for MODFLOW output, the column names in the observation CSV output
    are used for the prefixes. Therefore, the identifiers in ``observed_values_site_id_col``
    should correspond to observations in the MODFLOW observation input. The date suffix
    is formatted using the ``obsnme_date_suffix_format`` parameter, which is also
    passed to :func:`~mfobs.modflow.get_mf6_single_variable_obs` for assigning observation
    names to the MODFLOW observation output.

    Optionally, a model stress period can be labeled as steady-state (``label_period_as_steady_state``),
    representing average conditions over a time period bracked by a ``steady_state_period_start`` and
    ``steady_state_period_end``. In this case, the simulated values for the labeled stress period are
    matched to average values for the steady-state time period.

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    modelgrid_transform : str
        An `affine.Affine <https://github.com/sgillies/affine>`_ object describing the orientation
        of the model grid. Modflow-setup :class:`~mfsetup.grid.MFsetupGrid` have this attached
        via the :meth:`~mfsetup.grid.MFsetupGrid.transform` property. Example::

            modelgrid_transform=affine.Affine(1000.0, 0.0, 500955,
                                              0.0, -1000.0, 1205285)

        for a uniform spacing of 1000 and upper left corner of 500955, 1205285
        with a rotation of 45 degrees, counter-clockwise about the upper left corner::

            modelgrid_transform=affine.Affine(1000.0, 0.0, 500955,
                                              0.0, -1000.0, 1205285).rotation(45.)

        An ``affine.Affine`` instance can also be created from a
        `Modflow-setup <https://github.com/aleaf/modflow-setup>`_
        grid JSON file via the :func:`~mfobs.modflow.get_modelgrid_transform` function.

    model_output_file : str
        Modflow-6 head observation CSV output file.
        Read by :func:`~mfobs.modflow.get_mf6_single_variable_obs`.
    observed_values_file : str or DataFrame
        CSV file or DataFrame with observed values. Must have the following columns
        (default names are shown, other names can be specified with
        observed_values_**_col variables below):

        ============= ========================
        site_id       site identifier
        datetime      date/time of observation
        obsval        observed value
        ============= ========================

        can optionally include these columns, or this information can be supplied
        in an observed_values_metadata_file, which will be joined on site_id

        ============= ========================
        x             x location
        y             y location
        screen_top    screen top elevation
        screen_botm   screen bottom elevation
        ============= ========================

        If supplied, observation group and uncertainty information will be
        passed through to the output ``base_data`` DataFrame:

        ============= ==================================
        obgnme         observation group
        uncertainty   estimated measurement uncertainty
        ============= ==================================

        Locations and screen tops and bottoms are assumed to be in the same
        CRS and length units as the model.

    observed_values_metadata_file : str, optional
        Site information for the observed values timeseries. Should include a
        `site_id` column that is the same as observed_values_site_id_col, and any of
        the following columns that are not in the observed_values_file:

        ============= ========================
        x             x location
        y             y location
        screen_top    screen top elevation
        screen_botm   screen bottom elevation
        ============= ========================

    gwf_obs_input_file : str
        Input file to MODFLOW-6 observation utility (contains layer information).
    variable_name : str, optional
        Column with simulated output will be named "sim_<variable_name",
        by default 'head'
    observed_values_site_id_col : str, optional
        Column name in observed_values_file with site identifiers,
        by default 'obsprefix'
    observed_values_datetime_col : str, optional
        Column name in observed_values_file with observation date/times,
        by default 'datetime'
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
    observed_values_obsval_col : str, optional
        Column name in observed_values_file with observed values,
        by default 'obsval'
    observed_values_x_col : str, optional
        Column name in observed_values_file with x-coordinates,
        by default 'x'
    observed_values_y_col : str, optional
        Column name in observed_values_file with y-coordinates,
        by default 'y'
    observed_values_screen_top_col : str, optional
        Column name in observed_values_file with screen top elevations,
        by default 'screen_top'
    observed_values_screen_botm_col : str, optional
        Column name in observed_values_file with screen bottom elevations,
        by default 'screen_botm'
    observed_values_layer_col : str, optional
        As an alternative to providing screen tops and bottoms, the model layer
        for each observation can be specified directly via a layer column
        of zero-based layer numbers.
        by default None
    observed_values_group_column : str, optional
        Column name in observed_values_file with observation group information.
        Passed through to output ``base_data`` DataFrame, otherwise not required.
        by default 'obgnme'
    observed_values_unc_column : str, optional
        Column name in observed_values_file with observation uncertainty values.
        Passed through to output ``base_data`` DataFrame, otherwise not required.
        by default 'uncertainty'
    aggregrate_observed_values_by : str
        Method for aggregating observed values to the model stress periods,
        if there are multiple observed values in a stress period. Can be any
        of the method calls on the pandas
        `Resampler <https://pandas.pydata.org/pandas-docs/stable/reference/resampling.html>`_
        object. By default, 'mean'
    gwf_obs_block : None or int
        Argument to read a specific observation block or all blocks from GWF 
        observation utility file. Value of None returns observations from all 
        blocks. Integer value returns obs from a specifc block, in (zero-based)
        order, from top to bottom. For example, a value of 0 would return the 
        first obs block, value of 1 would return the second obs block, and so 
        on. Modflow-setup writes all of the observation input to a single block, 
        but observation input can be broken out into multiple blocks (one per 
        file). by default, None (All blocks)
    drop_groups : sequence, optional
        Observation groups to exclude from output, by default None
    hk_arrays : list-like of pathlikes or ndarray, optional
        File paths to text arrays with hydraulic conductivity values
        (ordered by model layer). Used in the transmissivity-weighted averaging.
        by default None
    top_array : str, pathlike or ndarray, optional
        File paths to text array with model top elevations.
        Used in the transmissivity-weighted averaging.
        by default None
    botm_arrays : str, pathlike or ndarray, optional
        File paths to text arrays with model cell bottom elevations.
        (ordered by model layer). Used in the transmissivity-weighted averaging.
        by default None
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
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    base_data : DataFrame
        With the following columns:

        ===================== ====================================================
        datetime              pandas datetimes for the start of each stress period
        per                   model stress period
        obsprefix             observation site identifier
        obsnme                observation name based on format of <obsprefix>_'%Y%m'
        obs_<variable_name>   observed values
        sim_<variable_name>   simulated observation equivalents
        screen_top            screen top elevation
        screen_botm           screen bottom elevation
        ===================== ====================================================

        Example observation names:

        site1000_202001, for a Jan. 2020 observation at site1000 (obsnme_date_suffix=True)
        
        site1000_001, for a stress period 1 observation at site1000 (obsnme_date_suffix=False)

        a steady-state stress period specified with label_period_as_steady_state 
        is given the suffix of 'ss'
        e.g. site1000_ss

    Notes
    -----
    All observation names and observation prefixes are converted to lower case
    to avoid potential case issues.


    """
    # validation checks
    check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
                        function_name='get_head_obs')
    
    outpath = Path('.')
    if outfile is not None:
        outpath = Path(outfile).parent

    obs_values_column = 'obsval'  # + variable_name  # output column with observed values
    sim_values_column = 'sim_obsval' # + variable_name  # output column with simulated equivalents to observed values

    perioddata = perioddata.copy()
    set_period_start_end_dates(perioddata)
    perioddata.index = perioddata.per
    results = get_mf6_single_variable_obs(perioddata, model_output_file=model_output_file,
                                          gwf_obs_input_file=gwf_obs_input_file,
                                          #variable_name=variable_name,
                                          #obsnme_date_suffix=obsnme_date_suffix,
                                          #obsnme_suffix_format=obsnme_suffix_format,
                                          #label_period_as_steady_state=label_period_as_steady_state,
                                          gwf_obs_block=gwf_obs_block
                                          )

    # rename columns to their defaults
    renames = {#observed_values_site_id_col: 'obsprefix',
               observed_values_datetime_col: 'datetime',
               observed_values_x_col: 'x',
               observed_values_y_col: 'y',
               observed_values_screen_top_col: 'screen_top',
               observed_values_screen_botm_col: 'screen_botm',
               observed_values_layer_col: 'layer',
               observed_values_group_column: 'obgnme',
               observed_values_unc_column: 'uncertainty'
               }

    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file,
                               dtype={observed_values_site_id_col: object})
    else:
        observed = observed_values_file
    if len(observed) == 0:
        raise ValueError("No observed values to process!")
    observed.rename(columns=renames, inplace=True)
    observed['obsprefix'] = observed[observed_values_site_id_col]

    # read in the observed values metadata
    if observed_values_metadata_file is not None:
        if not isinstance(observed_values_metadata_file, pd.DataFrame):
            metadata = pd.read_csv(observed_values_metadata_file,
                               dtype={observed_values_site_id_col: object})
        else:
            metadata = observed_values_metadata_file
        metadata.rename(columns=renames, inplace=True)
        metadata['obsprefix'] = metadata[observed_values_site_id_col]

        # join the metadata to the observed data
        metadata.index = metadata['obsprefix'].values
        observed.index = observed['obsprefix'].values
        join_cols = [c for c in ['screen_top', 'screen_botm', 'x', 'y', 'layer']
                     if c in metadata.columns]
        observed = observed.join(metadata[join_cols], rsuffix='_m')
        assert not observed[['x', 'y']].isna().any().any()

    # convert obs names and prefixes to lower case
    observed['obsprefix'] = observed['obsprefix'].str.lower()
    
    # make a dictionary of site metadata for possible use later
    temp = observed.copy()
    temp.index = temp['obsprefix'].str.lower()
    site_info_dict = temp.to_dict()
    # delete some columns from the observed values file
    # which result in values assigned by the function 
    # later being overwritten with nans
    del_cols = ['datetime', 'per']
    for col in del_cols:
        if col in site_info_dict:
            del site_info_dict[col]
    del temp

    # cast datetimes to pandas datetimes
    # (observed data may not have a datetime col. if model is steady-state)
    if 'datetime' in observed.columns:
        observed['datetime'] = pd.to_datetime(observed['datetime'])
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
    
    # no forecast observations;
    # drop sites that don't have an observed/sim equivalent pair
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
    if len(no_info_sites) > 0:
        print('Dropping {} sites with no information'.format(len(no_info_sites)))
        dropped_obs_outfile = outpath / 'dropped_head_observation_sites.csv'
        results.loc[results.obsprefix.isin(no_info_sites)].to_csv(dropped_obs_outfile,
                                                              index=False)
        results = results.loc[~results.obsprefix.isin(no_info_sites)].copy()
        observed = observed.loc[~observed.obsprefix.isin(no_info_sites)].copy()

    # get_mf6_single_variable_obs returns values for each layer
    # collapse these into one value for each location, time
    # by taking the transmissivity-weighted average
    if observed_values_layer_col is None:
        if isinstance(hk_arrays[0], str) or isinstance(hk_arrays[0], Path):
            hk = load_array(hk_arrays)
        else:
            hk = hk_arrays
        if isinstance(top_array, str) or isinstance(top_array, Path):
            top = load_array(top_array)
        else:
            top = top_array
        if isinstance(botm_arrays[0], str) or isinstance(botm_arrays[0], Path):
            botm = load_array(botm_arrays)
        else:
            botm = botm_arrays

    # get the x and y location and open interval corresponding to each head observation
    x = dict(zip(observed['obsprefix'], observed['x']))
    y = dict(zip(observed['obsprefix'], observed['y']))
    results['x'] = [x[obsprefix] for obsprefix in results.obsprefix]
    results['y'] = [y[obsprefix] for obsprefix in results.obsprefix]

    # get head values based on T-weighted average of open interval
    if observed_values_layer_col is None:
        if 'screen_top' in observed.columns:
            screen_top = dict(zip(observed['obsprefix'], observed['screen_top']))
            if 'screen_botm' not in observed.columns:
                screen_botm = screen_top
        elif 'screen_botm' not in observed.columns:
            raise ValueError('Observed values need at least a screen top or screen bottom column')
        if 'screen_botm' in observed.columns:
            screen_botm = dict(zip(observed['obsprefix'], observed['screen_botm']))
            if 'screen_top' not in observed.columns:
                screen_top = screen_botm
        elif 'screen_top' not in observed.columns:
            raise ValueError('Observed values need at least a screen top or screen bottom column')
        results['screen_top'] = [screen_top[obsprefix] for obsprefix in results.obsprefix]
        results['screen_botm'] = [screen_botm[obsprefix] for obsprefix in results.obsprefix]

    # for each model stress period, get the simulated values
    # and the observed equivalents
    observed.index = pd.to_datetime(observed.datetime)
    results.index = pd.to_datetime(results.datetime)
    
    # integer column for stress period- or timestep-based obsnme suffixes
    # timestep-based observations
    if 'timestep' in perioddata.columns:
        perioddata['unique_timestep'] = list(range(len(perioddata)))
        per_column = 'unique_timestep'
        times = dict(zip(perioddata['time'], perioddata['unique_timestep']))
        results[per_column] = [times[t] for t in results['time']]
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
        start, end = str(r['start_datetime']), str(r['end_datetime'])
        if start[:4] == '2010':
            j=2
            
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
                start = str(steady_state_period_start)
            if steady_state_period_end is not None:
                end = str(steady_state_period_end)
        # don't process observations for a steady-state period unless 
        # it is explicitly labeled as such and given a representative date range
        #elif r['steady']:
        #    continue
        
        #aggregrate_observed_values_method = 'mean'
        
        #observed_in_period_rs = aggregrate_to_period(
        #    observed, start, end, 
        #    aggregrate_observed_values_method=aggregrate_observed_values_method,
        #    obsnme_suffix=suffix)
        #if observed_in_period_rs is None:
        #    if per_column == 'per':
        #        warnings.warn(('Stress period {}: No observations between start and '
        #                        'end dates of {} and {}!'.format(r['per'], start, end)))
        #    continue

        # for now, require results to have period or unique timestep column
        # otherwise, if slicing by datetimes, can run into problem of
        # for example, initial steady-state periods being included 
        # with first transient period
        # leading to a duplicate index error when trying to pivot below
        #data = results.loc[start:end].copy()
        data = results.loc[results[per_column] == r[per_column]].copy()
        if len(data) == 0:
            continue
        # kludge to assign obsnmes to model results
        # until head obs handling gets refactored into obs.get_base_obs
        data['obsnme'] = ['{}_{}'.format(prefix.lower(), suffix)
                          for prefix in data.obsprefix]
        data.index = data['obsnme']
        
        observed_in_period = observed.sort_index().loc[start:end].reset_index(drop=True)
        
        # No forecast observations and no observed values in period
        if forecast_sites is None and len(observed_in_period) == 0:
            warnings.warn(('Stress period {}: No observations between start and '
                           'end dates of {} and {}!'.format(r['per'], start, end)))
            continue
        
        # If there are forecast sites and observed data in this period
        elif len(observed_in_period) > 0:
        
            observed_in_period.sort_values(by=['obsprefix', 'datetime'], inplace=True)
            if 'n' not in observed_in_period.columns:
                observed_in_period['n'] = 1
            by_site = observed_in_period.groupby('obsprefix')
            observed_in_period_rs = by_site.first()
            data_cols = [c for c, dtype in observed_in_period.dtypes.items() 
                         if 'float' in dtype.name]
            for c in data_cols:
                observed_in_period_rs[c] = getattr(by_site[c], aggregrate_observed_values_by)()
            observed_in_period_rs['n'] = by_site.n.sum()
            observed_in_period_rs['datetime'] = pd.Timestamp(end)
            if observed_in_period_rs[observed_values_obsval_col].isna().any():
                raise ValueError(f'Nan {observed_values_obsval_col} values in observation data')
            observed_in_period_rs.reset_index(inplace=True)  # put obsprefix back

            missing_cols = set(observed_in_period.columns).difference(observed_in_period_rs.columns)
            for col in missing_cols:
                observed_in_period_rs[col] = by_site[col].first().values
            observed_in_period_rs = observed_in_period_rs[observed_in_period.columns]
            obsnames = ['{}_{}'.format(prefix.lower(), suffix)
                        for prefix in observed_in_period_rs.obsprefix]
            observed_in_period_rs['obsnme'] = obsnames
            observed_in_period_rs.index = observed_in_period_rs['obsnme']
            
        # Forecast sites, but no observed data
        else:
            observed_in_period_rs = pd.DataFrame(columns=observed.columns)

        # Simulated equivalents
        # Option to get head values based on T-weighted average of open interval        
        if observed_values_layer_col is None:
            # get a n layers x n sites array of simulated head observations
            data = data.reset_index(drop=True)
            heads_2d = data.pivot(columns='layer', values='sim_obsval', index='obsnme').T.values
            obsnme = data.pivot(columns='layer', values='obsnme', index='obsnme').index.tolist()

            # layers containing head values
            # (in case there are any layers not represented in the head data)
            head_layers = data['layer'].unique()
            if heads_2d.shape[0] != botm.shape[0]:
                heads_rs = np.ones((botm.shape[0], heads_2d.shape[1])) * np.nan
                heads_rs[head_layers] = heads_2d
                heads_2d = heads_rs

            # x, y, screen_top and screen_botm have one value for each site
            kwargs = {}
            for arg in 'x', 'y', 'screen_top', 'screen_botm':
                # pivot data to nsites rows x nlay columns
                # positions without data are filled with nans
                pivoted = data.pivot(columns='layer', values=arg, index='obsnme')
                # reduce pivoted data to just one value per site by taking the mean
                # (values should be the same across columns, which represent layers)
                kwargs[arg] = pivoted.mean(axis=1).values

            # get the transmissivity associated with each head obs
            T = get_transmissivities(heads_2d, hk, top, botm,
                                     modelgrid_transform=modelgrid_transform, **kwargs
                                     )

            # compute transmissivity-weighted average heads
            Tr_frac = T / T.sum(axis=0)
            Tr_frac_df = pd.DataFrame(Tr_frac.transpose())
            Tr_frac_df['obsnme'] = obsnme
            Tr_frac_df.to_csv(outpath / 'obs_layer_transmissivities.csv', float_format='%.2f')
            mean_t_weighted_heads = np.nansum((heads_2d * Tr_frac), axis=0)

            # in some cases, the open interval might be mis-matched with the layering
            # for example, an open interval might be primarily in layer 4,
            # in a location where layer 5 is the only active layer
            # this would result in a mean_t_weighted_heads value of 0
            # (from the zero transmissivity in that layer)
            # fill these instances with the mean of any valid heads at those locations
            mean_heads = np.nanmean(heads_2d, axis=0)
            misaligned = mean_t_weighted_heads == 0
            mean_t_weighted_heads[misaligned] = mean_heads[misaligned]

            # verify that there are no nans in the extracted head values (one per obs)
            assert not np.any(np.isnan(mean_t_weighted_heads))

            # add the simulated heads onto the list for all periods
            mean_t_weighted_heads_df = pd.DataFrame({sim_values_column: mean_t_weighted_heads}, 
                                                    index=obsnme)
            if forecast_sites is not None:
                observed_in_period_rs = observed_in_period_rs.reindex(obsnme)
                obsprefix = observed_in_period_rs.index.str.split('_', expand=True).levels[0]
                observed_in_period_rs['obsprefix'] = obsprefix
                observed_in_period_rs['datetime'] = data['datetime'].values[0]

            observed_in_period_rs[sim_values_column] = mean_t_weighted_heads_df[sim_values_column]

        # Alternative option to get head values for specified layers
        # (or closest layer if the specified layer doesn't have obs output)
        else:
            any_simulated_obs = data.obsnme.isin(observed_in_period_rs.obsnme).any()
            if not any_simulated_obs:
                continue
            sim_values = []
            for obsnme, layer in zip(observed_in_period_rs.obsnme, observed_in_period_rs.layer):
                obsnme_results = data.loc[obsnme]
                
                # if a DataFrame (with simulated values for multiple layers) is returned
                if len(obsnme_results.shape) == 2:
                    layer = obsnme_results.iloc[np.argmin(obsnme_results.layer - layer)]['layer']
                    sim_value = obsnme_results.iloc[layer][sim_values_column]
                # Series (row) in results DataFrame with single simulated value
                else:
                    sim_value = obsnme_results[sim_values_column]
                sim_values.append(sim_value)
            observed_in_period_rs[sim_values_column] = sim_values

        # add stress period and observed values
        observed_in_period_rs['per'] = r['per']
        observed_in_period_rs[obs_values_column] = observed_in_period_rs[observed_values_obsval_col]
        observed_simulated_combined.append(observed_in_period_rs)

    # Combined DataFrame of observed heads and simulated equivalents
    head_obs = pd.concat(observed_simulated_combined)

    # raise an error if there are duplicates- reindexing below will fail if this is the case
    if head_obs.index.duplicated().any():
        msg = ('The following observations have duplicate names. There should only be'
               'one observation per site, for each time period implied by the '
               'obsnme_date_suffix_format parameter.\n{}'
               .format(head_obs.loc[head_obs.duplicated()]))
        raise ValueError(msg)

    # drop any observations in specified groups
    # (e.g. lake stages that should be compared with lake package output)
    if drop_groups is not None and 'obgnme' in head_obs.columns:
        head_obs = head_obs.loc[~head_obs.obgnme.isin(drop_groups)].copy()

    # add standard obsval and obgmne columns
    #head_obs['obsval'] = head_obs[obs_values_column]
    if 'obgnme' not in head_obs.columns:
        head_obs['obgnme'] = variable_name

    # fill forecast obs with site info from observed dataframe
    if forecast_sites is not None:
        for k, v in site_info_dict.items():
            head_obs[k] = [v[p] for p in head_obs['obsprefix']]
        head_obs['obsnme'] = head_obs.index
    else:
        # nans are where sites don't have observation values for that period
        # or sites that are in other model (inset or parent)
        head_obs.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # label forecasts in own group
    if forecast_sites is not None:
        is_forecast = head_obs[obs_values_column].isna()
        head_obs.loc[is_forecast, 'obgnme'] += '-forecast'
    
        # cull forecasts to specified date window
        # and specific sites (if specified)
        keep_forecasts = is_forecast.copy()  #np.array([True] * len(head_obs))
        if forecast_start_date is not None:
            keep_forecasts = (head_obs['datetime'] >= forecast_start_date)
        if forecast_end_date is not None:
            keep_forecasts &= (head_obs['datetime'] <= forecast_end_date)
        #drop = drop & is_forecast
        #head_obs = head_obs.loc[~drop].copy()
        #is_forecast = head_obs[obs_values_column].isna()
        if forecast_sites != 'all':
            keep_forecasts &= head_obs['obsprefix'].isin(forecast_sites)
        # option to only include forecast obs
        # (those without observed equivalents)
        if forecasts_only:
            keep = keep_forecasts
        else:
            keep = keep_forecasts | ~is_forecast
        # option to only include output from designated forecast sites
        if forecast_sites_only:
            keep = keep & head_obs['obsprefix'].isin(forecast_sites)
        head_obs = head_obs.loc[keep].copy()

    # reorder the columns
    columns = ['datetime', 'per', 'site_no', 'obsprefix', 'obsnme', 
               obs_values_column, sim_values_column,
               'n', 'uncertainty', 'screen_top', 'screen_botm', 'layer', 'obgnme']
    columns = [c for c in columns if c in head_obs.columns]
    head_obs = head_obs[columns].copy()
    if 'layer' in columns:
        head_obs['layer'] = head_obs['layer'].astype(int)

    # fill NaT (not a time) datetimes
    fill_nats(head_obs, perioddata)

    head_obs.sort_values(by=['obsprefix', 'per'], inplace=True)
    if outfile is not None:
        head_obs.fillna(-9999).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(head_obs):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(head_obs, str(outfile) + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=sim_values_column, index=False)
    return head_obs
