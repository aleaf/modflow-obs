"""
Functions for processing head observations
"""
import warnings
import numpy as np
import pandas as pd
from mfobs.fileio import load_array, write_insfile
from mfobs.modflow import get_mf6_single_variable_obs, get_transmissivities
from mfobs.utils import fill_nats


def get_head_obs(perioddata, modelgrid_transform, model_output_file,
                 observed_values_file,
                 gwf_obs_input_file,
                 observed_values_metadata_file=None,
                 variable_name='head',
                 outfile='processed_head_obs.dat',
                 observed_values_site_id_col='obsprefix',
                 observed_values_datetime_col='datetime',
                 obsnme_date_suffix_format='%Y%m',
                 observed_values_obsval_col='obsval',
                 observed_values_x_col='x',
                 observed_values_y_col='y',
                 observed_values_screen_top_col='screen_top',
                 observed_values_screen_botm_col='screen_botm',
                 observed_values_group_column='group',
                 observed_values_unc_column='uncertainty',
                 drop_groups=None,
                 hk_arrays=None, top_array=None, botm_arrays=None,
                 label_period_as_steady_state=None, steady_state_period_start=None,
                 steady_state_period_end=None,
                 write_ins=False):
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

    Steady-state observations are created

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

        If supplied, observation group an uncertainty information will be
        passed through to the output ``head_obs`` DataFrame:

        ============= ==================================
        group         observation group
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
    outfile : str, optional
        Output file of values to be read by PEST,
        by default 'processed_head_obs.dat'
    observed_values_site_id_col : str, optional
        Column name in observed_values_file with site identifiers,
        by default 'obsprefix'
    observed_values_datetime_col : str, optional
        Column name in observed_values_file with observation date/times,
        by default 'datetime'
    obsnme_date_suffix_format : str, optional
        Format for date suffix of obsnmes. By default, '%Y%m',
        which would yield '202001' for a Jan, 2020 observation.
        Observation names are created following the format of
        <obsprefix>_<date suffix>
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
    observed_values_group_column : str, optional
        Column name in observed_values_file with observation group information.
        Passed through to output ``head_obs`` DataFrame, otherwise not required.
        by default 'group'
    observed_values_unc_column : str, optional
        Column name in observed_values_file with observation uncertainty values.
        Passed through to output ``head_obs`` DataFrame, otherwise not required.
        by default 'uncertainty'
    drop_groups : sequence, optional
        Observation groups to exclude from output, by default None
    hk_arrays : list-like, optional
        File paths to text arrays with hydraulic conductivity values
        (ordered by model layer). Used in the transmissivity-weighted averaging.
        by default None
    top_array : str, optional
        File paths to text array with model top elevations.
        Used in the transmissivity-weighted averaging.
        by default None
    botm_arrays : str, optional
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
        By default None, in which case no steady-state observatons are created.
    steady_state_period_end : str, optional
        End date for the period representing steady-state conditions.
        By default None, in which case no steady-state observatons are created.
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    head_obs : DataFrame
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

        site1000_202001, for a Jan. 2020 observation at site1000

        steady-state stress periods are given the suffix of 'ss'
        e.g. site1000_ss

    Notes
    -----
    All observation names and observation prefixes are converted to lower case
    to avoid potential case issues.


    """
    obs_values_column = 'obs_' + variable_name  # output column with observed values
    sim_values_column = 'sim_' + variable_name  # output column with simulated equivalents to observed values

    perioddata = perioddata.copy()
    results = get_mf6_single_variable_obs(perioddata, model_output_file=model_output_file,
                                          gwf_obs_input_file=gwf_obs_input_file,
                                          variable_name=variable_name,
                                          obsnme_date_suffix_format=obsnme_date_suffix_format,
                                          label_period_as_steady_state=label_period_as_steady_state)

    # rename required columns to their defaults
    renames = {observed_values_site_id_col: 'obsprefix',
               observed_values_datetime_col: 'datetime',
               observed_values_x_col: 'x',
               observed_values_y_col: 'y',
               observed_values_screen_top_col: 'screen_top',
               observed_values_screen_botm_col: 'screen_botm',
               observed_values_group_column: 'group',
               observed_values_unc_column: 'uncertainty'
               }

    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file)
    else:
        observed = observed_values_file
    observed.rename(columns=renames, inplace=True)

    # read in the observed values metadata
    if observed_values_metadata_file is not None:
        if not isinstance(observed_values_metadata_file, pd.DataFrame):
            metadata = pd.read_csv(observed_values_metadata_file)
        else:
            metadata = observed_values_file
        metadata.rename(columns=renames, inplace=True)

        # join the metadata to the observed data
        metadata.index = metadata['obsprefix'].values
        observed.index = observed['obsprefix'].values
        observed = observed.join(metadata[['screen_top', 'screen_botm', 'x', 'y']])

    # convert obs names and prefixes to lower case
    observed['obsprefix'] = observed['obsprefix'].str.lower()

    # make unique observation names for each observation,
    # using obsprefix and date
    observed['datetime'] = pd.to_datetime(observed['datetime'])
    obsnames = ['{}_{}'.format(prefix.lower(),
                               timestamp.strftime(obsnme_date_suffix_format))
                for prefix, timestamp in zip(observed.obsprefix,
                                             observed.datetime)]
    observed['obsnme'] = obsnames

    # make steady-state observations
    if steady_state_period_start is not None and steady_state_period_end is not None:
        observed.index = observed.datetime
        steady_per = observed.loc[steady_state_period_start: steady_state_period_end]
        if len(steady_per) == 0:
            warnings.warn(('No observations between steady-state start and '
                           'end dates of {} and {}!'.format(steady_state_period_start,
                                                            steady_state_period_end)))
        else:
            steady_obs = steady_per.groupby('obsprefix').mean().reset_index()
            missing = set(observed.columns).difference({'datetime'}.union(steady_obs))
            for col in missing:
                steady_obs[col] = steady_per.groupby('obsprefix')[col].first().values
            steady_obs['obsnme'] = ['{}_ss'.format(prefix.lower()) for prefix in steady_obs.obsprefix]
            steady_obs['n'] = steady_per.groupby('obsprefix').n.count().values
            observed = observed.append(steady_obs)

    # index by observation name (location and date)
    observed.index = observed.obsnme

    # drop model results that aren't in the obs information file
    # these are probably observations that aren't in the model time period
    # (and therefore weren't included in the parent model calibration;
    # but modflow-setup would include them in the MODFLOW observation input)
    # also drop sites that are in the obs information file, but not in the model results
    # these include sites outside of the model (i.e. in the inset when looking at the parent)
    no_info_sites = set(results.obsprefix).symmetric_difference(observed.obsprefix)
    # dump these out to a csv
    print('Dropping {} sites with no information'.format(len(no_info_sites)))
    results.loc[results.obsprefix.isin(no_info_sites)].to_csv('dropped_head_observation_sites.csv',
                                                              index=False)
    results = results.loc[~results.obsprefix.isin(no_info_sites)].copy()
    observed = observed.loc[~observed.obsprefix.isin(no_info_sites)].copy()

    # get_mf6_single_variable_obs returns values for each layer
    # collapse these into one value for each location, time
    # by taking the transmissivity-weighted average
    hk = load_array(hk_arrays)
    top = load_array(top_array)
    botm = load_array(botm_arrays)

    # get the x and y location and open interval corresponding to each head observation
    x = dict(zip(observed['obsprefix'], observed['x']))
    y = dict(zip(observed['obsprefix'], observed['y']))
    screen_top = dict(zip(observed['obsprefix'], observed['screen_top']))
    screen_botm = dict(zip(observed['obsprefix'], observed['screen_botm']))
    results['x'] = [x[obsprefix] for obsprefix in results.obsprefix]
    results['y'] = [y[obsprefix] for obsprefix in results.obsprefix]
    results['screen_top'] = [screen_top[obsprefix] for obsprefix in results.obsprefix]
    results['screen_botm'] = [screen_botm[obsprefix] for obsprefix in results.obsprefix]
    periods = results.groupby('per')
    simulated_heads = []
    for per, data in periods:

        # get a n layers x n sites array of simulated head observations
        data = data.reset_index(drop=True)
        heads_2d = data.pivot(columns='layer', values='sim_head', index='obsnme').T.values
        obsnme = data.pivot(columns='layer', values='obsnme', index='obsnme').index.tolist()

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
        Tr_frac_df.to_csv('obs_layer_transmissivities.csv', float_format='%.2f')
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
        mean_t_weighted_heads_df = pd.DataFrame({sim_values_column: mean_t_weighted_heads}, index=obsnme)
        simulated_heads.append(mean_t_weighted_heads_df)
    all_simulated_heads = pd.concat(simulated_heads)

    # reduce results dataframe from nobs x nlay x nper to just nobs x nper
    head_obs = results.reset_index(drop=True).groupby('obsnme').first()

    # raise an error if there are duplicates- reindexing below will fail if this is the case
    if observed.index.duplicated().any():
        msg = ('The following observations have duplicate names. There should only be'
               'one observation per site, for each time period implied by the '
               'obsnme_date_suffix_format parameter.\n{}'
               .format(observed.loc[observed.duplicated()]))
        raise ValueError(msg)

    # replace the simulated heads column with the transmissivity-weighted heads computed above
    head_obs[obs_values_column] = observed.reindex(head_obs.index)[observed_values_obsval_col]
    head_obs[sim_values_column] = all_simulated_heads.reindex(head_obs.index)[sim_values_column]
    for column in ['group', 'uncertainty']:
        if column in observed.columns:
            head_obs[column] = observed.reindex(head_obs.index)[column]

    # drop any observations that are lake stages
    # (need to compare output from the lake package instead of any head values at these locations)
    if drop_groups is not None and 'group' in head_obs.columns:
        head_obs = head_obs.loc[~head_obs.group.isin(drop_groups)].copy()

    # nans are where sites don't have observation values for that period
    # or sites that are in other model (inset or parent)
    head_obs.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # reorder the columns
    head_obs['obsnme'] = head_obs.index
    columns = ['datetime', 'per', 'obsprefix', 'obsnme', obs_values_column, sim_values_column,
               'group', 'uncertainty', 'screen_top', 'screen_botm']
    columns = [c for c in columns if c in head_obs.columns]
    head_obs = head_obs[columns].copy()

    # fill NaT (not a time) datetimes
    fill_nats(head_obs, perioddata)

    head_obs.sort_values(by=['obsprefix', 'per'], inplace=True)
    if outfile is not None:
        head_obs.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(head_obs, str(outfile) + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=sim_values_column, index=False)
    return head_obs


def get_spatial_head_differences(head_obs, perioddata,
                                 lake_head_difference_sites,
                                 head_obs_values_col='obs_head',
                                 head_sim_values_col='sim_head',
                                 obs_diff_value_col='obsval',
                                 sim_diff_values_col='sim_obsval',
                                 use_gradients=False,
                                 write_ins=False, outfile=None):
    """Takes the head_obs dataframe output by get_head_obs_near_lakes, and
    maybe some other input, and creates spatial head difference observations
    at locations where there are vertical head difference, and writes them
    to a csv file in tables/.

    Parameters
    ----------
    head_obs : DataFrame
        Table of preprocessed head observations
    lake_head_difference_sites : dict
        Dictionary of lake site numbers (keys) and gw level sites (values) to compare.
        Values is list of strings; observations containing these strings will be compared
        to lake stage on the date of measurement.
    use_gradients : bool
        If True, use hydraulic gradients, if False, use vertical head differences.
        By default False.
    """

    # get subset of head_obs sites to compare to each lake in lake_head_difference_sites
    groups = head_obs.groupby('obsprefix')
    spatial_head_differences = []
    for lake_site_no, patterns in lake_head_difference_sites.items():
        compare = []
        for pattern in patterns:
            matches = [True if pattern in site_name else False
                       for site_name in head_obs.obsprefix]
            compare.append(matches)
        compare = np.any(compare, axis=0)
        sites = set(head_obs.loc[compare, 'obsprefix'])

        # for each site in the subset, compare the values to the lake
        # index by stress period
        lake_values = groups.get_group(lake_site_no).copy()
        lake_values.index = lake_values.per

        for obsprefix, site_observations in groups:
            if obsprefix in sites:
                site_obs = site_observations.copy()
                site_obs.index = site_obs.per
                site_obs['other_obsnme'] = lake_values['obsnme']
                site_obs['obs_lake_stage'] = lake_values[head_obs_values_col]
                site_obs['sim_lake_stage'] = lake_values[head_sim_values_col]
                # negative values indicate discharge to the lake
                # (lake stage < head)
                site_obs['obs_dh'] = site_obs['obs_lake_stage'] - site_obs[head_obs_values_col]
                site_obs['sim_dh'] = site_obs['sim_lake_stage'] - site_obs[head_sim_values_col]

                # get a screen midpoint and add gradient
                # assume 1 meter between midpoint and lake if there is no open interval info
                screen_midpoint = site_obs[['screen_top', 'screen_botm']].mean(axis=1).fillna(1)
                site_obs['dz'] = (site_obs['obs_lake_stage'] - screen_midpoint)
                site_obs['obs_grad'] = site_obs['obs_dh'] / site_obs['dz']
                site_obs['sim_grad'] = site_obs['sim_dh'] / site_obs['dz']
                spatial_head_differences.append(site_obs)
    spatial_head_differences = pd.concat(spatial_head_differences)

    # name the spatial head difference obs as
    # <obsprefix>_<obsname suffix>dlake
    obsnme = []
    for i, r in spatial_head_differences.iterrows():
        obs_b_suffix = r.other_obsnme
        obsnme.append('{}d{}'.format(r.obsnme, obs_b_suffix))
    spatial_head_differences['obsnme'] = obsnme
    spatial_head_differences['group'] = ['{}_sdiff'.format(g)
                                         for g in spatial_head_differences['group']]

    # drop some columns that aren't really valid
    spatial_head_differences.drop(['n', 'obsnme_in_parent'], axis=1, inplace=True, errors='ignore')

    # whether to use gradients for the obsvals, or just head differences
    if use_gradients:
        spatial_head_differences['obsval'] = spatial_head_differences['obs_grad']
        spatial_head_differences[sim_diff_values_col] = spatial_head_differences['sim_grad']
        obstype = 'vertical head gradients'
    else:
        spatial_head_differences['obsval'] = spatial_head_differences['obs_dh']
        spatial_head_differences[sim_diff_values_col] = spatial_head_differences['sim_dh']
        obstype = 'vertical head difference'
    spatial_head_differences.dropna(axis=0, subset=['obsval'], inplace=True)
    spatial_head_differences['type'] = obstype

    # uncertainty column is from head_obs;
    # assume that spatial head differences have double the uncertainty
    # (two wells/two measurements per obs)
    spatial_head_differences['uncertainty'] *= 2

    # check for duplicates
    assert not spatial_head_differences['obsnme'].duplicated().any()

    # fill NaT (not a time) datetimes
    fill_nats(spatial_head_differences, perioddata)

    if outfile is not None:
        spatial_head_differences.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(spatial_head_differences, outfile + '.ins',
                          obsnme_column='obsnme',
                          simulated_obsval_column=sim_diff_values_col, index=False)
    return spatial_head_differences


def get_temporal_head_difference_obs(head_obs, perioddata,
                                     head_obs_values_col='obs_head',
                                     head_sim_values_col='sim_head',
                                     obs_diff_value_col='obsval',
                                     sim_diff_values_col='sim_obsval',
                                     exclude_suffix='ss',
                                     outfile=None,
                                     write_ins=False):
    """Takes the head_obs dataframe output by get_head_obs,
    creates temporal head difference observations.

    Parameters
    ----------
    head_obs : DataFrame
        Head observation data with same column structure as
        output from :func:`mfobs.heads.get_head_obs`
    head_obs_values_col : str
        Column in head_obs with observed values to difference.
        By default, 'obs_head'
    head_sim_values_col : str
        Column in head_obs with simulated values to difference.
        By default, 'sim_head`
    obs_diff_value_col : str
        Name of column with computed observed differences.
        By default, 'obsval'
    sim_diff_values_col : str
        Name of column with computed simulated differences.
        By default, 'sim_obsval'
    exclude_suffix : str or list-like
        Option to exclude observations from differencing by suffix;
        e.g. 'ss' to include steady-state observations.
        By default, 'ss'
    outfile : str, optional
        CSV file to write output to.
        By default, None (no output written)
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    period_diffs : DataFrame
    """
    # only compute differences on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in head_obs.obsnme]
    keep = ~np.in1d(suffix, exclude_suffix)
    head_obs = head_obs.loc[keep].copy()

    # group observations by site (prefix)
    sites = head_obs.groupby('obsprefix')
    period_diffs = []
    for site_no, values in sites:
        values = values.sort_values(by=['per'])

        # compute the differences
        values[obs_diff_value_col] = values[head_obs_values_col].diff()
        values[sim_diff_values_col] = values[head_sim_values_col].diff()

        # base the uncertainty on the amount of time that has elapsed
        # assume 1 meter of annual variability; leaving periods > 1 year at 1 m
        # todo: is there a general uncertainty approach for temporal head differences that makes sense?
        #values['uncertainty'] = values.per.diff() / 12
        #values.loc[values.uncertainty > 1, 'uncertainty'] = 1.

        period_diffs.append(values)
    period_diffs = pd.concat(period_diffs).reset_index(drop=True)
    period_diffs['datetime'] = pd.to_datetime(period_diffs['datetime'])

    # name the temporal head difference obs as
    # <obsprefix>_<obsname a suffix>d<obsname b suffix>
    # where the obsval = obsname b - obsname a
    obsnme = []
    for i, r in period_diffs.iterrows():
        obs_b_suffix = ''
        if i > 0:
            obs_b_suffix = period_diffs.loc[i - 1, 'datetime'].strftime('%Y%m')
        obsnme.append('{}d{}'.format(r.obsnme, obs_b_suffix))
    period_diffs['obsnme'] = obsnme
    if 'group' in period_diffs.columns:
        group = ['{}_tdiff'.format(g) for g in period_diffs['group']]
    else:
        group = 'heads_tdiff'
    period_diffs['group'] = group

    # drop some columns that aren't really valid; if they exist
    period_diffs.drop(['n'], axis=1, inplace=True, errors='ignore')

    # drop observations with no difference (first observations at each site)
    period_diffs.dropna(axis=0, subset=[obs_diff_value_col, sim_diff_values_col], inplace=True)

    # fill NaT (not a time) datetimes
    fill_nats(period_diffs, perioddata)

    if outfile is not None:
        period_diffs.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(period_diffs, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column=sim_diff_values_col,
                          index=False)
    return period_diffs