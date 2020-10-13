"""
Functions for processing head observations
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from mfobs.fileio import load_array, write_insfile
from mfobs.modflow import get_mf6_single_variable_obs, get_transmissivities
from mfobs.utils import fill_nats, set_period_start_end_dates


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
                 observed_values_layer_col=None,
                 observed_values_group_column='group',
                 observed_values_unc_column='uncertainty',
                 aggregrate_observed_values_by='mean',
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
    observed_values_layer_col : str, optional
        As an alternative to providing screen tops and bottoms, the model layer
        for each observation can be specified directly via a layer column
        of zero-based layer numbers.
        by default None
    observed_values_group_column : str, optional
        Column name in observed_values_file with observation group information.
        Passed through to output ``head_obs`` DataFrame, otherwise not required.
        by default 'group'
    observed_values_unc_column : str, optional
        Column name in observed_values_file with observation uncertainty values.
        Passed through to output ``head_obs`` DataFrame, otherwise not required.
        by default 'uncertainty'
    aggregrate_observed_values_by : str
        Method for aggregating observed values to the model stress periods,
        if there are multiple observed values in a stress period. Can be any
        of the method calls on the pandas
        `Resampler <https://pandas.pydata.org/pandas-docs/stable/reference/resampling.html>`_
        object. By default, 'mean'
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
    outpath = Path(outfile).parent

    obs_values_column = 'obs_' + variable_name  # output column with observed values
    sim_values_column = 'sim_' + variable_name  # output column with simulated equivalents to observed values

    perioddata = perioddata.copy()
    set_period_start_end_dates(perioddata)
    perioddata.index = perioddata.per
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
               observed_values_layer_col: 'layer',
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

    # get_mf6_single_variable_obs returns values for each layer
    # collapse these into one value for each location, time
    # by taking the transmissivity-weighted average
    if observed_values_layer_col is None:
        hk = load_array(hk_arrays)
        top = load_array(top_array)
        botm = load_array(botm_arrays)

    # get the x and y location and open interval corresponding to each head observation
    x = dict(zip(observed['obsprefix'], observed['x']))
    y = dict(zip(observed['obsprefix'], observed['y']))
    results['x'] = [x[obsprefix] for obsprefix in results.obsprefix]
    results['y'] = [y[obsprefix] for obsprefix in results.obsprefix]

    # get head values based on T-weighted average of open interval
    if observed_values_layer_col is None:
        screen_top = dict(zip(observed['obsprefix'], observed['screen_top']))
        screen_botm = dict(zip(observed['obsprefix'], observed['screen_botm']))
        results['screen_top'] = [screen_top[obsprefix] for obsprefix in results.obsprefix]
        results['screen_botm'] = [screen_botm[obsprefix] for obsprefix in results.obsprefix]

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
            warnings.warn(('No observations between start and '
                           'end dates of {} and {}!'.format(start, end)))
            continue
        observed_in_period.sort_values(by=['obsprefix', 'datetime'], inplace=True)
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

        # get head values based on T-weighted average of open interval
        if observed_values_layer_col is None:
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
            mean_t_weighted_heads_df = pd.DataFrame({sim_values_column: mean_t_weighted_heads}, index=obsnme)
            observed_in_period_rs[sim_values_column] = mean_t_weighted_heads_df[sim_values_column]

        # Get head values for specified layers
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
        observed_in_period_rs['per'] = per
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
    if drop_groups is not None and 'group' in head_obs.columns:
        head_obs = head_obs.loc[~head_obs.group.isin(drop_groups)].copy()

    # nans are where sites don't have observation values for that period
    # or sites that are in other model (inset or parent)
    head_obs.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # reorder the columns
    columns = ['datetime', 'per', 'obsprefix', 'obsnme', obs_values_column, sim_values_column,
               'group', 'uncertainty', 'screen_top', 'screen_botm', 'layer']
    columns = [c for c in columns if c in head_obs.columns]
    head_obs = head_obs[columns].copy()
    if 'layer' in columns:
        head_obs['layer'] = head_obs['layer'].astype(int)

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