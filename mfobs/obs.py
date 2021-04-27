"""
General functions that apply to multiple types of observations
"""
import numpy as np
import pandas as pd
from mfobs.checks import check_obsnme_suffix
from mfobs.fileio import write_insfile
from mfobs.utils import fill_nats, set_period_start_end_dates


def get_spatial_differences(base_data, perioddata,
                            difference_sites,
                            obs_values_col='obs_head',
                            sim_values_col='sim_head',
                            obstype='head',
                            use_gradients=False,
                            sep='-d-',
                            write_ins=False, outfile=None):
    """Takes the base_data dataframe output by :func:`mfobs.obs.get_obs` and creates
    spatial difference observations. Optionally writes an output csvfile
    and a PEST instruction file.

    Parameters
    ----------
    base_data : DataFrame
        Table of preprocessed observations, such as that produced by
        :func:`mfobs.obs.get_obs`
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
    obstype : str  {'head', 'flux', or other}
        Type of observation being processed. Simulated and observed values
        columns are named in the format 'sim_<obstype>' and 'obs_<obstype>',
        respectively. If there is no 'obgnme' column in ``base_data``,
        ``obstype`` is also used as a default base group name. Finally,
        a 'type' column is included in the output with the label
        'vertical <obstype> gradient' or '<obstype> difference', depending on whether
        ``use_gradients=True``.
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
        CSV file to write output to. Nan values are filled with -9999.
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
        * Columns relating to well open interval are only created if ``obstype='head'``
          and ``base_data`` has 'screen_top' and 'screen_botm' columns.
        * Negative difference or gradient values indicate a gradient towards the key site.

    """

    # model stress period data:
    perioddata = perioddata.copy()
    # make sure start and end dates don't overlap
    set_period_start_end_dates(perioddata)
    perioddata.index = perioddata.per

    # get subset of base_data sites to compare to each key site in difference_sites
    base_data_sites = set(base_data.obsprefix)
    groups = base_data.groupby('obsprefix')
    spatial_differences = []
    for key_site_no, patterns in difference_sites.items():
        
        if key_site_no not in base_data_sites:
            print((f'warning: site {key_site_no} not in base_data. '
                   'Skipping spatial differencing.'))
            continue
        
        compare = []
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            matches = [True if pattern in site_name else False
                       for site_name in base_data.obsprefix]
            compare.append(matches)
        compare = np.any(compare, axis=0)
        sites = set(base_data.loc[compare, 'obsprefix'])

        # for each site in the subset, compare the values to the keys site
        # index by stress period
        key_values = groups.get_group(key_site_no).copy()
        key_values.index = key_values.per

        for obsprefix, site_observations in groups:
            if obsprefix in sites:
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
                if {'screen_top1', 'screen_botm1'}.intersection(site_obs.columns):
                    screen_midpoint1 = site_obs[['screen_top1', 'screen_botm1']].mean(axis=1)
                if {'screen_top2', 'screen_botm2'}.intersection(site_obs.columns):
                    screen_midpoint2 = site_obs[['screen_top2', 'screen_botm2']].mean(axis=1)
                    if screen_midpoint1 is not None:
                        site_obs['dz'] = (screen_midpoint1 - screen_midpoint2)
                        site_obs['obs_grad'] = site_obs['obs_diff'] / site_obs['dz']
                        site_obs['sim_grad'] = site_obs['sim_diff'] / site_obs['dz']
                spatial_differences.append(site_obs)
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
        prefix = '{}{}{}'.format(prefix1, sep, prefix2, )
        obsnme.append('{}_{}'.format(prefix, suffix2))
        obsprefix.append(prefix)
    spatial_differences['obsnme'] = obsnme
    spatial_differences['obsprefix'] = obsprefix
    if 'obgnme' not in spatial_differences.columns:
        spatial_differences['obgnme'] = obstype
    spatial_differences['obgnme'] = ['{}_sdiff'.format(g)
                                         for g in spatial_differences['obgnme']]

    # clean up columns
    cols = ['datetime', 'per', 'obsprefix',
            'obsnme1', f"{obs_values_col}1", f"{sim_values_col}1", 'screen_top1', 'screen_botm1', 'layer1',
            'obsnme2', f"{obs_values_col}2", f"{sim_values_col}2", 'screen_top2', 'screen_botm2', 'layer2',
            'obs_diff', 'sim_diff', 'dz', 'obs_grad', 'sim_grad', 'obgnme', 'obsnme'
            ]
    cols = [c for c in cols if c in spatial_differences.columns]
    spatial_differences = spatial_differences[cols]

    # whether to use gradients for the obsvals, or just head differences
    if use_gradients:
        spatial_differences['obsval'] = spatial_differences['obs_grad']
        spatial_differences['sim_obsval'] = spatial_differences['sim_grad']
        obstype = f'{obstype} gradients'
    else:
        spatial_differences['obsval'] = spatial_differences['obs_diff']
        spatial_differences['sim_obsval'] = spatial_differences['sim_diff']
        obstype = f'spatial {obstype} difference'
    spatial_differences.dropna(axis=0, subset=['obsval'], inplace=True)
    spatial_differences['type'] = obstype

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
        spatial_differences.fillna(-9999).to_csv(outfile, sep=' ', index=False)
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
                             obstype='head',
                             get_displacements=False,
                             displacement_from=None,
                             obsnme_date_suffix=True,
                             obsnme_suffix_format='%Y%m',
                             exclude_suffix='ss',
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
    obstype : str  {'head', 'flux', or other}
        Type of observation being processed. Simulated and observed values
        columns are named in the format 'sim_<obstype>' and 'obs_<obstype>',
        respectively. If there is no 'obgnme' column in ``base_data``,
        ``obstype`` is also used as a default base group name.
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

    Notes
    -----
    Differences are computed by subtracting the previous time from the current,
    so a positive value indicates an increase.
    """
    # validation checks
    check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
                        function_name='get_head_obs', obsdata=base_data)
    
    # only compute differences on transient obs
    if isinstance(exclude_suffix, str):
        exclude_suffix = [exclude_suffix]
    suffix = [obsnme.split('_')[1] for obsnme in base_data.obsnme]
    keep = ~np.in1d(suffix, exclude_suffix)
    base_data = base_data.loc[keep].copy()

    # group observations by site (prefix)
    sites = base_data.groupby('obsprefix')
    period_diffs = []
    for site_no, values in sites:
        values = values.sort_values(by=['per']).copy()
        values.index = values['datetime']

        # compute the differences
        if get_displacements:
            values = values.loc[displacement_from:]
            
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
            values['obsval'].iloc[0] = np.nan
            values['sim_obsval'].iloc[0] = np.nan
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
    period_diffs['datetime'] = pd.to_datetime(period_diffs['datetime'])

    # name the temporal difference obs as
    # <obsprefix>_<obsname1 suffix>d<obsname2 suffix>
    # where the obsval = obsname2 - obsname1
    #obsnme = []
    #for i, r in period_diffs.iterrows():
    #    obsname2_suffix = ''
    #    if i > 0:
    #        if get_displacements:
    #            obsname_2_loc = 0
    #        else:
    #            obsname_2_loc = i - 1
    #        # date-based suffixes
    #        if obsnme_date_suffix:
    #            obsname2_suffix = period_diffs.loc[obsname_2_loc, 
    #                                               'datetime'].strftime(obsnme_suffix_format)
    #        # stress period-based suffixes
    #        else:
    #            per = period_diffs.loc[obsname_2_loc, 'per']
    #            obsname2_suffix = f"{per:{obsnme_suffix_format.strip('{:}')}}"
    #    obsnme.append('{}d{}'.format(r.obsnme, obsname2_suffix))
    #period_diffs['obsnme'] = obsnme
    if 'obgnme' not in period_diffs.columns:
        period_diffs['obgnme'] = obstype
    
    if get_displacements:
        period_diffs['type'] = f'{obstype} displacement'
        period_diffs['obgnme'] = [f'{g}_disp' for g in period_diffs['obgnme']]
    else:
        period_diffs['type'] = f'temporal {obstype} difference'
        period_diffs['obgnme'] = [f'{g}_tdiff' for g in period_diffs['obgnme']]

    # drop some columns that aren't really valid; if they exist
    period_diffs.drop(['n'], axis=1, inplace=True, errors='ignore')

    # clean up columns
    cols = ['datetime', 'per', 'obsprefix', 'obsnme',
            f'obs_{obstype}', f'sim_{obstype}', 'screen_top', 'screen_botm', 'layer',
            'obsval', 'sim_obsval', 'obgnme', 'type']
    cols = [c for c in cols if c in period_diffs.columns]
    period_diffs = period_diffs[cols]

    # drop observations with no difference (first observations at each site)
    period_diffs.dropna(axis=0, subset=['obsval', 'sim_obsval'], inplace=True)

    # fill NaT (not a time) datetimes
    fill_nats(period_diffs, perioddata)

    if outfile is not None:
        period_diffs.fillna(-9999).to_csv(outfile, sep=' ', index=False)
        print(f'wrote {len(period_diffs):,} observations to {outfile}')

        # write the instruction file
        if write_ins:
            write_insfile(period_diffs, str(outfile) + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column='sim_obsval',
                          index=False)
    return period_diffs
