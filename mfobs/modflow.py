"""
Functions specific to MODFLOW and flopy
"""
import json
from affine import Affine
import numpy as np
import pandas as pd
try:
    import flopy
except:
    flopy = False
from mfobs.checks import check_obsnme_suffix
from mfobs.fileio import write_insfile


def get_gwf_obs_input(gwf_obs_input_file):
    """Read the first BEGIN continuous  FILEOUT block of an input
    file to the MODFLOW-6 GWF observation utility.

    Parameters
    ----------
    gwf_obs_input_file : str
        Input file to MODFLOW-6 observation utility (contains layer information).

    Note
    ----
    As-is, this only reads the first block. Modflow-setup writes all of the
    observation input to a single block, but observation input can
    be broken out into multiple blocks (one per file).

    This also doesn't work with open/close statements.
    """
    with open(gwf_obs_input_file) as src:
        for line in src:
            if 'BEGIN continuous' in line:
                df = pd.read_csv(src, delim_whitespace=True, header=None,
                                 error_bad_lines=False)
    df.dropna(axis=0, inplace=True)
    df.columns = ['obsname', 'obstype', 'k', 'i', 'j']
    # cast columns as ints and convert to zero-based
    for index_col in 'k', 'i', 'j':
        df[index_col] = df[index_col].astype(int) - 1
    return df


def get_ij(transform, x, y):
    """Return the row and column of a point or sequence of points
    in real-world coordinates. Uses the affine package. Basically,

    * to get an x, y: transform * (col, row)
    * the inverse, ~transform * (x, y) returns a fractional i, j location on the grid

    Parameters
    ----------
    transform : affine.Affine instance
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

    x : scalar or sequence of x coordinates
    y : scalar or sequence of y coordinates

    Returns
    -------
    i : row or sequence of rows (zero-based)
    j : column or sequence of columns (zero-based)
    """

    j, i = ~transform * (x, y)
    # decrement so that points within a cell (pixel)
    # round to the cell center
    # (transform returns fractional row/column positions 
    # relative to the upper left corner)
    i = np.round(i - 0.5, 0).astype(int)
    j = np.round(j - 0.5, 0).astype(int)
    return i, j


def get_kstp_kper(nstp):
    """Given a sequence of the number of timesteps in each stress period,
    return a sequence of timestep, period tuples (kstp, kper) used
    by various flopy methods.
    """
    kstp = []
    kper = []
    for i, nstp in enumerate(nstp):
        for j in range(nstp):
            kstp.append(j)
            kper.append(i)
    return kstp, kper


def get_layer(column_name):
    """Pandas appends duplicate column names with a .*,
    where * is the number of duplicate names from left to right
    (e.g. obs, obs.1, obs.2, ...). Modflow-setup writes observation input
    for each model layer, at each site location, with the same observation prefix (site identifier).
    MODFLOW-6 reports the result for each layer with duplicate column names,
    with layers increasing from left to right (e.g. obs, obs, obs, ...).

    Parse the layer number from column_name, returning zero if there is no '.' separator.

    Notes
    -----
    The approach can't be used if the model includes inactive cells (including pinched layers)
    at the locations of observations, because it assumes that the layer numbers are consecutive,
    starting at 0. For example, a pinched layer 2 in a 4 layer model would result in the observation
    being in layers 0, 1, 3, which would be misinterpreted as 0, 1, 2.
    """
    if '.' not in column_name:
        return 0
    return int(column_name.split('.')[-1])


def get_mf6_single_variable_obs(perioddata,
                                model_output_file,
                                gwf_obs_input_file=None,
                                variable_name='values',
                                obsnme_date_suffix=True,
                                obsnme_suffix_format='%Y%m',
                                label_period_as_steady_state=None,
                                abs=True):
    """Read raw MODFLOW-6 observation output from csv table with
    times along the row axis and observations along the column axis. Reshape
    (stack) results to be n times x n sites rows, with a single observation value
    in each row. If there is more than one time in a stress period, retain only
    the last time (so that there is one observation per stress period for each site.
    If an input file to the MODFLOW-6 observation utility is included,
    include the observation layer number in the output.

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'per' (stress period number), 'time' (modflow time, in days), 
        'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str
        Path to MODFLOW-6 observation csv output (shape: n times rows x n obs columns).
    gwf_obs_input_file : str
        Input file to MODFLOW-6 observation utility (contains layer information).
    variable_name : str, optional
        Column with simulated output will be named "sim_<variable_name",
        by default 'head'
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
    label_period_as_steady_state : int, optional
        Zero-based model stress period where observations will be
        assigned the suffix 'ss' instead of a date suffix.
        By default, None, in which case all model output is assigned
        a date suffix based on the start date of the stress period.
    abs : bool, optional
        Option to convert simulated values to absolute values

    Returns
    -------
    results : DataFrame
        DataFrame with one head observation per row, with the following columns:

        =================== =============================================================
        per                 zero-based model stress period
        obsprefix           prefix of observation name (site identifier)
        sim_<variable_name> column with simulated values
        datetime            pandas datetimes, based on stress period start date
        layer               zero-based model layer
        obsnme              observation name based on format of <obsprefix>_'%Y%m'

        =================== =============================================================

        Example observation names:

        site1000_202001, for a Jan. 2020 observation at site1000 (obsnme_date_suffix=True)
        
        site1000_001, for a stress period 1 observation at site1000 (obsnme_date_suffix=False)

        a steady-state stress period specified with label_period_as_steady_state 
        is given the suffix of 'ss'
        e.g. site1000_ss


    """
    # validation checks
    check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format, 
                        function_name='get_mf6_single_variable_obs')
    
    if perioddata.index.name == 'per':
        perioddata = perioddata.sort_index()
    else:
        perioddata = perioddata.sort_values(by='per')
    if 'perlen' not in perioddata.columns:
        perioddata['perlen'] = perioddata['time'].diff().fillna(0).tolist()
    print('reading model output from {}...'.format(model_output_file))
    model_output = pd.read_csv(model_output_file)

    # convert all observation names to lower case
    model_output.columns = model_output.columns.str.lower()

    # add stress period information to model output
    # by having pandas match time floats in indices
    # the last time in each stress period is retained
    # (corresponding to the model end time listed in perioddata)
    model_output.index = model_output['time']
    perioddata.index = perioddata['time']
    model_output['per'] = perioddata['per']
    model_output['perioddata_time'] = perioddata['time']
    model_output.dropna(subset=['per'], axis=0, inplace=True)
    model_output['per'] = model_output['per'].astype(int)
    assert np.allclose(model_output.time.values, model_output.perioddata_time.values)
    model_output.index = model_output['per']

    # reshape the model output from (nper rows, nsites columns) to nper x nsites rows
    stacked = model_output.drop(['time', 'perioddata_time', 'per'], axis=1).stack(level=0).reset_index()
    simval_col = 'sim_{}'.format(variable_name)
    stacked.columns = ['per', 'obsprefix', simval_col]

    # optionally convert simulated values to absolute values
    if abs:
        stacked[simval_col] = stacked[simval_col].abs()

    # add dates
    #perlen = dict(zip(perioddata.per, perioddata.perlen))
    #period_start_dates = dict(zip(perioddata.per, perioddata.start_datetime))
    period_end_dates = dict(zip(perioddata.per, perioddata.end_datetime))
    stacked['datetime'] = pd.to_datetime([period_end_dates.get(per) for per in stacked.per])
    # get the start date of the next period
    # so that suffix for an observation would be consistent with the start date of the next obs
    #next_period_start = [period_start_dates.get(per) for per in stacked.per][1:]
    #last_per = perioddata.per.max()
    #last_end_date = pd.Timestamp(period_start_dates[last_per]) + \
    #    pd.Timedelta(perlen[last_per], unit='d')
    #next_period_start.append(last_end_date)

    # parse the layers from the column positions (prior to stacking)
    if gwf_obs_input_file is not None:
        gwf_obs_input = get_gwf_obs_input(gwf_obs_input_file)
        # Assign layer to each observation,
        # assuming that numbering in gwf_obs_input is repeated nper times
        nper = len(stacked.per.unique())
        stacked['layer'] = gwf_obs_input['k'].tolist() * nper

    # reset the obsprefixes to be the same for different layers at a location
    stacked['obsprefix'] = [prefix.split('.')[0] for prefix in stacked.obsprefix]

    # assign obsnames using the prefixes (location identifiers) and month
    obsnames = []
    for prefix, per, dt in zip(stacked.obsprefix, stacked.per, stacked.datetime):
        if per == label_period_as_steady_state:
            name = f"{prefix}_ss"
        elif obsnme_date_suffix and not pd.isnull(dt):
            name = f"{prefix}_{dt.strftime(obsnme_suffix_format)}"
        elif not obsnme_date_suffix:
            suffix = f"{per:{obsnme_suffix_format.strip('{:}')}}"
            name = f"{prefix}_{suffix}"
        else:
            name = prefix
        obsnames.append(name)
    stacked['obsnme'] = obsnames

    # drop any duplicate observations, keeping those from transient periods
    # (for example, an initial steady-state period that isn't being used for observations
    # and a subsequent transient period with the same start date)
    # first make temp obsnames that include layer
    # steady state periods that are being used for observations (label_period_as_steady_state=True)
    # won't be dropped because their obs will have an "ss" suffix instead of a date suffix
    if gwf_obs_input_file is not None:
        unique_obsnames = ['{}_{}'.format(name, layer)
                           for name, layer in zip(stacked.obsnme, stacked.layer)]
    else:
        unique_obsnames = stacked.obsnme.to_list()
    stacked['unique_obsnames'] = unique_obsnames
    are_duplicates = stacked['unique_obsnames'].duplicated(keep=False)
    #are_duplicates = pd.Series(unique_obsnames).duplicated(keep=False).values
    if any(are_duplicates):
        #duplicated_obsnames = set(stacked.loc[are_duplicates.values, 'obsnme'])
        steady_obs = stacked.per.isin(perioddata.loc[perioddata.steady, 'per'].values)
        drop = are_duplicates & steady_obs
        stacked = stacked.loc[~drop]
        #unique_obsnames = np.array(unique_obsnames)[~drop]
        #assert not any(pd.Series(unique_obsnames).duplicated())
    if stacked['unique_obsnames'].duplicated().any():
        duplicates = stacked.loc[stacked['unique_obsnames']. \
            duplicated(keep=False)].sort_values(by='unique_obsnames')
        msg = ("mfobs.modflow.get_mf6_single_variable_obs:"
               "Duplicate observation names. If obsnme_date_suffix=True, "
               "you may need a more specific obsnme_suffix_format, e.g. '%Y%m%d\n"
               "Or there may be a mismatch between the model results (e.g. perlen) "
               "and start and end dates in the stress period data table (perioddata).\n"
               "In the latter case, you may need to re-run the model "
               f"and possibly the model setup.\nDuplicated obs:{duplicates}"
               ""
               )
        raise ValueError(msg)

    stacked.index = stacked['obsnme']
    sort_cols = [c for c in ['obsprefix', 'per', 'layer'] if c in stacked.columns]
    stacked.sort_values(by=sort_cols, inplace=True)
    results = stacked
    return results


def get_modflow_mass_balance(modroot, outfile=None, write_ins=True):
    """
    read in the percent discrepancy for inset and parent models

    Parameters
    ----------
    modroot: root name of the model scenario
    outfile: filepath for output
    write_ins: bool. whether or not to write instruction file
    """
    print('reading in the mass balance files')
    # make a list with which to concatenate results
    dfs = []
    # read in both inset and parent list files
    for cmod in ['inset', 'parent']:
        # read in the list files
        mfl6 = fp.utils.Mf6ListBudget("{0}{1}_{2}.list".format(rundir, modroot, cmod))
        # get all the budget information
        df, _ = mfl6.get_dataframes(start_datetime="1-1-2012")
        # construct the obsname with the date etc.
        df['obsnme'] = ['{0}_discrep_{1:d}{2:02d}'.format(cmod, i.year, i.month) for i in df.index]
        # append on the max absolute percent discrepancy
        df = df.append({'obsnme': '{}_discrep_max'.format(cmod),
                        'PERCENT_DISCREPANCY': df.PERCENT_DISCREPANCY.abs().max()},
                       ignore_index=True)
        dfs.append(df[['obsnme', 'PERCENT_DISCREPANCY']])
    outdf = pd.concat(dfs)
    outdf['group'] = 'percent_discrep'
    outdf['obsval'] = 0
    outdf.to_csv(outfile, index=False, sep=' ')
    print(f'wrote {len(outdf):,} observations to {outfile}')
    if write_ins:
        write_insfile(outdf, outfile + '.ins', obsnme_column='obsnme',
                      simulated_obsval_column='PERCENT_DISCREPANCY', index=False)


def get_modelgrid_transform(grid_json_file, shift_to_cell_centers=False):
    """Create an affine.Affine that describes the model grid
    from a json file. The affine package comes with rasterio
    as a dependency or can be installed separately.

    Parameters
    ----------
    grid_json_file : str
        Model grid json file produced by modflow-setup
    shift_to_cell_centers : bool
        By default, transform reflects the upper left corner of
        the first cell in the model, and any conversions of x, y
        coordinates to pixels will be relative to upper left corners.
        If shift_to_cell_centers=True, x,y points will be referenced
        to the nearest cell centers.
    """
    with open(grid_json_file) as f:
        cfg = json.load(f)

    for dx in 'delr', 'delc':
        if not np.isscalar(cfg[dx]):
            cfg[dx] = cfg[dx][0]
    xul = cfg['xul']
    yul = cfg['yul']
    if shift_to_cell_centers:
        xul += 0.5 * cfg['delr']
        yul -= 0.5 * cfg['delc']

    transform = Affine(cfg['delr'], 0., xul,
                       0., -cfg['delr'], yul) * \
                Affine.rotation(cfg['angrot'])
    return transform


def read_mf6_lake_obs(f, perioddata, start_date='2012-01-01',
                      keep_only_last_timestep=True):
    df = pd.read_csv(f)
    df.columns = df.columns.str.lower()

    # convert modflow time to actual elapsed time
    # (by subtracting off the day for the initial steady-state period)
    if df.time.iloc[0] == 1:
        df['time'] -= 1

    # get stress period information for each timestep recorded
    kstp, kper = get_kstp_kper(perioddata.nstp)
    df['kstp'] = kstp
    df['kper'] = kper
    if len(df) == len(kstp) + 1:
        df = df.iloc[1:].copy()
    if keep_only_last_timestep:
        df = df.groupby('kper').last().reset_index()
    start_ts = pd.Timestamp(start_date)
    df['datetime'] = pd.to_timedelta(df.time, unit='D') + start_ts
    df.index = df.datetime
    return df.drop('datetime', axis=1)


def get_transmissivities(heads, hk, top, botm,
                         r=None, c=None, x=None, y=None, modelgrid_transform=None,
                         screen_top=None, screen_botm=None, nodata=-999):
    """
    Computes transmissivity in each model layer at specified locations and
    open intervals. A saturated thickness is determined for each row, column
    or x, y location supplied, based on the open interval (sctop, screen_botm),
    if supplied, otherwise the layer tops and bottoms and the water table
    are used.

    Parameters
    ----------
    heads : 2D array OR 3D array
        numpy array of shape nlay by n locations (2D) OR complete heads array
        of the model for one time (3D)
    hk : 3D numpy array
        horizontal hydraulic conductivity values.
    top : 2D numpy array
        model top elevations.
    botm : 3D numpy array
        layer botm elevations.
    r : 1D array-like of ints, of length n locations
        row indices (optional; alternately specify x, y)
    c : 1D array-like of ints, of length n locations
        column indices (optional; alternately specify x, y)
    x : 1D array-like of floats, of length n locations
        x locations in real world coordinates (optional).
        If x and y are specified, a modelgrid_transform must also be provided.
    y : 1D array-like of floats, of length n locations
        y locations in real world coordinates (optional)
        If x and y are specified, a modelgrid_transform must also be provided.
    modelgrid_transform : affine.Affine instance, optional
        An `affine.Affine <https://github.com/sgillies/affine>`_ object describing the orientation
        of the model grid. Only required for getting i, j if x and y are specified.
        Modflow-setup :class:`~mfsetup.grid.MFsetupGrid` have this attached
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

    screen_top : 1D array-like of floats, of length n locations
        open interval tops (optional; default is model top)
    screen_botm : 1D array-like of floats, of length n locations
        open interval bottoms (optional; default is model bottom)
    nodata : numeric
        optional; locations where heads=nodata will be assigned T=0

    Returns
    -------
    T : 2D array of same shape as heads (nlay x n locations)
        Transmissivities in each layer at each location

    """
    if r is not None and c is not None:
        pass
    elif x is not None and y is not None:
        # get row, col for observation locations
        r, c = get_ij(modelgrid_transform, x, y)
    else:
        raise ValueError('Must specify row, column or x, y locations.')

    # get k-values and botms at those locations
    # (make nlayer x n sites arrays)
    hk2d = hk[:, r, c]
    botm2d = botm[:, r, c]

    if len(heads.shape) == 3:
        heads = heads[:, r, c]

    msg = 'Shape of heads array must be nlay x nhyd'
    assert heads.shape == botm2d.shape, msg

    # set open interval tops/bottoms to model top/bottom if None
    if screen_top is None:
        screen_top = top[r, c]
    if screen_botm is None:
        screen_botm = botm[-1, r, c]

    # make an nlayers x n sites array of layer tops
    tops = np.empty_like(botm2d, dtype=float)
    tops[0, :] = top[r, c]
    tops[1:, :] = botm2d[:-1]

    # expand top and bottom arrays to be same shape as botm, thickness, etc.
    # (so we have an open interval value for each layer)
    sctoparr = np.zeros(botm2d.shape)
    sctoparr[:] = screen_top
    scbotarr = np.zeros(botm2d.shape)
    scbotarr[:] = screen_botm

    # start with layer tops
    # set tops above heads to heads
    # set tops above screen top to screen top
    # (we only care about the saturated open interval)
    openinvtop = tops.copy()
    openinvtop[openinvtop > heads] = heads[openinvtop > heads]
    openinvtop[openinvtop > sctoparr] = sctoparr[openinvtop > screen_top]

    # start with layer bottoms
    # set bottoms below screened interval to screened interval bottom
    # set screen bottoms below bottoms to layer bottoms
    openinvbotm = botm2d.copy()
    openinvbotm[openinvbotm < scbotarr] = scbotarr[openinvbotm < screen_botm]
    openinvbotm[scbotarr < botm2d] = botm2d[scbotarr < botm2d]

    # compute thickness of open interval in each layer
    thick = openinvtop - openinvbotm

    # assign open intervals above or below model to closest cell in column
    not_in_layer = np.sum(thick < 0, axis=0)
    not_in_any_layer = not_in_layer == thick.shape[0]
    for i, n in enumerate(not_in_any_layer):
        if n:
            closest = np.argmax(thick[:, i])
            thick[closest, i] = 1.
    thick[thick < 0] = 0
    thick[heads == nodata] = 0  # exclude nodata cells
    thick[np.isnan(heads)] = 0  # exclude cells with no head value (inactive cells)

    # compute transmissivities
    T = thick * hk2d
    return T


def read_mf6_block(filename, blockname):
    blockname = blockname.lower()
    data = {}
    read = False
    per = None
    with open(filename) as src:
        for line in src:
            line = line.lower()
            if 'begin' in line and blockname in line:
                if blockname == 'period':
                    per = int(line.strip().split()[-1])
                    data[per] = []
                elif blockname == 'continuous':
                    fname = line.strip().split()[-1]
                    data[fname] = []
                elif blockname == 'packagedata':
                    data['packagedata'] = []
                else:
                    blockname = line.strip().split()[-1]
                    data[blockname] = []
                read = blockname
                continue
            if 'end' in line and blockname in line:
                per = None
                read = False
                #break
            if read == 'options':
                line = line.strip().split()
                data[line[0]] = line[1:]
            elif read == 'packages':
                pckg, fname, ext = line.strip().split()
                data[pckg] = fname
            elif read == 'period':
                data[per].append(' '.join(line.strip().split()))
            elif read == 'continuous':
                data[fname].append(' '.join(line.strip().split()))
            elif read == 'packagedata':
                data['packagedata'].append(' '.join(line.strip().split()))
            elif read == blockname:
                data[blockname].append(' '.join(line.strip().split()))
    return data


def get_perioddata(tdis_file, sto_file=None):
    """Make the perioddata table required by other modflow-obs
    functions from a MODFLOW-6 Temporal Discretization (TDIS) file.

    Parameters
    ----------
    tdis_file : str
        TDIS Package file
        
    Returns
    -------
    perioddata : DataFrame
    """
    options = read_mf6_block(tdis_file, 'options')
    perioddata = read_mf6_block(tdis_file, 'perioddata')
    steady_period_blocks = None
    if sto_file is not None:
        steady_period_blocks = read_mf6_block(sto_file, 'period')
    
    start_datetime = options['start_date_time']
    if len(start_datetime) > 0:
        start_datetime = pd.to_datetime(start_datetime[0])
    else:
        msg = (f'No start_date_time in {tdis_file};' 
                'start_date_time is needed to construct perioddata.')
        raise ValueError(msg)
    time_units = options['time_units']
    if len(time_units) > 0:
        time_units = time_units[0]
    else:
        msg = (f'No time_units in {tdis_file};' 
                'time_units is needed to construct perioddata.')
        raise ValueError(msg)
    perlen = []
    nstp = []
    tsmult = []
    for row in perioddata['perioddata']:
        if row.strip().startswith('#'):
            continue
        rperlen, rnstp, rtsmult = row.split('#')[0].split()
        perlen.append(float(rperlen))
        nstp.append(int(rnstp))
        tsmult.append(float(rtsmult))
        
    if steady_period_blocks is not None:
        steady = [True if 'steady' in steady_period_blocks[i+1][0] 
                  else False for i in range(len(perlen))]
    else:
        steady = True
        
    perlen = np.array(perlen)
    actual_period_length = perlen.copy()
    if not np.isscalar(steady) and steady[0]:
        actual_period_length[0] = 0
    elapsed_time = np.cumsum(actual_period_length).tolist()
    # modflow time (includes length of initial steady-state period)
    mf_time = np.cumsum(perlen)
    start_elapsed_times = [0] + elapsed_time[:-1]
    start_datetimes = start_datetime + pd.to_timedelta(start_elapsed_times, unit=time_units)
    end_datetimes = start_datetimes + pd.to_timedelta(perlen - 1, unit='d')
    
    perioddata = pd.DataFrame({'start_datetime': start_datetimes.strftime('%Y-%m-%d'),
                               'end_datetime': end_datetimes.strftime('%Y-%m-%d'),
                               'time': mf_time,
                               'per': np.arange(len(perlen)),
                               'perlen': perlen,
                               'nstp': nstp,
                               'tsmult': tsmult,
                               'steady': steady
                               })
    return perioddata