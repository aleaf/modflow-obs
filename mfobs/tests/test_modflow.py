"""Tests for the modflow module.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from affine import Affine
import pytest
from mfobs.modflow import (
    get_ij, 
    get_modelgrid_transform,
    get_perioddata, 
    get_site_no,
    add_timesteps_to_perioddata,
    get_mf6_single_variable_obs,
    read_mf_gage_package_output_files,
    get_mf_gage_package_obs,
    get_gwf_obs_input,
    get_transmissivities
)


def test_get_ij():
    
    # upper left corner
    xul, yul = 434955., 1342785.
    spacing = 500.
    transform = Affine(spacing, 0.0, xul,
                       0.0, -spacing, yul)
    
    # get the i, j location for a point just past the first cell center
    x = xul + spacing/2 + 1
    y = yul - spacing/2 - 1
    i, j = get_ij(transform, x, y)
    assert (i, j) == (0, 0)
    
    # test the upper left corner
    i, j = get_ij(transform, xul, yul)
    assert (i, j) == (0, 0)
    
    # test a lower left corner
    expected_i, expected_j = 1000, 1000
    i, j = get_ij(transform, 
                  xul + spacing * expected_i, 
                  yul - spacing * expected_j
                  )
    assert (i, j) == (expected_i, expected_j)  
    

def get_rotated_transform(angrot, xul=434955., yul=1342785., delr=500., delc=None):
    """Make an Affine transform for a uniform grid with an upper left corner at xul, yul
    and cells that are delr wide and delc tall (delc=delr by default),
    rotated counter-clockwise about the upper left corner by angrot (degrees),
    following the example in the get_ij docstring: the rotation about the upper left corner
    is applied to the grid transform (after the cell sizes), so that the grid isn't skewed
    when delr != delc. Affine.rotation is a classmethod that returns a rotation matrix only;
    the transforms are composed with @ in affine >= 3.0, or * in earlier versions."""
    delc = delr if delc is None else delc
    grid_transform = Affine(delr, 0.0, xul, 0.0, -delc, yul)
    rotation = Affine.rotation(angrot, pivot=(xul, yul))
    if hasattr(Affine, '__matmul__'):
        return rotation @ grid_transform
    return rotation * grid_transform


def assert_grid_axes(transform, angrot, delr, delc):
    """Check that a transform's column and row axes have the expected lengths, are at right angles,
    and are rotated counter-clockwise by angrot (the MODFLOW angrot convention)."""
    column_axis = np.array([transform.a, transform.d])  # real-world displacement for one column
    row_axis = np.array([transform.b, transform.e])  # real-world displacement for one row
    assert np.isclose(np.hypot(*column_axis), delr)
    assert np.isclose(np.hypot(*row_axis), delc)
    assert np.isclose(column_axis @ row_axis, 0., atol=1e-6 * delr * delc)
    # the column axis points angrot degrees counter-clockwise from east;
    # the row axis (down the grid) points 90 degrees clockwise from the column axis
    column_angle = np.degrees(np.arctan2(*column_axis[::-1]))
    row_angle = np.degrees(np.arctan2(*row_axis[::-1]))
    assert np.isclose((column_angle - angrot + 180) % 360 - 180, 0.)
    assert np.isclose((column_angle - row_angle) % 360, 90.)


@pytest.mark.parametrize('angrot,delr,delc', (
    (0., 1000., 1000.),
    (30., 1000., 1000.),
    (45., 1000., 1000.),
    (-20., 1000., 1000.),
    # unequal row and column spacings
    (0., 500., 800.),
    (30., 500., 800.),
    (-20., 800., 500.),
))
def test_rotated_transform_convention(angrot, delr, delc):
    """The rotated transform described in the get_ij docstring retains the grid's
    upper left corner and cell sizes, and rotates counter-clockwise (the MODFLOW angrot convention)."""
    xul, yul = 500955., 1205285.
    transform = get_rotated_transform(angrot, xul, yul, delr, delc)
    # upper left corner is the pivot
    assert np.allclose((transform.c, transform.f), (xul, yul))
    assert_grid_axes(transform, angrot, delr, delc)


def test_rotated_transform_first_cell_center():
    """For a grid rotated 45 degrees counter-clockwise about its upper left corner,
    the first cell center is directly east of the upper left corner
    (the half-cell offsets down the first column and row cancel in y)."""
    xul, yul, spacing = 500955., 1205285., 1000.
    transform = get_rotated_transform(45., xul, yul, spacing)
    x = transform.a * 0.5 + transform.b * 0.5 + transform.c
    y = transform.d * 0.5 + transform.e * 0.5 + transform.f
    assert np.isclose(y, yul)
    assert np.isclose(x, xul + spacing * np.sqrt(2) / 2)
    assert get_ij(transform, x, y) == (0, 0)


def write_grid_json(path, angrot, xul=500955., yul=1205285., delr=1000., delc=1000.,
                    nrow=3, ncol=4, scalar_spacing=False):
    """Write a (minimal) modflow-setup style grid json file,
    with delr and delc as lists (one value per column and row) or scalars."""
    cfg = {'angrot': angrot, 'xul': xul, 'yul': yul,
           'delr': delr if scalar_spacing else [delr] * ncol,
           'delc': delc if scalar_spacing else [delc] * nrow,
           'nrow': nrow, 'ncol': ncol}
    with open(path, 'w') as dest:
        json.dump(cfg, dest)
    return path


@pytest.mark.parametrize('angrot,delr,delc', (
    (0., 1000., 1000.),
    (30., 1000., 1000.),
    (45., 1000., 1000.),
    (-20., 1000., 1000.),
    # unequal row and column spacings (the grid shouldn't be skewed when rotated)
    (0., 500., 800.),
    (30., 500., 800.),
    (90., 500., 800.),
    (-20., 800., 500.),
))
def test_get_modelgrid_transform(tmp_path, angrot, delr, delc):
    """The transform from a grid json file is the grid rotated counter-clockwise
    about its upper left corner (the MODFLOW angrot convention), including for cells
    with different row and column spacings, without any warnings
    (the Affine * Affine operation is deprecated in affine 3.0)."""
    xul, yul = 500955., 1205285.
    grid_file = write_grid_json(tmp_path / 'grid.json', angrot, xul, yul, delr, delc)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        transform = get_modelgrid_transform(grid_file)
    expected = get_rotated_transform(angrot, xul, yul, delr, delc)
    assert np.allclose(tuple(transform), tuple(expected))
    # same properties as the transform in the get_ij docstring
    assert np.allclose((transform.c, transform.f), (xul, yul))
    assert_grid_axes(transform, angrot, delr, delc)
    # the transform locates the first cell center, and get_ij maps it back to row 0, column 0
    x = transform.a * 0.5 + transform.b * 0.5 + transform.c
    y = transform.d * 0.5 + transform.e * 0.5 + transform.f
    assert get_ij(transform, x, y) == (0, 0)


@pytest.mark.parametrize('scalar_spacing', (False, True))
def test_get_modelgrid_transform_scalar_or_list_spacing(tmp_path, scalar_spacing):
    """delr and delc can be lists or scalars."""
    xul, yul, delr, delc = 500955., 1205285., 500., 800.
    grid_file = write_grid_json(tmp_path / 'grid.json', 0., xul, yul, delr, delc,
                                scalar_spacing=scalar_spacing)
    transform = get_modelgrid_transform(grid_file)
    assert np.allclose(tuple(transform), tuple(Affine(delr, 0., xul, 0., -delc, yul)))


def apply_transform(transform, col, row):
    """Real-world x, y of a (col, row) location (without the Affine * (col, row)
    operation that is deprecated in affine 3.0)."""
    col, row = np.asarray(col, dtype=float), np.asarray(row, dtype=float)
    return (transform.a * col + transform.b * row + transform.c,
            transform.d * col + transform.e * row + transform.f)


@pytest.mark.parametrize('angrot,delr,delc', (
    (0., 500., 500.),
    (30., 500., 500.),
    (45., 500., 500.),
    (-20., 500., 500.),
    (0., 500., 800.),
    (30., 500., 800.),
    (-20., 800., 500.),
))
def test_get_modelgrid_transform_shift_to_cell_centers(tmp_path, angrot, delr, delc):
    """With shift_to_cell_centers=True, the transform locates (col, row) = (0, 0) at the center of
    the first cell, whether or not the grid is rotated (the half cell shift follows the
    rotated column and row directions). The grid orientation and cell sizes are unchanged."""
    xul, yul = 500955., 1205285.
    grid_file = write_grid_json(tmp_path / 'grid.json', angrot, xul, yul, delr, delc)
    transform = get_modelgrid_transform(grid_file)
    centered = get_modelgrid_transform(grid_file, shift_to_cell_centers=True)
    # same orientation and cell sizes
    assert np.allclose(tuple(centered)[:2] + tuple(centered)[3:5],
                       tuple(transform)[:2] + tuple(transform)[3:5])
    assert_grid_axes(centered, angrot, delr, delc)
    # (col, row) in the centered transform is the center of cell col, row
    # (in the unshifted transform, cell centers are at col + 0.5, row + 0.5)
    cols, rows = np.array([0, 1, 3, 2]), np.array([0, 2, 1, 4])
    assert np.allclose(apply_transform(centered, cols, rows),
                       apply_transform(transform, cols + 0.5, rows + 0.5))
    # the center of the first cell is half a column and half a row from the upper left corner
    x0, y0 = apply_transform(centered, 0, 0)
    assert np.isclose(np.hypot(x0 - xul, y0 - yul), 0.5 * np.hypot(delr, delc))
    if angrot == 0.:
        # unrotated: half a cell east, and half a cell south
        assert np.allclose((x0, y0), (xul + delr / 2, yul - delc / 2))


@pytest.mark.parametrize('angrot,delr,delc', (
    (0., 500., 500.),
    (30., 500., 500.),
    (30., 500., 300.),
))
def test_get_ij_no_warnings_scalar_and_array(angrot, delr, delc):
    """get_ij gives the row and column of cell centers for scalar and array inputs,
    for rotated and unrotated grids with square and rectangular cells, without raising any warnings
    (the Affine * (x, y) operation is deprecated in affine 3.0)."""
    transform = get_rotated_transform(angrot, delr=delr, delc=delc)
    rows = np.array([0, 3, 20, 7])
    cols = np.array([0, 5, 2, 40])
    # cell centers in real-world coordinates
    # (written out with the transform coefficients, to avoid the deprecated operator)
    x = transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c
    y = transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        i, j = get_ij(transform, x, y)
        assert (i == rows).all() and (j == cols).all()
        # sequences and scalars
        i, j = get_ij(transform, list(x), list(y))
        assert (i == rows).all() and (j == cols).all()
        i, j = get_ij(transform, x[1], y[1])
        assert (i, j) == (3, 5)


@pytest.mark.parametrize('tdis_file,sto_file,start_date,end_date,time_units,include_timesteps',
                         (('mf2005/br_trans.dis', None, '2000-01-01', None, None, True),
                          ('mf2005/br_trans.dis', None, '2000-01-01', '2000-01-01', 'days', False),
                          ('mf2005/br_trans.dis', None, '2000-01-01', '2001-02-01', 'days', True),
                          ('shellmound/mfsim.tdis', 'shellmound/shellmound.sto', '2000-01-01', None, None, True),
                          ('shellmound/mfsim.tdis', 'shellmound/shellmound.sto', None, None, None, False),
                         ))    
def test_get_perioddata(tdis_file, sto_file, start_date, end_date, 
                        time_units, include_timesteps, test_data_path):
    tdis_file = test_data_path / tdis_file
    if sto_file is not None:
        sto_file = test_data_path / sto_file
    results = get_perioddata(tdis_file, sto_file, start_datetime=start_date, end_datetime=end_date, 
                             include_timesteps=include_timesteps, model_time_units=time_units)
    if not include_timesteps:
        elapsed_time = np.cumsum(results['perlen'].values) -1
        elapsed_time[0] = 1
        assert pd.api.types.is_string_dtype(results['start_datetime'].dtype)
        assert pd.api.types.is_string_dtype(results['end_datetime'].dtype)
        expected_end_datetimes = pd.Timestamp(results['start_datetime'][0]) + \
            pd.to_timedelta(elapsed_time -1, unit='d')
        assert all(results['end_datetime'] == expected_end_datetimes.strftime('%Y-%m-%d'))
    assert pd.api.types.is_integer_dtype(results['nstp'])
    
    if not include_timesteps:
        # explicitly check for time units of days
        # (either entered or read)
        ndays = (pd.Timestamp(results['end_datetime'].values[-1]) - 
                pd.Timestamp(results['start_datetime'].values[-1])).days
        assert ndays == results['perlen'].values[-1] - 1


def test_add_timesteps_to_perioddata():
    perioddata = pd.DataFrame({
        'start_datetime': ['2000-01-01', '2000-01-01'],
        'end_datetime': ['2000-01-01', '2000-01-11'],
        'time': [1, 11.],
        'per': [0, 1],
        'perlen': [1, 10.],
        'nstp': [1, 5],
        'tsmult': [1, 1.5],
        })
    results = add_timesteps_to_perioddata(perioddata)
    ti1 = 10*(1.5-1)/(1.5**5 -1)
    expected_timesteps = [1]
    expected_timesteps += list(1 + np.cumsum([ti1] + list(ti1 * np.cumprod([1.5] * (5 -1)))))
    assert np.allclose(results.time, expected_timesteps)
    assert not results.isna().any().any()
    assert results['timestep'].values[:2].sum() == 0
        

@pytest.mark.parametrize('fileinput,variable,expected_varname,expected_values', 
                         (('mf2005/badger_mill_ck.ggo', None, None, None),
                          ('mf2005/badger_mill_ck.ggo', 'Hyd.Grad.', 'hydgrad', None),
                          ('mf2005/P32S.ggo', 'gw-head', 'gw-head', [919.12427, 916.03711]),
                          (['mf2005/badger_mill_ck.ggo', 'mf2005/badger_mill_ck2.ggo'], 'Flow', 'flow', \
                           [[101740.06, 137382.61], [142659.840, 978764.000]])
                          )
                         )
def test_read_mf_gage_package_output_files(fileinput, variable, expected_varname, expected_values, test_data_path):
    
    if isinstance(fileinput, str):
        fileinput = test_data_path / fileinput
    else:
        fileinput = [test_data_path / f for f in fileinput]
        # test str as well as pathlike
        fileinput[0] = str(fileinput[0])
    results = read_mf_gage_package_output_files(fileinput, variable)
    assert results.index.name == 'time'
    if expected_varname is None:
        assert np.all(results.columns == 
                      ['time', 'stage', 'flow', 'depth', 'width', 'midpt-flow', 
                       'precip', 'et', 'sfr-runoff', 'uzf-runoff', 
                       'conductance', 'headdiff', 'hydgrad'])
    else:
        if isinstance(fileinput, str) or isinstance(fileinput, Path):
            fileinput = [fileinput]
        for i, col in enumerate(results.columns):
            obsprefix = col.split('-')[0]
            variable = col.split(f'{obsprefix}-')[1]
            assert Path(fileinput[i]).stem.lower().replace('-', '').replace('_', '') == obsprefix
            assert variable == expected_varname
    assert not results.isna().any().any()
    if expected_values is not None:
        assert np.all(results.iloc[0] == expected_values[0])
        assert np.all(results.iloc[-1] == expected_values[-1])
        

def test_get_mf_gage_package_obs(test_data_path):
    gage_file = test_data_path / 'mf2005/badger_mill_ck2.ggo'
    perioddata = get_perioddata(test_data_path / 'mf2005/br_trans.dis', start_datetime='2000-01-01',
                                end_datetime='2001-02-01', include_timesteps=True)
    gage_results = get_mf_gage_package_obs(perioddata, gage_file, variable='flow',
                                           enfore_absolute_values=True)
    assert np.all(gage_results.columns ==
                ['datetime', 'site_no', 'variable', 'obsprefix',
                'sim_obsval', 'time', 'per']
                )
    loc = (gage_results.time == 2) & (gage_results.site_no == 'badgermillck2')
    assert gage_results.loc[loc, 'sim_obsval'].values[0] == 132293.11
    assert gage_results.loc[loc, 'variable'].values[0] == 'flow'
    assert gage_results.loc[loc, 'obsprefix'].values[0] == 'badgermillck2-flow'
    assert gage_results.loc[loc, 'datetime'].values[0] == '2000-01-01'
    assert np.all(gage_results['per'].values == [0] + [1] * (len(gage_results)-1))


@pytest.mark.parametrize('variable', ('head', None))
def test_get_mf6_single_variable_obs(test_data_path, variable):
    tdis_file = test_data_path / 'shellmound/mfsim.tdis'
    sto_file = test_data_path / 'shellmound/shellmound.sto'
    model_output_file = test_data_path / 'shellmound/shellmound.head.obs'
    gwf_obs_input_file = test_data_path / 'shellmound/shellmound.obs'

    perioddata = get_perioddata(tdis_file, sto_file, #start_datetime=start_date, end_datetime=end_date, 
                                model_time_units='days')
    results = get_mf6_single_variable_obs(perioddata,
                                          model_output_file,
                                          gwf_obs_input_file=gwf_obs_input_file,
                                          variable=variable,
                                          enfore_absolute_values=True)
    assert np.all(results.columns ==
                ['datetime', 'site_no', 'variable', 'obsprefix',
                'sim_obsval', 'layer', 'time', 'per']
                )
    # obsprefixes should include variable if one is specified
    if variable is not None:
        assert results.obsprefix.values[0].split('-')[1] == variable
    else:
        assert len(results.obsprefix.values[0].split('-')) == 1

@pytest.mark.parametrize('gwf_obs_input_file', (None, 'mf6/br_mf6.obs'))
def test_get_mf6_single_variable_obs_daily(test_data_path, gwf_obs_input_file):
    model_output_file = test_data_path / 'mf6/head.obs'
    if gwf_obs_input_file is not None:
        gwf_obs_input_file = test_data_path / 'mf6/br_mf6.obs'

    perioddata = get_perioddata(test_data_path / 'mf2005/br_trans.dis', start_datetime='2000-01-01',
                            end_datetime='2001-02-01', include_timesteps=True)
    results = get_mf6_single_variable_obs(perioddata,
                                          model_output_file,
                                          gwf_obs_input_file=gwf_obs_input_file,
                                          variable='head',
                                          enfore_absolute_values=True,
                                          gwf_obs_block=None)
    # 1 steady-state period; 39 timesteps in transient period
    assert len(results.datetime.unique()) == 39

@pytest.mark.parametrize('gwf_obs_block', (None, 0, 1))
def test_get_gwf_obs(test_data_path, gwf_obs_block):
    gwf_obs_input_file = test_data_path / 'shellmound/shellmound.alt.obs'
    gwf_obs_input = get_gwf_obs_input(gwf_obs_input_file, gwf_obs_block)
    if gwf_obs_block is None:
        gwf_obs_input.shape[0] == 489
    elif gwf_obs_block == 0:
        gwf_obs_input.shape[0] == 389
    else:
        gwf_obs_input.shape[0] == 109
        

@pytest.mark.parametrize('obspefix,arg,expected', (
    ('15266500', True, '15266500'),
    ('15266500-flow', True, '15266500'),
    ('bc-east-fork-s12-flow', True, 'bc-east-fork-s12'),
    ('bc-east-fork-s12', False, 'bc-east-fork-s12'),
    ('15266500.1', False, '15266500'),
    ('15266500-flow.1', True, '15266500')
                         ))
def test_get_site_no(obspefix, arg, expected):
    output = get_site_no(obspefix, obsprefix_includes_variable=arg)
    assert output == expected
    

def test_get_transmissivities():
    sctop = [-0.25, 0.5, 1.7, 1.5, 3.0, 2.5, 3.0, -10.0]
    scbot = [-1.0, -0.5, 1.2, 0.5, 1.5, -0.2, 2.5, -11.0]
    heads = np.array(
        [
            [1.0, 2.0, 2.05, 3.0, 4.0, 2.5, 2.5, 2.5],
            [1.1, 2.1, 2.2, 2.0, 3.5, 3.0, 3.0, 3.0],
            [1.2, 2.3, 2.4, 0.6, 3.4, 3.2, 3.2, 3.2],
        ]
    )
    nl, nr = heads.shape
    nc = nr
    botm = np.ones((nl, nr, nc), dtype=float)
    top = np.ones((nr, nc), dtype=float) * 2.1
    hk = np.ones((nl, nr, nc), dtype=float) * 2.0
    for i in range(nl):
        botm[nl - i - 1, :, :] = i

    #m = Modflow("junk", version="mfnwt", model_ws=function_tmpdir)
    #dis = ModflowDis(m, nlay=nl, nrow=nr, ncol=nc, botm=botm, top=top)
    #upw = ModflowUpw(m, hk=hk)

    #get_transmissivities(heads, hk, top, botm,
    #                        r=None, c=None, x=None, y=None, modelgrid_transform=None,
    #                        screen_top=None, screen_botm=None, 
    #                        include_zero_open_intervals=True,
    #                        nodata=-999)

    # test with open intervals
    i, j = np.arange(nr), np.arange(nc)
    T = get_transmissivities(heads, hk=hk, top=top, botm=botm, 
                             i=i, j=j,
                             screen_top=sctop, screen_botm=scbot)
    assert (
        T
        - np.array(
            [
                [0.0, 0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.2, 0.0, 2.0, 0.0, 2.0],
            ]
        )
    ).sum() < 1e-3

    # test without specifying open intervals
    T = get_transmissivities(heads, hk=hk, top=top, botm=botm, 
                             i=i, j=j)
    assert (
        T
        - np.array(
            [
                [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 1.2, 2.0, 2.0, 2.0, 2.0],
            ]
        )
    ).sum() < 1e-3