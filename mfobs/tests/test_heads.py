"""
Tests for the heads.py module

"""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import affine
from mfobs.heads import get_head_obs
from mfobs.modflow import (
    get_modelgrid_transform, 
    get_ij, 
    get_mf6_single_variable_obs)
from mfobs.obs import (get_spatial_differences, 
                       get_temporal_differences, 
                       )

@pytest.fixture
def shellmound_grid_transform(shellmound_data_path):
    return get_modelgrid_transform(shellmound_data_path / 'shellmound_grid.json')


@pytest.fixture
def head_obs_input(shellmound_data_path, shellmound_grid_transform,
                   shellmound_output_path):

    # make a layer column
    head_obs_info = pd.read_csv(shellmound_data_path / 'tables/preprocessed_head_obs_info.csv')
    layers = {'mrva': 0,
              'vicksburg': 1,
              'upper claiborne': 1,
              'lower claiborne': 2,
              'middle wilcox': 2,
              'lower wilcox': 2
              }
    head_obs_info['layer'] = [layers[code] for code in head_obs_info.regional_aquifer]

    class HeadObsInput:
        modelgrid_transform = shellmound_grid_transform
        perioddata = pd.read_csv(shellmound_data_path / 'tables/stress_period_data.csv')
        headobs_input_file = shellmound_data_path / 'shellmound.obs'
        headobs_output_file = shellmound_data_path / 'shellmound.head.obs'
        head_obs_file = shellmound_data_path / 'tables/preprocessed_head_obs.csv'
        head_obs_info_file = head_obs_info
        top_array = shellmound_data_path / 'external/top.dat'
        botm_arrays = [shellmound_data_path / 'external/botm{}.dat'.format(i)
                       for i in range(13)]
        hk_arrays = [shellmound_data_path / 'external/k{}.dat'.format(i)
                     for i in range(13)]
        outfile = shellmound_output_path / 'processed_head_obs.dat'
        thead_diff_obs_outfile = shellmound_output_path / 'processed_thead_diff_obs.dat'
        shead_diff_obs_outfile = shellmound_output_path / 'processed_shead_diff_obs.dat'
        label_period_as_steady_state = 4
        steady_state_period_start = '2008-04-01'
        steady_state_period_end = '2008-9-30'

    return HeadObsInput


@pytest.fixture()
def head_obs(head_obs_input):
    head_obs = get_head_obs(head_obs_input.perioddata,
                            modelgrid_transform=head_obs_input.modelgrid_transform,
                            model_output_file=head_obs_input.headobs_output_file,
                            observed_values_file=head_obs_input.head_obs_file,
                            observed_values_metadata_file=head_obs_input.head_obs_info_file,
                            observed_values_obsval_col='head',
                            observed_values_layer_col=None,
                            gwf_obs_input_file=head_obs_input.headobs_input_file,
                            hk_arrays=head_obs_input.hk_arrays,
                            top_array=head_obs_input.top_array,
                            botm_arrays=head_obs_input.botm_arrays,
                            label_period_as_steady_state=head_obs_input.label_period_as_steady_state,
                            steady_state_period_start=head_obs_input.steady_state_period_start,
                            steady_state_period_end=head_obs_input.steady_state_period_end,
                            write_ins=False, outfile=head_obs_input.outfile)
    return head_obs


@pytest.mark.parametrize(('observed_values_layer_col,'
                          'steady,obsnme_date_suffix,'
                          'obsnme_suffix_format'), 
                         ((None, True, False, '03d'),
                          (None, False, False, '{:03d}'),
                          ('layer', True, True, '%Y%m'),
                          ('layer', True, True, '%Y%m%d'),
                          ))
def test_get_head_obs(test_data_path, head_obs_input, head_obs,
                      shellmound_output_path, observed_values_layer_col, steady,
                      obsnme_date_suffix, obsnme_suffix_format):

    # test min/max layers where screen top == screen bottom
    loc = head_obs['obsprefix'] == 'usgs:333145090261901'
    assert np.allclose(head_obs.loc[loc, ['screen_top']], 
                       head_obs.loc[loc, ['screen_botm']])
    assert (head_obs.loc[loc, ['min_layer', 'max_layer']] == 4).all().all()

    # fixture generates base_data without writing ins
    assert Path(shellmound_output_path, 'processed_head_obs.dat').exists()
    assert not Path(shellmound_output_path, 'processed_head_obs.dat.ins').exists()
    Path(shellmound_output_path, 'processed_head_obs.dat').unlink()  # delete it
    expected_columns = ['datetime', 'per', 'site_no', 'obsprefix', 'obsnme',
                        'obsval', 'sim_obsval', 'n', 'screen_top', 'screen_botm', 'layer',
                        'min_layer', 'max_layer', 'i', 'j', 'obgnme'
                        ]
    assert np.all(head_obs.columns == expected_columns)
    assert len(set(head_obs.obsnme)) == len(head_obs)
    assert not head_obs['obsval'].isna().any()
    assert not head_obs.obsnme.str.isupper().any()

    # check sorting
    assert np.all(head_obs.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)

    # test with specified layer and T-weighted averaging
    if observed_values_layer_col is not None:
        head_obs_input.hk_arrays = None
        head_obs_input.top_array = None
        head_obs_input.botm_arrays = None

    # test with and without steady-state observations
    if not steady:
        head_obs_input.label_period_as_steady_state = None
        head_obs_input.steady_state_period_start = None
        head_obs_input.steady_state_period_end = None

    # test write_ins=True
    results = get_head_obs(head_obs_input.perioddata,
                           modelgrid_transform=head_obs_input.modelgrid_transform,
                           model_output_file=head_obs_input.headobs_output_file,
                           observed_values_file=head_obs_input.head_obs_file,
                           observed_values_metadata_file=head_obs_input.head_obs_info_file,
                           observed_values_obsval_col='head',
                           observed_values_layer_col=observed_values_layer_col,
                           obsnme_date_suffix=obsnme_date_suffix,
                           obsnme_suffix_format=obsnme_suffix_format,
                           gwf_obs_input_file=head_obs_input.headobs_input_file,
                           hk_arrays=head_obs_input.hk_arrays,
                           top_array=head_obs_input.top_array,
                           botm_arrays=head_obs_input.botm_arrays,
                           label_period_as_steady_state=head_obs_input.label_period_as_steady_state,
                           steady_state_period_start=head_obs_input.steady_state_period_start,
                           steady_state_period_end=head_obs_input.steady_state_period_end,
                           write_ins=True, outfile=head_obs_input.outfile)
    assert Path(shellmound_output_path, 'processed_head_obs.dat.ins').exists()
    Path(shellmound_output_path, 'processed_head_obs.dat.ins').unlink()
    
    # test observation name suffixes
    is_trans = [False if obsnme.split('_')[1] == 'ss' else True 
                for obsnme in results['obsnme']]
    # test steady-state suffixes
    if head_obs_input.label_period_as_steady_state:
        assert np.all(results.loc[~np.array(is_trans), 'per'] == \
            head_obs_input.label_period_as_steady_state)
    # stress period-based suffixes
    if not obsnme_date_suffix:
        parsed_periods = [int(obsnme.split('_')[1]) for obsnme in results.loc[is_trans, 'obsnme']]
        assert np.array_equal(parsed_periods, results.loc[is_trans, 'per'].values)
    # date-based suffixes
    else:
        timestamps = [pd.Timestamp(datetime.strptime(obsnme.split('_')[1], obsnme_suffix_format)) 
                      for obsnme in results.loc[is_trans, 'obsnme']]
        years = [ts.year for ts in timestamps]
        months = [ts.month for ts in timestamps]
        assert np.array_equal(years, results.loc[is_trans, 'datetime'].dt.year.values)
        assert np.array_equal(months, results.loc[is_trans, 'datetime'].dt.month.values)
    # todo: add specific check for steady-state observations


@pytest.mark.parametrize('write_ins,get_displacements,displacement_from', 
                         ((True, True, '2010-01-01'),
                          (False, True, None),
                          (False, False, None)
                                       ))
def test_get_temporal_head_difference_obs(head_obs, head_obs_input, 
                                          write_ins, get_displacements, 
                                          displacement_from):
    results = get_temporal_differences(head_obs,
                                       head_obs_input.perioddata,
                                       obs_values_col='obsval',
                                       sim_values_col='sim_obsval',
                                       variable='head',
                                       get_displacements=get_displacements,
                                       displacement_from=displacement_from,
                                       write_ins=write_ins,
                                       outfile=head_obs_input.thead_diff_obs_outfile)
    assert head_obs_input.thead_diff_obs_outfile.exists()
    insfile = Path(str(head_obs_input.thead_diff_obs_outfile) + '.ins')
    if not write_ins:
        assert not insfile.exists()
    else:
        assert insfile.exists()
        insfile.unlink()
    head_obs_input.thead_diff_obs_outfile.unlink()  # delete it

    assert np.all(results.columns ==
                  ['datetime', 'per', 'obsprefix', 'obsnme',
                   'obs_head', 'sim_head', 'screen_top', 'screen_botm', 'layer',
                   'obsval', 'sim_obsval', 'obgnme', 'type']
                  )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    suffixes = np.ravel([obsnme.split('_')[1].split('d') for obsnme in results.obsnme])
    assert 'ss' not in suffixes
    expected_obgnme = 'head_tdiff'
    if get_displacements:
        expected_obgnme = 'head_disp'
    assert results.obgnme.unique().tolist() == [expected_obgnme]
    
    # check displacement obs
    if get_displacements:
        first_obs = results.groupby(['per', 'obsprefix']).first()
        datums = set([obsnme.split('d')[1] for obsnme in first_obs.obsnme])
        if displacement_from is not None:
            loc = head_obs.datetime > displacement_from
        else:
            loc = slice(None, None)
        first_head_obs = head_obs.loc[loc].groupby(['obsprefix']).first()
        multiple_obs = head_obs.groupby(['obsprefix']).per.count() > 1
        first_head_obs = first_head_obs.loc[multiple_obs]
        
        expected_datums = set([obsnme.split('_')[1] for obsnme in first_head_obs.obsnme])
        assert datums == expected_datums



@pytest.mark.parametrize('write_ins', (True,
                                       False
                                       ))
def test_get_spatial_head_difference_obs(head_obs, head_obs_input, write_ins):
    head_difference_sites = {'usgs:333904090123801':  # well in money, ms
                                 'usgs:333145090261901'
                             }
    results = get_spatial_differences(head_obs, head_obs_input.perioddata,
                                      head_difference_sites,
                                      obs_values_col='obsval',
                                      sim_values_col='sim_obsval',
                                      #variable='head',
                                      use_gradients=False,
                                      write_ins=write_ins,
                                      outfile=head_obs_input.shead_diff_obs_outfile)
    assert head_obs_input.shead_diff_obs_outfile.exists()
    insfile = Path(str(head_obs_input.shead_diff_obs_outfile) + '.ins')
    if not write_ins:
        assert not insfile.exists()
    else:
        assert insfile.exists()
        insfile.unlink()
    head_obs_input.shead_diff_obs_outfile.unlink()  # delete it

    assert np.all(results.columns ==
                  ['datetime', 'per', 'obsprefix',
                   'obsnme1', 'base_obsval1', 'base_sim_obsval1', 'screen_top1', 'screen_botm1', 'layer1',
                   'obsnme2', 'base_obsval2', 'base_sim_obsval2', 'screen_top2', 'screen_botm2', 'layer2',
                   'obs_diff', 'sim_diff', 'obsnme',
                   'obsval', 'sim_obsval', 'obgnme', 'type'])
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    assert results.obgnme.unique().tolist() == ['head_sdiff']


@pytest.mark.parametrize('forecast_sites,forecast_start_date,forecast_end_date,forecasts_only', (
    ('all', '2010-01-01', '2015-01-01', True),
    (['usgs:333145090261901', 'usgs:334630090274201'], None, None, False)
))
def test_get_forecast_head_obs(head_obs_input, forecast_sites, 
                               forecast_start_date, forecast_end_date,
                               forecasts_only):
    
    # test input with a dataframe too
    observed_values = pd.read_csv(head_obs_input.head_obs_file,
                                  dtype={'obsprefix': object})
    
    head_obs = get_head_obs(head_obs_input.perioddata,
                            modelgrid_transform=head_obs_input.modelgrid_transform,
                            model_output_file=head_obs_input.headobs_output_file,
                            observed_values_file=observed_values,
                            observed_values_metadata_file=head_obs_input.head_obs_info_file,
                            observed_values_obsval_col='head',
                            observed_values_layer_col=None,
                            gwf_obs_input_file=head_obs_input.headobs_input_file,
                            hk_arrays=head_obs_input.hk_arrays,
                            top_array=head_obs_input.top_array,
                            botm_arrays=head_obs_input.botm_arrays,
                            label_period_as_steady_state=head_obs_input.label_period_as_steady_state,
                            steady_state_period_start=head_obs_input.steady_state_period_start,
                            steady_state_period_end=head_obs_input.steady_state_period_end,
                            forecast_sites=forecast_sites,
                            forecast_start_date=forecast_start_date,
                            forecast_end_date=forecast_end_date,
                            forecasts_only=forecasts_only,
                            write_ins=False, outfile=head_obs_input.outfile)
        
    # pick a site with an incomplete record
    # the missing times should be filled with forecast obs
    loc = (head_obs.obsprefix == 'usgs:333145090261901') & head_obs.obsval.isna()
    if forecast_start_date is not None and forecast_end_date is not None:
        assert len(head_obs.loc[loc]) == 9
    else:
        assert len(head_obs.loc[loc]) == 16
    assert set(head_obs['obgnme']) == {'head-forecast', 'head'}
    
    # outside of observed values
    # and top and bottom info at some sites
    # the resulting dataframe should be entirely filled
    loc = ~head_obs_input.head_obs_info_file[['screen_top', 
                                                         'screen_botm']].isna().any(axis=1)
    sites_with_screen_info = head_obs_input.head_obs_info_file.loc[loc, 'obsprefix'].str.lower()
    head_obs_with_screen_info = head_obs.loc[head_obs.obsprefix.isin(sites_with_screen_info)]
    assert not head_obs_with_screen_info.drop(['obsval'], axis=1).isna().any().any()
    
    # if forecast_sites != 'all':
    # check that only forecasts for specified sites were generated
    if forecast_sites is not None and forecast_sites != 'all':
        assert set(head_obs.loc[head_obs['obsval'].isna(), 'obsprefix']) ==\
            set(forecast_sites)
    # check that forecasts were only generated within specified time window
    if forecast_start_date is not None:
        assert not np.any(head_obs.loc[head_obs['obsval'].isna(), 'datetime'] <\
            forecast_start_date)
    if forecast_end_date is not None:
        assert not np.any(head_obs.loc[head_obs['obsval'].isna(), 'datetime'] >\
            forecast_end_date)
    

def read_arrays(sorted_list):
    arrays = []
    for f in sorted_list:
        arrays.append(np.loadtxt(f))
    return np.array(arrays)


def test_t_weighted_head_averaging(test_data_path, head_obs_input, head_obs,
                      shellmound_output_path):

    obs_info = head_obs_input.head_obs_info_file
    head_obs.dropna(subset='screen_botm', axis=0, inplace=True)
    head_obs = head_obs.groupby('obsprefix').first()
    
    # check that open intervals are correct
    np.allclose(obs_info.loc[head_obs.index, 'screen_top'].values, 
                head_obs['screen_top'].values, atol=0.01, equal_nan=True)
    np.allclose(obs_info.loc[head_obs.index, 'screen_botm'].values, 
                head_obs['screen_botm'].values, atol=0.01, equal_nan=True)

    i, j = get_ij(head_obs_input.modelgrid_transform, 
                  obs_info.loc[head_obs.index, 'x'], 
                  obs_info.loc[head_obs.index, 'y'])
    head_obs['i'], head_obs['j'] = i, j
    botm = read_arrays(head_obs_input.botm_arrays)
    top = np.loadtxt(head_obs_input.top_array)
    hk = read_arrays(head_obs_input.hk_arrays)
    
    hk2d = hk[:, i, j]
    botm2d = botm[:, i, j]
    screen_top = top[i, j]
    
    mf_output = get_mf6_single_variable_obs(
        head_obs_input.perioddata, 
        model_output_file=head_obs_input.headobs_output_file,
        gwf_obs_input_file=head_obs_input.headobs_input_file
    )
    
    # check min/max layer
    head_obs['layer_botm'] = botm[head_obs['max_layer'], head_obs['i'], head_obs['j']]
    head_obs['layer_top'] = botm[head_obs['min_layer'] -1, head_obs['i'], head_obs['j']]
    assert np.all(head_obs['layer_top'] > head_obs['screen_top'])
    assert np.all(head_obs['screen_botm'] >= head_obs['layer_botm'])
    
    # check single layer obs
    in_single_layer = head_obs['min_layer'] == head_obs['max_layer']
    head_obs_sl = head_obs.loc[in_single_layer].copy()
    grouped = mf_output.groupby(['obsprefix', 'per', 'layer'])
    sim_heads = []
    for obsprefix, r in head_obs_sl.iterrows():
        sim_head = grouped.get_group((obsprefix, r['per'], r['min_layer']))['sim_obsval'].values[0]
        sim_heads.append(sim_head)
    assert np.allclose(head_obs_sl['sim_obsval'], sim_heads)
    
    # check multiplayer (T-weighted) obs
    head_obs_ml = head_obs.loc[~in_single_layer].copy()
    head_obs_ml['interval_top'] = head_obs_ml['screen_top']
    head_below_sctop = head_obs_ml['sim_obsval'] < head_obs_ml['screen_top']
    head_obs_ml.loc[head_below_sctop, 'interval_top'] = head_obs_ml.loc[head_below_sctop, 'sim_obsval']
    grouped = mf_output.groupby(['obsprefix', 'per'])
    sim_heads_list = []
    for obsprefix, r in head_obs_ml.iterrows():
        sim_heads = grouped.get_group((obsprefix, r['per'])).copy()
        sim_heads.index = sim_heads['layer']
        sim_heads = sim_heads['sim_obsval']
        # account for water table position in first thickness
        # (minimum of screen top or water table)
        b = [r['interval_top'] - botm[r['min_layer'], r['i'], r['j']]]
        # subsequent thicknesses are the cell thicknesses
        thicknesses = np.abs(np.diff(botm[:, r['i'], r['j']])).tolist()
        thicknesses = [top[r['i'], r['j']] - botm[0, r['i'], r['j']]] +\
            thicknesses
        b += thicknesses[r['min_layer'] + 1: r['max_layer']]
        # account for screen bottom being above the last layer bottom
        b += [botm[r['max_layer'] - 1, r['i'], r['j']] - r['screen_botm']]
        hk_interval = hk[r['min_layer']: r['max_layer'] + 1, r['i'], r['j']]
        T = hk_interval * np.array(b)
        Tfrac = T/T.sum()
        # min and max layer are zero-based
        layers = np.arange(r['min_layer'], r['max_layer']+1)
        T_weighted_head = np.sum(sim_heads.loc[layers].values * Tfrac)
        sim_heads_list.append(T_weighted_head)
    # consideration of sat. thickness should be required to pass this
    assert np.allclose(head_obs_ml['sim_obsval'].values, sim_heads_list, rtol=1e-8, atol=1e-8)