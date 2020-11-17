"""
Tests for the heads.py module

"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import affine
from mfobs.heads import get_head_obs
from mfobs.modflow import get_modelgrid_transform
from mfobs.obs import get_spatial_differences, get_temporal_differences


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


@pytest.mark.parametrize('observed_values_layer_col,steady', ((None, True),
                                                              (None, False),
                                                              ('layer', True)
                                                              ))
def test_get_head_obs(test_data_path, head_obs_input, head_obs,
                      shellmound_output_path, observed_values_layer_col, steady):

    # fixture generates base_data without writing ins
    assert Path(shellmound_output_path, 'processed_head_obs.dat').exists()
    assert not Path(shellmound_output_path, 'processed_head_obs.dat.ins').exists()
    Path(shellmound_output_path, 'processed_head_obs.dat').unlink()  # delete it
    expected_columns = ['datetime', 'per', 'obsprefix', 'obsnme',
                        'obs_head', 'sim_head', 'n', 'screen_top', 'screen_botm', 'layer',
                        'obsval', 'obgnme'
                        ]
    assert np.all(head_obs.columns == expected_columns)
    assert len(set(head_obs.obsnme)) == len(head_obs)
    assert not head_obs.obs_head.isna().any()
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

    # todo: add specific check for steady-state observations


@pytest.mark.parametrize('write_ins', (True,
                                       False
                                       ))
def test_get_temporal_head_difference_obs(head_obs, head_obs_input, write_ins):
    results = get_temporal_differences(head_obs,
                                       head_obs_input.perioddata,
                                       obs_values_col='obs_head',
                                       sim_values_col='sim_head',
                                       obstype='head',
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
    assert results.obgnme.unique().tolist() == ['head_tdiff']


@pytest.mark.parametrize('write_ins', (True,
                                       False
                                       ))
def test_get_spatial_head_difference_obs(head_obs, head_obs_input, write_ins):
    head_difference_sites = {'usgs:333904090123801':  # well in money, ms
                                 'usgs:333145090261901'
                             }
    results = get_spatial_differences(head_obs, head_obs_input.perioddata,
                                      head_difference_sites,
                                      obs_values_col='obs_head',
                                      sim_values_col='sim_head',
                                      obstype='head',
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
                   'obsnme1', 'obs_head1', 'sim_head1', 'screen_top1', 'screen_botm1', 'layer1',
                   'obsnme2', 'obs_head2', 'sim_head2', 'screen_top2', 'screen_botm2', 'layer2',
                   'obs_diff', 'sim_diff', 'dz', 'obs_grad', 'sim_grad', 'obgnme', 'obsnme',
                   'obsval', 'sim_obsval', 'type'])
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    assert results.obgnme.unique().tolist() == ['head_sdiff']
