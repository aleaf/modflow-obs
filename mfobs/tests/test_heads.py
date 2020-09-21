"""
Tests for the heads.py module

"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import affine
from mfobs.heads import get_head_obs, get_temporal_head_difference_obs
from mfobs.modflow import get_modelgrid_transform


@pytest.fixture
def shellmound_grid_transform(shellmound_data_path):
    return get_modelgrid_transform(shellmound_data_path / 'shellmound_grid.json')


@pytest.fixture
def shellmound_data_path(test_data_path):
    return test_data_path / 'shellmound'


@pytest.fixture
def shellmound_output_path(test_output_folder):
    output_path = test_output_folder / 'shellmound'
    if not output_path.exists():
        output_path.mkdir()
    return output_path


@pytest.fixture
def head_obs_input(shellmound_data_path, shellmound_grid_transform,
                   shellmound_output_path):

    class HeadObsInput:
        modelgrid_transform = shellmound_grid_transform
        perioddata = pd.read_csv(shellmound_data_path / 'tables/stress_period_data.csv')
        headobs_input_file = shellmound_data_path / 'shellmound.obs'
        headobs_output_file = shellmound_data_path / 'shellmound.head.obs'
        head_obs_file = shellmound_data_path / 'tables/preprocessed_head_obs.csv'
        head_obs_info_file = shellmound_data_path / 'tables/preprocessed_head_obs_info.csv'
        top_array = shellmound_data_path / 'external/top.dat'
        botm_arrays = [shellmound_data_path / 'external/botm{}.dat'.format(i)
                       for i in range(13)]
        hk_arrays = [shellmound_data_path / 'external/k{}.dat'.format(i)
                     for i in range(13)]
        outfile = shellmound_output_path / 'processed_head_obs.dat'
        inset_thead_diff_obs_outfile = shellmound_output_path / 'processed_thead_diff_obs.dat'
        label_period_as_steady_state = 4
        steady_state_period_start = '2008-04-01'
        steady_state_period_end = '2008-9-30'

    return HeadObsInput


@pytest.fixture
def head_obs(head_obs_input):
    head_obs = get_head_obs(head_obs_input.perioddata,
                            modelgrid_transform=head_obs_input.modelgrid_transform,
                            model_output_file=head_obs_input.headobs_output_file,
                            observed_values_file=head_obs_input.head_obs_file,
                            observed_values_metadata_file=head_obs_input.head_obs_info_file,
                            observed_values_obsval_col='head_m',
                            gwf_obs_input_file=head_obs_input.headobs_input_file,
                            hk_arrays=head_obs_input.hk_arrays,
                            top_array=head_obs_input.top_array,
                            botm_arrays=head_obs_input.botm_arrays,
                            label_period_as_steady_state=head_obs_input.label_period_as_steady_state,
                            steady_state_period_start=head_obs_input.steady_state_period_start,
                            steady_state_period_end=head_obs_input.steady_state_period_end,
                            write_ins=False, outfile=head_obs_input.outfile)
    return head_obs


def test_get_head_obs(test_data_path, head_obs_input, head_obs, shellmound_output_path):

    # fixture generates head_obs without writing ins
    assert Path(shellmound_output_path, 'processed_head_obs.dat').exists()
    assert not Path(shellmound_output_path, 'processed_head_obs.dat.ins').exists()
    Path(shellmound_output_path, 'processed_head_obs.dat').unlink()  # delete it
    assert np.all(head_obs.columns ==
                  ['datetime', 'per', 'obsprefix', 'obsnme',
                   'obs_head', 'sim_head', 'screen_top', 'screen_botm'])
    assert len(set(head_obs.obsnme)) == len(head_obs)
    assert not head_obs.obs_head.isna().any()
    assert head_obs.obsnme.str.islower().all()

    # check sorting
    assert np.all(head_obs.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)

    # test write_ins=True
    results = get_head_obs(head_obs_input.perioddata,
                           modelgrid_transform=head_obs_input.modelgrid_transform,
                           model_output_file=head_obs_input.headobs_output_file,
                           observed_values_file=head_obs_input.head_obs_file,
                           observed_values_metadata_file=head_obs_input.head_obs_info_file,
                           observed_values_obsval_col='head_m',
                           gwf_obs_input_file=head_obs_input.headobs_input_file,
                           hk_arrays=head_obs_input.hk_arrays,
                           top_array=head_obs_input.top_array,
                           botm_arrays=head_obs_input.botm_arrays,
                           label_period_as_steady_state=head_obs_input.label_period_as_steady_state,
                           steady_state_period_start=head_obs_input.steady_state_period_start,
                           steady_state_period_end=head_obs_input.steady_state_period_end,
                           write_ins=True, outfile=head_obs_input.outfile)
    assert Path(shellmound_output_path, 'processed_head_obs.dat.ins').exists()


@pytest.mark.parametrize('write_ins', (True,
                                       False
                                       ))
def test_get_temporal_head_difference_obs(head_obs, head_obs_input, write_ins):
    results = get_temporal_head_difference_obs(head_obs,
                                               head_obs_input.perioddata,
                                               write_ins=write_ins,
                                               outfile=head_obs_input.inset_thead_diff_obs_outfile)
    assert head_obs_input.inset_thead_diff_obs_outfile.exists()
    insfile = Path(str(head_obs_input.inset_thead_diff_obs_outfile) + '.ins')
    if not write_ins:
        assert not insfile.exists()
    else:
        assert insfile.exists()
        insfile.unlink()
    head_obs_input.inset_thead_diff_obs_outfile.unlink()  # delete it

    assert np.all(results.columns ==
                  ['datetime', 'per', 'obsprefix', 'obsnme',
                   'obs_head', 'sim_head', 'screen_top', 'screen_botm',
                   'obsval', 'sim_obsval', 'group'])
    assert len(set(results.obsnme)) == len(results)
    assert not results.obs_head.isna().any()
    assert results.obsnme.str.islower().all()
    suffixes = np.ravel([obsnme.split('_')[1].split('d') for obsnme in results.obsnme])
    assert 'ss' not in suffixes
    j=2
