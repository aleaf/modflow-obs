"""
Tests for the swflows.py module

"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from mfobs.obs import get_spatial_differences, get_temporal_differences
from mfobs.swflows import get_flux_obs


@pytest.fixture
def flux_obs_input(shellmound_data_path):
    class FluxObsInput:
        perioddata = pd.read_csv(shellmound_data_path / 'tables/stress_period_data.csv')
        model_output_file = shellmound_data_path / 'shellmound.sfr.obs.output.csv'
        observed_values_file = shellmound_data_path / 'tables/processed_flow_obs.csv'
        tflux_diff_obs_outfile = shellmound_data_path / 'tables/processed_flow_obs_tdiffs.csv'
        sflux_diff_obs_outfile = shellmound_data_path / 'tables/processed_flow_obs_sdiffs.csv'
    return FluxObsInput


@pytest.fixture
def flux_obs(flux_obs_input):
    results = get_flux_obs(flux_obs_input.perioddata,
                           model_output_file=flux_obs_input.model_output_file,
                           observed_values_file=flux_obs_input.observed_values_file,
                           observed_values_site_id_col='site_no',
                           observed_values_obsval_col='obsval',
                           variable_name='flux',
                           outfile=None,
                           write_ins=False)
    return results


def test_get_flux_obs(flux_obs_input, flux_obs):

    results = flux_obs
    expected_columns = ['datetime', 'per', 'obsprefix', 'obsnme',
                        'obs_flux', 'sim_flux']
    assert np.all(results.columns == expected_columns)
    assert len(set(results.obsnme)) == len(results)
    assert not results.obs_flux.isna().any()
    assert not results.obsnme.str.isupper().any()

    # check sorting
    assert np.all(results.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)


@pytest.mark.parametrize('write_ins', (True,
                                       False
                                       ))
def test_get_temporal_flux_differences(flux_obs, flux_obs_input, write_ins):
    results = get_temporal_differences(flux_obs,
                                       flux_obs_input.perioddata,
                                       obs_values_col='obs_flux',
                                       sim_values_col='sim_flux',
                                       obstype='flux',
                                       write_ins=write_ins,
                                       outfile=flux_obs_input.tflux_diff_obs_outfile)
    assert flux_obs_input.tflux_diff_obs_outfile.exists()
    insfile = Path(str(flux_obs_input.tflux_diff_obs_outfile) + '.ins')
    if not write_ins:
        assert not insfile.exists()
    else:
        assert insfile.exists()
        insfile.unlink()
    flux_obs_input.tflux_diff_obs_outfile.unlink()  # delete it

    assert np.all(results.columns ==
                  ['datetime', 'per', 'obsprefix', 'obsnme',
                   'obs_flux', 'sim_flux',
                   'obsval', 'sim_obsval', 'group', 'type']
                  )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    suffixes = np.ravel([obsnme.split('_')[1].split('d') for obsnme in results.obsnme])
    assert 'ss' not in suffixes


@pytest.mark.parametrize('flux_difference_sites, write_ins',
                         (({'07288280':  # sunflower r. at merigold
                           '07288500'   # sunflower r. at sunflower
                             }, True),
                          ({'07288500': '07288280' }, False)))
def test_get_spatial_flux_difference_obs(flux_obs, flux_obs_input, flux_difference_sites, write_ins):

    results = get_spatial_differences(flux_obs, flux_obs_input.perioddata,
                                      flux_difference_sites,
                                      obs_values_col='obs_flux',
                                      sim_values_col='sim_flux',
                                      obstype='flux',
                                      use_gradients=False,
                                      write_ins=write_ins,
                                      outfile=flux_obs_input.sflux_diff_obs_outfile)
    assert flux_obs_input.sflux_diff_obs_outfile.exists()
    insfile = Path(str(flux_obs_input.sflux_diff_obs_outfile) + '.ins')
    if not write_ins:
        assert not insfile.exists()
    else:
        assert insfile.exists()
        insfile.unlink()
    flux_obs_input.sflux_diff_obs_outfile.unlink()  # delete it

    assert np.all(results.columns ==
                  ['datetime', 'per', 'obsprefix',
                   'obsnme1', 'obs_flux1', 'sim_flux1',
                   'obsnme2', 'obs_flux2', 'sim_flux2',
                   'obs_diff', 'sim_diff', 'group', 'obsnme',
                   'obsval', 'sim_obsval', 'type']
                  )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()