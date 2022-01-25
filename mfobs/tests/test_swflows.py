"""
Tests for the swflows.py module

"""
from datetime import datetime
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
                           observed_values_group_column='category',
                           variable_name='flux',
                           outfile=None,
                           write_ins=False)
    return results


@pytest.fixture
def flux_obs_per_based_suffixes(flux_obs_input):
    results = get_flux_obs(flux_obs_input.perioddata,
                           model_output_file=flux_obs_input.model_output_file,
                           observed_values_file=flux_obs_input.observed_values_file,
                           observed_values_site_id_col='site_no',
                           observed_values_obsval_col='obsval',
                           observed_values_group_column='category',
                           obsnme_date_suffix=False,
                           obsnme_suffix_format='03d',
                           variable_name='flux',
                           outfile=None,
                           write_ins=False)
    return results


def test_get_flux_obs(flux_obs):

    results = flux_obs
    expected_columns = ['datetime', 'per', 'obsprefix', 'obsnme',
                        'obs_flux', 'sim_flux', 'obsval', 'obgnme']
    assert np.all(results.columns == expected_columns)
    assert len(set(results.obsnme)) == len(results)
    assert not results.obs_flux.isna().any()
    assert not results.obsnme.str.isupper().any()

    # check sorting
    assert np.all(results.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)


def test_get_flux_obs_per_based_suffixes(flux_obs_per_based_suffixes):

    results = flux_obs_per_based_suffixes
    expected_columns = ['datetime', 'per', 'obsprefix', 'obsnme',
                        'obs_flux', 'sim_flux', 'obsval', 'obgnme']
    assert np.all(results.columns == expected_columns)
    assert len(set(results.obsnme)) == len(results)
    assert not results.obs_flux.isna().any()
    assert not results.obsnme.str.isupper().any()

    # check sorting
    assert np.all(results.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)
    
    # test observation name suffixes
    is_trans = [False if obsnme.split('_')[1] == 'ss' else True 
                for obsnme in results['obsnme']]
    parsed_periods = [int(obsnme.split('_')[1]) for obsnme in results.loc[is_trans, 'obsnme']]
    assert np.array_equal(parsed_periods, results.loc[is_trans, 'per'].values)
    

@pytest.mark.parametrize(('write_ins,obsnme_date_suffix,'
                          'obsnme_suffix_format'), 
                         ((True, False, '03d'),
                          (False, True, '%Y%m%d'),
                          ))
def test_get_temporal_flux_differences(flux_obs_input, write_ins, 
                                       obsnme_date_suffix, obsnme_suffix_format):
    flux_obs = get_flux_obs(flux_obs_input.perioddata,
                        model_output_file=flux_obs_input.model_output_file,
                        observed_values_file=flux_obs_input.observed_values_file,
                        observed_values_site_id_col='site_no',
                        observed_values_obsval_col='obsval',
                        observed_values_group_column='category',
                        obsnme_date_suffix=obsnme_date_suffix,
                        obsnme_suffix_format=obsnme_suffix_format,
                        variable_name='flux',
                        outfile=None,
                        write_ins=False)
    results = get_temporal_differences(flux_obs,
                                       flux_obs_input.perioddata,
                                       obs_values_col='obs_flux',
                                       sim_values_col='sim_flux',
                                       obsnme_date_suffix=obsnme_date_suffix,
                                       obsnme_suffix_format=obsnme_suffix_format,
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
                   'obsval', 'sim_obsval', 'obgnme', 'type']
                  )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    suffixes = np.ravel([obsnme.split('_')[1].split('d') for obsnme in results.obsnme])
    assert 'ss' not in suffixes
    
    # check observation names
    for i, r in results.iterrows():
        prefix, suffix = r.obsnme.split('_')
        suffix1, suffix2 = suffix.split('d')
        if obsnme_date_suffix:
            datetime1 = datetime.strptime(suffix1, obsnme_suffix_format)
            datetime2 = datetime.strptime(suffix2, obsnme_suffix_format)
        else:
            per1 = int(suffix1)
            per2 = int(suffix1)
    


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
                   'obs_diff', 'sim_diff', 'obgnme', 'obsnme',
                   'obsval', 'sim_obsval', 'type']
                  )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    
    
@pytest.mark.parametrize('forecast_sites,forecast_start_date,forecast_end_date', (
    ('all', '2010-01-01', '2015-01-01'),
    (['07281600', '07288500'], None, None)
))
def test_get_forecast_flux_obs(flux_obs_input, forecast_sites, 
                               forecast_start_date, forecast_end_date):
    
    # test input with a dataframe too
    observed_values = pd.read_csv(flux_obs_input.observed_values_file,
                                  dtype={'obsprefix': object,
                                         'site_no': object})
    observed_values['obgnme'] = 'flux'
    # for the sake of testing forecasts, pretend that the data stop in 2010
    observed_values = observed_values.loc[observed_values['datetime'] < '2010']
    
    obs = get_flux_obs(flux_obs_input.perioddata,
                            model_output_file=flux_obs_input.model_output_file,
                            observed_values_file=observed_values,
                            observed_values_site_id_col='site_no',
                            observed_values_obsval_col='obsval',
                            observed_values_group_column='obgnme',
                            variable_name='flux',
                            forecast_sites=forecast_sites,
                            forecast_start_date=forecast_start_date,
                            forecast_end_date=forecast_end_date,
                            write_ins=False, outfile=None)
        
    # pick a site with an incomplete record
    # the missing times should be filled with forecast obs
    loc = (obs.obsprefix == '07281600') & obs.obsval.isna()
    if forecast_start_date is not None and forecast_end_date is not None:
        assert len(obs.loc[loc]) == 9
    else:
        assert len(obs.loc[loc]) == 12
    if forecast_sites == 'all':
        assert set(obs['obgnme']) == {'flux-forecast', 'flux',
                                    # site that is in model results, 
                                    # but missing site information
                                    '07288580-forecast' 
                                    }
    else:
        assert set(obs['obgnme']) == {'flux-forecast', 'flux',
                                    }
    
    # outside of observed values
    # the resulting dataframe should be entirely filled
    assert not obs.drop(['obsval', 'obs_flux'], axis=1).isna().any().any()
    
    # if forecast_sites != 'all':
    # check that only forecasts for specified sites were generated
    if forecast_sites is not None and forecast_sites != 'all':
        assert set(obs.loc[obs['obsval'].isna(), 'obsprefix']) ==\
            set(forecast_sites)
    # check that forecasts were only generated within specified time window
    if forecast_start_date is not None:
        assert not np.any(obs.loc[obs['obsval'].isna(), 'datetime'] <\
            forecast_start_date)
    if forecast_end_date is not None:
        assert not np.any(obs.loc[obs['obsval'].isna(), 'datetime'] >\
            forecast_end_date)