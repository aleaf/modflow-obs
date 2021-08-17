from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from mfobs.modflow import (
    get_perioddata, 
    get_mf6_single_variable_obs, 
    get_mf_gage_package_obs)
from mfobs.prms import get_prms_statvar_obs
from mfobs.obs import (get_base_obs, 
                       get_temporal_differences, 
                       get_spatial_differences,
                       get_annual_means,
                       get_monthly_means,
                       get_mean_monthly,
                       get_log10_observations,
                       get_baseflow_observations
)


@pytest.fixture
def flux_obs_input(shellmound_data_path):
    class FluxObsInput:
        perioddata = pd.read_csv(shellmound_data_path / 'tables/stress_period_data.csv')
        model_output_file = shellmound_data_path / 'shellmound.sfr.obs.output.csv'
        observed_values_file = shellmound_data_path / 'tables/processed_flow_obs.csv'
        tflux_diff_obs_outfile = shellmound_data_path / 'tables/processed_flow_obs_tdiffs.csv'
        sflux_diff_obs_outfile = shellmound_data_path / 'tables/processed_flow_obs_sdiffs.csv'
    return FluxObsInput


# TODO: refactor existing SW obs tests to work with get_base_obs
@pytest.fixture
def flux_obs(flux_obs_input):
    sfr_results = get_mf6_single_variable_obs(flux_obs_input.perioddata, 
                                              model_output_file=flux_obs_input.model_output_file,
                                              variable='flux'
                                              )

    results = get_base_obs(flux_obs_input.perioddata,
                           sfr_results,
                           observed_values_file=flux_obs_input.observed_values_file,
                           observed_values_site_id_col='site_no',
                           observed_values_obsval_col='obsval',
                           observed_values_group_column='category',
                           variable_name='flux',
                           obsnme_date_suffix=True,
                           obsnme_suffix_format='%Y%m',
                           label_period_as_steady_state=0,
                           outfile=None,
                           write_ins=False)
    return results


@pytest.fixture
def flux_obs_per_based_suffixes(flux_obs_input):
    sfr_results = get_mf6_single_variable_obs(flux_obs_input.perioddata, 
                                              model_output_file=flux_obs_input.model_output_file,
                                              variable='flux'
                                              )
    
    results = get_base_obs(flux_obs_input.perioddata,
                           sfr_results,
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


@pytest.fixture
def prms_statvar_obs(test_data_path):
    statvar_file = test_data_path / 'prms/brsql500.statvar'
    perioddata = get_perioddata(test_data_path / 'mf2005/br_trans.dis', start_datetime='2000-01-01',
                                end_datetime='2001-02-01', include_timesteps=True)
    statvar_sitenames = {66603: 'US1WIAS0003', 105512: 'USC00473332', 
                         54441: 'US1WIAS0006', 
                         160058: 'USC00475286', 158332: 'USC00478750', 
                         250762: 'USC00471604', 66561: 'USC00470349', 
                         13742: 'US1WIBY0004', 108149: 'US1WIBY0005'}
    statvar_results = get_prms_statvar_obs(perioddata, statvar_file, 
                                           statvar_sitenames=statvar_sitenames,
                                           obsnme_suffix_format='%Y%m%d')
    return statvar_results, perioddata


@pytest.fixture
def gage_package_obs(test_data_path):
    gage_file = test_data_path / 'mf2005/04027500.ggo'
    perioddata = get_perioddata(test_data_path / 'mf2005/br_trans.dis', start_datetime='2000-01-01',
                                end_datetime='2001-02-01', include_timesteps=True)
    gage_results = get_mf_gage_package_obs(perioddata, gage_file, variable='flow',
                                           abs=True)
    return gage_results, perioddata
    
    
def test_get_base_obs_statvar(prms_statvar_obs, test_data_path):
    
    statvar_results, perioddata = prms_statvar_obs
    observed_values_file = test_data_path / 'prms/daily_snowdepth.csv'
    observed_values_metadata = test_data_path / 'prms/snowpack_locations.csv'
    
    results = get_base_obs(perioddata,
                           statvar_results,
                           observed_values_file=observed_values_file,
                           #observed_values_metadata_file=observed_values_metadata,
                           variable_name='pk_depth',
                           observed_values_site_id_col='STATION',
                           observed_values_datetime_col='DATE',
                           obsnme_date_suffix=True,
                           obsnme_suffix_format='%Y%m%d',
                           observed_values_obsval_col='SNWD',
                           observed_values_group_column='obgnme',
                           observed_values_unc_column='uncertainty',
                           aggregrate_observed_values_method='mean',
                           drop_groups=None,
                           label_period_as_steady_state=None, steady_state_period_start=None,
                           steady_state_period_end=None,
                           outfile=None,
                           write_ins=False)


def test_get_flux_obs(flux_obs):

    results = flux_obs
    expected_columns = ['datetime', 'per', 'site_no', 'obsprefix', 'obsnme',
                        'obs_flux', 'sim_obsval', 'obsval', 'obgnme']
    assert np.all(results.columns == expected_columns)
    assert len(set(results.obsnme)) == len(results)
    assert not results.obs_flux.isna().any()
    assert not results.obsnme.str.isupper().any()

    # check sorting
    assert np.all(results.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)


def test_get_flux_obs_per_based_suffixes(flux_obs_per_based_suffixes):

    results = flux_obs_per_based_suffixes
    expected_columns = ['datetime', 'per', 'site_no', 'obsprefix', 'obsnme',
                        'obs_flux', 'sim_obsval', 'obsval', 'obgnme']
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
                          'obsnme_suffix_format, variable_name'), 
                         ((True, False, '03d', None),
                          (False, True, '%Y%m%d', 'flux'),
                          ))
def test_get_temporal_flux_differences(flux_obs_input, write_ins, 
                                       obsnme_date_suffix, obsnme_suffix_format, 
                                       variable_name):
    sfr_results = get_mf6_single_variable_obs(flux_obs_input.perioddata, 
                                              model_output_file=flux_obs_input.model_output_file,
                                              variable=variable_name
                                              )
    flux_obs = get_base_obs(flux_obs_input.perioddata,
                            sfr_results,
                            observed_values_file=flux_obs_input.observed_values_file,
                            observed_values_site_id_col='site_no',
                            observed_values_obsval_col='obsval',
                            observed_values_group_column='category',
                            variable_name=variable_name,
                            obsnme_date_suffix=obsnme_date_suffix,
                            obsnme_suffix_format=obsnme_suffix_format,
                            label_period_as_steady_state=0,
                            outfile=None,
                            write_ins=write_ins)
    results = get_temporal_differences(flux_obs,
                                       flux_obs_input.perioddata,
                                       obs_values_col='obsval',
                                       sim_values_col='sim_obsval',
                                       obsnme_date_suffix=obsnme_date_suffix,
                                       obsnme_suffix_format=obsnme_suffix_format,
                                       variable='flux',
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
    
    
@pytest.mark.parametrize('flux_difference_sites, write_ins, variable',
                         (({'07288280':  # sunflower r. at merigold
                           '07288500'   # sunflower r. at sunflower
                             }, True, 'flux'),
                          ({'07288500': '07288280' }, False, 'stage')))
def test_get_spatial_flux_difference_obs(flux_obs, flux_obs_input, flux_difference_sites, variable, write_ins):

    results = get_spatial_differences(flux_obs, flux_obs_input.perioddata,
                                      flux_difference_sites,
                                      obs_values_col='obsval',
                                      sim_values_col='sim_obsval',
                                      variable=variable,
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
                   'obsnme1', f'obs_{variable}1', f'sim_{variable}1',
                   'obsnme2', f'obs_{variable}2', f'sim_{variable}2',
                   'obs_diff', 'sim_diff', 'obgnme', 'obsnme',
                   'obsval', 'sim_obsval', 'type']
                  )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    
    
def test_get_annual_means(flux_obs):
    
    results = get_annual_means(flux_obs)
    
    # check the actual values
    grouped = flux_obs.groupby(flux_obs.datetime.dt.year).mean()
    expected = grouped[['obsval', 'sim_obsval']]
    assert np.allclose(results[['obsval', 'sim_obsval']].values, expected.values)
    
    assert np.all(results.columns ==
                ['datetime', 'year', 'site_no', 'obsprefix', 'obsnme', 'obs_flux',
                 'sim_obsval', 'obsval', 'obgnme']
                )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
    j=2
    
    
def test_mean_monthly(flux_obs):
    results = get_mean_monthly(flux_obs)
    
    # check the actual values
    grouped = flux_obs.groupby([flux_obs.datetime.dt.month]).mean()
    expected = grouped[['obsval', 'sim_obsval']]
    assert np.allclose(results[['obsval', 'sim_obsval']].values, expected.values)
    
    assert np.all(results.columns ==
                ['datetime', 'month', 'site_no', 'obsprefix', 'obsnme', 
                 'obs_flux', 'sim_obsval', 'obsval', 'obgnme']
                )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()


def test_get_monthly_means(flux_obs):
    results = get_monthly_means(flux_obs)
    
    # check the actual values
    grouped = flux_obs.groupby([flux_obs.datetime.dt.year,
                                flux_obs.datetime.dt.month]).mean()
    expected = grouped[['obsval', 'sim_obsval']]
    assert np.allclose(results[['obsval', 'sim_obsval']].values, expected.values)
    
    assert np.all(results.columns ==
                ['datetime', 'site_no', 'obsprefix', 'obsnme', 'obs_flux',
                 'sim_obsval', 'obsval', 'obgnme']
                )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()

def test_log_obs(flux_obs):
    results = get_log10_observations(flux_obs, fill_zeros_with=0)
    
    # check the actual values
    expected = np.log10(flux_obs[['obsval', 'sim_obsval']])
    expected.loc[flux_obs['obsval'] == 0, 'obsval'] = 0
    expected.loc[flux_obs['sim_obsval'] == 0, 'sim_obsval'] = 0
    assert np.allclose(results[['obsval', 'sim_obsval']].values, expected.values)
    
    assert np.all(results.columns ==
                ['datetime', 'site_no', 'obsprefix', 'obsnme', 'obs_flux',
                 'sim_obsval', 'obsval', 'obgnme']
                )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()


def test_get_baseflow_observations(gage_package_obs, test_data_path):
    """Test getting base obs from the gage package, and then baseflow 
    derivative obs."""
    
    gage_results, perioddata = gage_package_obs
    observed_values_file = test_data_path / 'mf2005/04027500.csv'
    
    base_obs = get_base_obs(perioddata,
                           gage_results,
                           observed_values_file=observed_values_file,
                           #observed_values_metadata_file=observed_values_metadata,
                           variable_name='flow',
                           observed_values_site_id_col='site_no',
                           observed_values_datetime_col='datetime',
                           obsnme_date_suffix=True,
                           obsnme_suffix_format='%Y%m%d',
                           observed_values_obsval_col='q_cfd',
                           #observed_values_group_column='obgnme',
                           #observed_values_unc_column='uncertainty',
                           #aggregrate_observed_values_method='mean',
                           #drop_groups=None,
                           #label_period_as_steady_state=0, steady_state_period_start=None,
                           #steady_state_period_end=None,
                           outfile=None,
                           write_ins=False)
    results = get_baseflow_observations(base_obs)
    
    assert np.all(results.columns ==
                ['datetime', 'site_no', 'obsprefix', 'obsnme', 'obs_flow',
                 'sim_obsval', 'obsval', 'obgnme']
                )
    assert len(set(results.obsnme)) == len(results)
    assert not results.obsval.isna().any()
    assert not results.sim_obsval.isna().any()
    assert results.obsnme.str.islower().all()
