"""
Tests for the swflows.py module

"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from mfobs.swflows import get_flux_obs


@pytest.fixture
def flux_obs_input(shellmound_data_path):
    class FluxObsInput:
        perioddata = pd.read_csv(shellmound_data_path / 'tables/stress_period_data.csv')
        model_output_file = shellmound_data_path / 'shellmound.sfr.obs.output.csv'
        observed_values_file = shellmound_data_path / 'tables/processed_flow_obs.csv'
    return FluxObsInput


def test_get_flux_obs(flux_obs_input):

    results = get_flux_obs(flux_obs_input.perioddata,
                           model_output_file=flux_obs_input.model_output_file,
                           observed_values_file=flux_obs_input.observed_values_file,
                           observed_values_site_id_col='site_no',
                           observed_values_obsval_col='obsval',
                           variable_name='flux',
                           outfile=None,
                           write_ins=False)

    expected_columns = ['datetime', 'per', 'obsprefix', 'obsnme',
                        'obs_flux', 'sim_flux']
    assert np.all(results.columns == expected_columns)
    assert len(set(results.obsnme)) == len(results)
    assert not results.obs_flux.isna().any()
    assert not results.obsnme.str.isupper().any()

    # check sorting
    assert np.all(results.reset_index(drop=True).groupby('obsprefix').per.diff().dropna() > 0)
