import numpy as np
from mfobs.modflow import get_perioddata
from mfobs.prms import read_statvar_file, get_prms_statvar_obs


def test_read_statvar(test_data_path):
    statvar_file = test_data_path / 'badriver/brsql500.statvar'
    df = read_statvar_file(statvar_file)
    
    assert df.loc['2000-01-01', 'seg_outflow-1527'] == 6.321514
    

def test_get_prms_statvar_obs(test_data_path):
    statvar_file = test_data_path / 'badriver/brsql500.statvar'
    perioddata = get_perioddata(test_data_path / 'badriver/br_trans.dis', start_datetime='2000-01-01')
    results = get_prms_statvar_obs(perioddata,
                         statvar_file)
    assert np.all(results.columns ==
                  ['datetime', 'obsprefix', 'obsnme',
                   'sim_obsval', 'per', 'time']
                  )
    assert results.loc['seg_outflow-1527_20000101', 'sim_obsval'] == 6.321514
