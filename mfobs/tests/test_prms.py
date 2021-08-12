import numpy as np
import pytest
from mfobs.modflow import get_perioddata
from mfobs.prms import read_statvar_file, get_prms_statvar_obs



def test_read_statvar(test_data_path):
    statvar_file = test_data_path / 'prms/brsql500.statvar'
    df = read_statvar_file(statvar_file)
    assert df.loc['2000-01-01', '1527-seg_outflow'] == 6.321514
    
@pytest.mark.parametrize('statvar_sitenames,expected_obsnme,expected_site_no,expected_value', (
     (None, '1527-seg_outflow_20000101', 1527, 6.321514),
     ({1527: 'SITE1'}, 'site1-seg_outflow_20000101', 'site1', 6.321514)
))
def test_get_prms_statvar_obs(test_data_path, statvar_sitenames, 
                              expected_obsnme, expected_site_no, expected_value):
    statvar_file = test_data_path / 'prms/brsql500.statvar'
    perioddata = get_perioddata(test_data_path / 'mf2005/br_trans.dis', 
                                start_datetime='2000-01-01'
                                )
    results = get_prms_statvar_obs(perioddata, statvar_file, 
                                   statvar_sitenames=statvar_sitenames
)
    assert np.all(results.columns ==
                  ['datetime', 'site_no', 'variable', 'obsprefix', 'obsnme',
                   'sim_obsval', 'per', 'time']
                  )
    assert results.loc[expected_obsnme, 'sim_obsval'] == expected_value
    assert results.loc[expected_obsnme, 'site_no'] == expected_site_no
