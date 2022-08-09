import numpy as np
import pandas as pd
import pytest
from mfobs.postproc import (
    export_residuals_shapefile,
    get_obs_info, 
    get_obs_info_from_files,
    get_summary_statistics,
    plot_one_to_one,
    read_res_file
)
from .test_heads import shellmound_grid_transform, head_obs_input, head_obs
from .test_obs import flux_obs_input, flux_obs

@pytest.mark.skip("incomplete")
def test_get_obs_info(test_data_path):
    obsnames = [
        'bgc-flow_20170920',
        'bgc-flow_001',
        '02469800-stage_041',
        'bgc-flow_ss',
        'usgs:331022090490701-d-usgs:331330090564801_102',
        'ar001:345152091502402_061d059',
        'ar008:344139091054202_051-140-trend',
        'hub701-trans'
    ]
    results = get_obs_info(obsnmes=obsnames)
    
    
def test_get_obs_info_from_files(test_output_folder, head_obs, flux_obs):
    flux_obs.to_csv(test_output_folder / 'processed_flux_obs.dat', sep=' ')
    files = [test_output_folder / 'shellmound/processed_head_obs.dat',
             test_output_folder / 'processed_flux_obs.dat']
    results = get_obs_info_from_files(files)
    for col in 'obsprefix', 'obsnme':
        assert np.all(results.iloc[:len(head_obs)][col].values ==\
            head_obs[col].values)
        assert np.all(results.iloc[len(head_obs):][col].values ==\
            flux_obs[col].values)


@pytest.mark.parametrize('include_zero_weighted', (True, False))
def test_get_summary_statistics(test_data_path, include_zero_weighted):
    resfile = test_data_path / 'pest/tcaap.1.base.rei'
    result = get_summary_statistics(resfile, 
                                    head_group_identifier='obgnme', 
                                    include_zero_weighted=include_zero_weighted)
    assert set(result.index) == {'obgnme', 'all_heads'}
    assert not result.isna().any().any()
    
    res = read_res_file(resfile)
    res['group'] = [f'{grp}-spinup' if i < 10 else grp for i, grp in enumerate(res['group'])]
    result = get_summary_statistics(res, 
                                    head_group_identifier='obgnme', 
                                    include_zero_weighted=include_zero_weighted)
    assert np.all(result.index == ['obgnme', 'all_heads', 'all_heads_cal', 'obgnme-spinup',
       'all_heads_spinup'])
    
    
def test_plot_one_to_one(test_data_path, test_output_folder):
    resfile = test_data_path / 'pest/tcaap.1.base.rei'
    res = read_res_file(resfile)
    outfile = test_output_folder / f'{resfile.stem}.1to1.pdf'
    plot_one_to_one(res, outfile=outfile)
    assert outfile.exists()
    
    
def test_export_residuals_shapefile(test_data_path, test_output_folder):
    resfile = test_data_path / 'pest/tcaap.1.base.rei'
    res = read_res_file(resfile)
    res['obsprefix'] = ['_'.join(s.split('_')[:-1]) for s in res['name']]
    outfile = test_output_folder / f'{resfile.stem}.residuals.shp'
    meta = pd.read_csv(test_data_path / 'tcaap_obsinfo.csv')
    export_residuals_shapefile(res, meta, 
                               obs_values_col='measured', sim_values_col='modelled',
                               obs_output_obsprefix_col='obsprefix',
                               meta_obsprefix_col='Name',
                               meta_x_col='X', meta_y_col='Y', meta_crs=26915,
                               outfile_name_base=outfile)
    # export_residuals_shapefile makes separate outfiles
    # one for negative residuals, one for positive
    # (so that these can easily be graduated by size 
    #  and colored as over or under in a GIS environment)
    neg_outfile = outfile.with_stem(outfile.stem + '-neg')
    pos_outfile = outfile.with_stem(outfile.stem + '-pos')
    assert neg_outfile.exists()
    assert pos_outfile.exists()
    
    
