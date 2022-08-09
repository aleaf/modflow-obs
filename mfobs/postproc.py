"""General functions for postprocessing and visualizing observation 
results from PEST.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pyemu


title_prefixes = {
    # ending part of obgnme: prefix to prepend to title
    'heads': 'Head',
    'head': 'Head', 
    'tdiff': 'Change in head', 
    'sdiff': 'Head difference', 
    'disp': 'Total change in head', 
    'flux': 'Streamflow', 
    'flux_tdiff': 'Change in streamflow',
    'flux_sdiff': 'Change in streamflow', 
    'priority_wells': 'Head',
                }
        
def get_obgnme_item(obgnme, obgnme_endings_dict):
    """Given a dictonary of obgnme endings,
    return the value for the key with a unique match.
    """
    matches = {k: v for k, v in obgnme_endings_dict.items() 
               if obgnme.endswith(k)}
    most_matching = 0
    item = ''
    for k, v in matches.items():
        n_matching = len(set(k).intersection(obgnme))
        if n_matching > most_matching:
            most_matching = n_matching
            item = v
    return item


def read_res_file(res_file):
    """Read a PEST residuals file into a pandas dataframe.
    """
    with open(res_file) as src:
        for skiprows, line in enumerate(src):
            if ('measured' in line.lower()) and ('modelled' in line.lower()):
                break
    # read the res file
    res = pd.read_csv(res_file, delim_whitespace=True, skiprows=skiprows)
    res.columns = [c.lower() for c in res.columns]
    res.index = res['name']
    return res


def get_obs_info(obsnmes=None, steady_state_time=None):
    """Parse information from observation names.

    Parameters
    ----------
    obsnmes : sequence
        Sequence of observation names.
    steady_state_time : str, optional
        Date to apply to steady-state observations, 
        by default None

    Returns
    -------
    df : DataFrame
        DataFrame with obsprefix (site number), 
        obstype (e.g. measurement, time or space-difference) and datetime 
        for each observation in obsnmes.
    """
    # parse dates from observation suffixes
    if obsnmes is not None:
        steady_state_time = '1998-04-01'
        obs_prefix = []
        obstypes = []
        datetime = []
        for c in obsnmes:
            if 'trend' in c:
                datetime.append(None)  
                continue      
            elif '-d-' in c:
                sites, date = c.split('_')
                site, site2 = sites.split('-d-')
                obstypes.append('sdiff')
            elif 'd' in c:
                site, dates = c.split('_')
                date, date2 = dates.split('d')
                obstypes.append('tdiff')
            else:
                site, date = c.split('_')
                obstypes.append('meas')
            obs_prefix.append(site)
            if date == 'ss':
                datetime.append(steady_state_time)
            else:
                try:
                    timestamp = pd.Timestamp(f'{date[:4]}-{date[4:]}')
                except:
                    try:
                        timestamp = int(date)
                    except:
                        j=2
                datetime.append(timestamp)
        obs_prefix = np.array(obs_prefix)
        obstypes = np.array(obstypes)
        datetime = np.array(datetime)
        df = pd.DataFrame({'obsnme': obsnmes,
                        'obsprefix': np.array(obs_prefix),
                        'obstype': np.array(obstypes),
                        'datetime': np.array(datetime)
                        })

    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def get_obs_info_from_files(obsfiles):
    """Parse information from observation names.

    Parameters
    ----------
    obsfiles : sequence
        Sequence of processed observations files
        (i.e. as produced by the functions in the mfobs.obs module)

    Returns
    -------
    df : DataFrame
        DataFrame with obsprefix (site number), 
        obstype (e.g. measurement, time or space-difference) and datetime 
        for each observation in obsnmes.
    """
    # get dates from observation info file
    # (i.e. for stress period-based suffixes)
    req_columns = ['obsnme', 'obsprefix']
    dfs = []
    for f in obsfiles:
        df = pd.read_csv(f, delim_whitespace=True, dtype={'obsprefix': object})
        for c in req_columns:
            if c not in df.columns:
                raise ValueError(f"Observation file {f} is missing an {c} column.")
        dfs.append(df)
    df = pd.concat(dfs)

    obstypes = []
    for c in df['obsnme']:        
        if '-d-' in c:
            obstypes.append('sdiff')
        elif 'd' in c:
            obstypes.append('tdiff')
        else:
            obstypes.append('meas')
    df['obstype'] = obstypes
    df = df[['obsnme', 'obsprefix', 'obstype', 'datetime']]
    df.index = df['obsnme']
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def get_summary_statistics(res_data, include_zero_weighted=False,
                           head_group_identifier='heads',
                           outfile=None):
    
    # get the header length in the resfile
    if  isinstance(res_data, str) or isinstance(res_data, Path):
        # read the res file
        res = read_res_file(res_data)
    else:
        res = res_data.copy()
    res.columns = [c.lower() for c in res.columns]
    
    dfs = []
    if not include_zero_weighted:
        for criteria in [res['weight'] > 0., res['weight'] == 0]:
            if np.any(criteria):
                dfs.append(res.loc[criteria].copy())
    else:
        dfs = [res.copy()]
    
    processed_dfs = []
    for df in dfs:
        groups = df.groupby('group')
        summary_dict = {}
        for obgnme, results in groups:
            summary_dict[obgnme] = {
                'ME': results['residual'].mean(),
                'MAE': results['residual'].abs().mean(),
                'RMSE': np.sqrt(np.mean(results['residual']**2)),
                'max': results['residual'].max(),
                'min': results['residual'].min(),
                'meas_range': results['measured'].quantile(0.99) -\
                    results['measured'].quantile(0.01),
                'n': len(results),
                'avg_weight': results['weight'].mean(),
            }
        summary = pd.DataFrame(summary_dict).T
        
        # TODO: add a get_obstype fn here
        summary['obstype'] = ['flux' if 'flux' in obgnme else 'head' 
                        for obgnme in summary.index]
        summary['obstype'] = [obgnme.split('_')[-1] if obgnme.split('_')[-1] 
                              in {'disp', 'sdiff', 'tdiff', 'transmissivity', 'trend'} else obstype
                        for obgnme, obstype in zip(summary.index, summary['obstype'])]
        all_head_groups  = [g for g in res.group.unique() if head_group_identifier in g 
                    and not g.endswith('diff') and not g.endswith('disp')]
        if not any(all_head_groups):
            all_head_groups = res.group.unique()
        
        spinup_heads = [g for g in all_head_groups if g.endswith('spinup')]
        summary.loc[summary.index.isin(spinup_heads), 'obstype'] = 'head-spinup'
        if any(spinup_heads):
            all_heads = {
                'all_heads_cal': [g for g in all_head_groups if not g.endswith('spinup')],
                'all_heads_spinup': [g for g in all_head_groups if g.endswith('spinup')],
                'all_heads': all_head_groups
            }
        else:
            all_heads = {'all_heads': all_head_groups}
            
        for col, subset in all_heads.items():
            group = df.loc[df.group.isin(subset)].copy()
            if df.group.isin(subset).any():
                obstype = 'head-all'
                if 'spinup' in col:
                    obstype = 'head-spinup-all'
                elif 'cal' in col:
                    obstype = 'head-post-spinup-all'
                all_head_stats = pd.Series({
                    'ME': group['residual'].mean(),
                    'MAE': group['residual'].abs().mean(),
                    'RMSE': np.sqrt(np.mean(group['residual']**2)),
                    'max': group['residual'].max(),
                    'min': group['residual'].min(),
                    'meas_range': group['measured'].quantile(0.99) -\
                        group['measured'].quantile(0.01),
                    'n': len(group),
                    'obstype': obstype,
                    'avg_weight': group['weight'].mean(),
                        })
                all_head_stats.name = col
                summary = summary.append(all_head_stats)
            summary['rmse_frac_range'] = summary['RMSE'] / summary['meas_range']
        processed_dfs.append(summary)
    df = pd.concat(processed_dfs)
    df['weighting'] = ['zero-weighted' if avg_weight == 0 else 'weighted' 
                     for avg_weight in df['avg_weight']]
    df.sort_values(by=['weighting', 'obstype', 'RMSE'], ascending=[True, True, True], inplace=True)
    df['n'] = df['n'].astype(int)
    df['group'] = df.index
    df = df[['weighting', 'group', 'RMSE', 'ME', 'MAE', 'max', 'min', 'meas_range', 'rmse_frac_range', 'n', 'avg_weight', 'obstype']]
    if outfile is not None:
        outfile.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(outfile, float_format='%.2f', index=False)
    return df


def plot_one_to_one(obs_output, 
                    obs_values_col='measured', sim_values_col='modelled',
                    include_zero_weighted=False,
                    title_prefix=None, units=None,
                    outfile=None):
    # make the output folder if it doesn't exist
    if outfile is not None:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        
    obs_output.columns = [c.lower() for c in obs_output.columns]
  
    # set the output folder
    if outfile is None:                                     
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        if len(obs_output) == 0:
            print(f'skipping {outfile} (no data)')                            
            return
    
    if not 'weight' in obs_output.columns:
        include_zero_weighted = True
    if not include_zero_weighted:
        dfs = []
        weighted_obs = obs_output.loc[obs_output['weight'] > 0].copy()
        zero_weighted_obs = obs_output.loc[obs_output['weight'] == 0].copy()
        if len(weighted_obs) > 0:
            dfs.append(weighted_obs)
        if len(zero_weighted_obs) > 0:
            dfs.append(zero_weighted_obs)
    else:
        dfs = [obs_output.copy()]
    
    for df in dfs:
        stats = get_summary_statistics(df, include_zero_weighted=include_zero_weighted)
        
        zero_weighted = False
        if 'weight' in df.columns and df['weight'].max() == 0:
            zero_weighted = True
            stats = stats.loc[stats['weighting'] == 'zero-weighted']
        else:
            stats = stats.loc[stats['weighting'] != 'zero-weighted']

        if 'all_heads' not in stats.index:
            raise ValueError("get_summary_statistics should have included "
                             "an 'all_heads' row in the output")
        fig, ax = plt.subplots() #figsize=(11, 8.5))
        ax.scatter(df[obs_values_col], df[sim_values_col], s=5, c='k', alpha=0.5, lw=0)
        vmin = np.min([df[obs_values_col].min(), 
                    df[sim_values_col].min()])
        vmax = np.max([df[obs_values_col].max(), 
                    df[sim_values_col].max()])
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        plt.plot([vmin, vmax], [vmin, vmax], c='k', lw=0.5, zorder=-1)
        ax.set_xlabel('measured')
        ax.set_ylabel('modelled')
        text = ''
        for stat in 'ME', 'MAE', 'RMSE':        
            text += f"{stat}: {stats.loc['all_heads', stat]:.2f}"
            if stat == 'ME' and units is not None:
                text += f" {units}"
            text += "\n"
        text += f"RMSE/range: {stats.loc['all_heads', 'rmse_frac_range']:.2%}\n"
        text += f"n: {stats.loc['all_heads', 'n']:,d}\n"
        
        # add groups to text
        if 'group' in df.columns:
            text += '\ngroups:\n  ' + '\n  '.join(sorted(df.group.unique()))
        ax.text(0.75, 0.6, text, transform=ax.transAxes, va='top')
        
        title = ''
        if title_prefix is not None:
            title = f'{title_prefix} '
        if zero_weighted:
            title += ' (zero-weighted)'
        ax.set_title(title)
            
        plt.tight_layout()
        if outfile is not None:
            if zero_weighted:
                outfile = outfile.with_stem(f'{outfile.stem}_zero-weighted')
            plt.savefig(outfile)
            plt.close()
            print(f'wrote {outfile}')
        
        
def export_residuals_shapefile(obs_output, obs_info, how='mean', 
                               obs_values_col='measured', sim_values_col='modelled',
                               obs_output_obsprefix_col='obsprefix',
                               meta_obsprefix_col='obsprefix',
                               meta_x_col='x', meta_y_col='y', meta_crs=None,
                               outfile_name_base=None):

    # make the output folder if it doesn't exist
    if outfile_name_base is not None:
        Path(outfile_name_base).parent.mkdir(parents=True, exist_ok=True)
        
    if not isinstance(obs_output, pd.DataFrame):
        df = pd.read_csv(obs_output, delim_whitespace=True, 
                             dtype={'obsprefix': np.object})
    else:
        df = obs_output
    
    # set the output filename suffix
    suffix = f'_{how}'
    if how != 'mean':
        raise NotImplementedError("residuals by stress period.")
        try:
            int(how)
        except:
            raise ValueError("'how' argument must be either 'mean' or a stress period number.")
        suffix = '_per' + suffix
        
    # set the output folder
    if outfile_name_base is None:
        if not isinstance(obs_output, pd.DataFrame):
            outpath = Path(obs_output) / 'postproc/shps'
        else:
            outpath = Path('.')
            obs_output = 'residuals'
        outfile = outpath / (Path(obs_output).stem + suffix + '.shp')
    else:
        outfile = Path(outfile_name_base)
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(obs_info, pd.DataFrame):
        meta = pd.read_csv(obs_info, dtype={meta_obsprefix_col: np.object})
    else:
        meta = obs_info.copy()
    meta['geometry'] = [Point(x, y) for x, y in zip(meta[meta_x_col], meta[meta_y_col])]
    meta.index = meta[meta_obsprefix_col].str.lower()
    # use actual PEST groups instead of observation site info groups
    meta.drop('group', axis=1, inplace=True, errors='ignore')

    df_agg = df.groupby(by=obs_output_obsprefix_col).mean()
    df_agg['group']= df.groupby(by=obs_output_obsprefix_col).first()['group']
    
    # compute mean absolute error at each site
    df_agg['mae'] = df.groupby(by=obs_output_obsprefix_col)['residual'].apply(lambda x: x.abs().mean())
    
    # compute root mean squared error at each site
    df_agg['rmse'] = df.groupby(by=obs_output_obsprefix_col)['residual'].apply(lambda x: np.sqrt(np.mean(x**2)))

    # compute nash-sutcliffe model efficiency at each site
    def nse(x):
        nse = 1 - np.sum((x['measured'] - x['modelled'])**2)/\
            np.sum((x['measured'] - x['measured'].mean())**2)
        return nse
    
    # compute modified nash-sutcliffe model
    # that uses absolute values instead of squares
    # (to reduce sensitivity to outliers)
    # https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    def nse_abs(x):
        nse = 1 - np.sum((x['measured'] - x['modelled']).abs())/\
            np.sum((x['measured'] - x['measured'].mean()).abs())
        return nse
        
    df_agg['nse'] = df.groupby(by=obs_output_obsprefix_col).apply(nse)
    df_agg['nse_abs'] = df.groupby(by=obs_output_obsprefix_col).apply(nse_abs)
    # compute normalized nash-sutcliffe model efficiency
    # (0-1)
    # for easier spatial comparison between sites
    # https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    df_agg['nse_norm'] = 1/(2 - df_agg['nse'])
    
    cols = [obs_values_col, sim_values_col, 'group', 'mae', 'rmse', 'nse', 'nse_abs', 'nse_norm']
    joined_df = meta.join(df_agg[cols], how='inner')
    joined_df['residual'] = joined_df[obs_values_col] - joined_df[sim_values_col]
    joined_df['abs_res'] = joined_df['residual'].abs()
    # drop kruft columns
    joined_df = joined_df.loc[:, ~joined_df.isna().all(axis=0)].copy()
    
    joined_df = gpd.GeoDataFrame(joined_df, crs=meta_crs)
    
    # write shapefiles of positive and negative mean residuals
    neg_outfile = outfile.with_stem(outfile.stem + '-neg')
    joined_df.loc[joined_df['residual'] < 0].to_file(neg_outfile, index=False)
    print(f'wrote {neg_outfile}')
    
    pos_outfile = outfile.with_stem(outfile.stem + '-pos')
    joined_df.loc[joined_df['residual'] > 0].to_file(pos_outfile, index=False)
    print(f'wrote {pos_outfile}')