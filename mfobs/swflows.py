"""
Functions for working with streamflows
"""
import pandas as pd
from mfobs.fileio import write_insfile
from mfobs.modflow import get_mf6_single_variable_obs


def get_flux_obs(perioddata,
                 model_output_file='meras3_1L.sfr.obs.output.csv',
                 observed_values_file='../tables/flux_obs.csv',
                 observed_values_column='measured',
                 variable_name='flux_m3',
                 outfile=None,
                 write_ins=False):
    """[summary]

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str, optional
        [description], by default 'meras3_1L.sfr.obs.output.csv'
    observed_values_file : str, optional
        [description], by default '../tables/flow_obs_by_stress_period.csv'
    observed_values_column : str, optional
        Column in obs_values_file with measured flux values
    variable_name : str, optional
        [description], by default 'measured'
    outfile : str, optional
        [description], by default 'processed_flux_obs.dat'
    write_ins : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    results = get_mf6_single_variable_obs(perioddata, model_output_file=model_output_file,
                          variable_name=variable_name)
    observed = pd.read_csv(observed_values_file)
    observed.index = observed['obsnme']

    sim_values_column = 'sim_' + variable_name
    obs_values_column = 'obs_' + variable_name
    results[obs_values_column] = observed[observed_values_column]
    for column in ['uncertainty', 'group']:
         results[column] = observed[column]

    # nans are where sites don't have observation values for that period
    results.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # reorder the columns
    results = results[['per', 'obsprefix', 'obsnme', obs_values_column, sim_values_column, 'group', 'uncertainty']].copy()
    if outfile is not None:
        results.to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(results, outfile + '.ins', obsnme_column='obsnme',
                  simulated_obsval_column=sim_values_column, index=False)
    return results

