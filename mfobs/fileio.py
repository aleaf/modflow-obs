"""
Functions for reading and writing files
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd


def load_array(files):
    """Create a 3D numpy array from a list
    of ascii files containing 2D arrays.

    Parameters
    ----------
    files : sequence
        List of text files. Array layers will be
        ordered the same as the files.

    Returns
    -------
    array3d : 3D numpy array
    """
    if isinstance(files, str) or isinstance(files, Path):
        return np.loadtxt(files)
    arrays = []
    for f in files:
        arrays.append(np.loadtxt(f))
    array3d = np.stack(arrays)
    return array3d


def read_csv(csvfile, col_limit=1e4, **kwargs):
    """Read tabular data with pandas.read_csv,
    unless the data are super wide (col_limit or greater columns),
    in which case read the data using pure python. The pure
    python approach below is apparently much faster than
    pandas.read_csv for very wide files.

    Parameters
    ----------
    csvfile : str or pathlike
    col_limit : int
        Column threshold at which to use pure python 
        instead of pandas.read_csv, by default 10e4
    **kwargs : keyword arguments to pandas.read_csv or 
        pandas.DataFrame (in the case of a wide file)
    """
    t0 = time.time()
    # get the header length
    delim = kwargs.get('delimiter', ',')
    if kwargs.get('delim_whitespace', False):
        delim = ' '
    with open(csvfile) as src:
        header = next(iter(src)).split(delim)
    if len(header) > col_limit:
        lines = []
        with open(csvfile) as src:
            header = next(iter(src)).strip().split(',')
            for line in src:
                lines.append(line.strip().split(','))
                
        # handle duplicate columns in the same way that pandas does
        col_counts = {}
        new_header = []
        for col in header:
            if col not in col_counts:
                col_counts[col] = 1
                append_column_name = col
            else:
                append_column_name = f"{col}.{col_counts[col]}"
                col_counts[col] += 1
            new_header.append(append_column_name)
            
        df = pd.DataFrame(lines, columns=new_header, **kwargs)
    else:
        df = pd.read_csv(csvfile, **kwargs)
    print("took {:.2f}s\n".format(time.time() - t0))
    return df
    
    
def write_insfile(results_dataframe, outfile, obsnme_column='obsnme',
                  simulated_obsval_column='modelled', index=True):
    """Write instruction file for PEST. Assumes that
    observations names are in an obsnme_column and
    that the observation values an obsval_column. The values in obsval_column
    will be replaced in the instruction file with the names in obsnme_column.

    Parameters
    ----------
    results_dataframe : pandas dataframe
        Processed model output, in same structure/format
        as the processed output file.
    outfile : filepath
        Name of instruction file.
    obsnme_column : str
        Column in results_dataframe with observation names
    simulated_obsval_column : str
        Column in results_dataframe with the simulated observation equivalents
    index : bool
        Whether or not the index should be included; needs to be the same as the
        actual results file.
    """
    ins = results_dataframe.copy()
    # if the index is included, move it to the columns
    if index:
        ins.reset_index(inplace=True)
    # fill the index with the 'l1' (line advance) flag for PEST ins file reader
    ins.index = ['l1'] * len(ins)
    cols = ins.columns.tolist()

    # replace the observation values with the obsnames
    ins[simulated_obsval_column] = ['!{}!'.format(s) for s in results_dataframe[obsnme_column]]

    # fill the remaining columns with whitespace flags
    for c in cols:
        if c != simulated_obsval_column:
            ins[c] = 'w'

    # write the output
    with open(outfile, 'w', newline="") as dest:
        dest.write('pif @\n@{}@\n'.format(obsnme_column))
        ins.to_csv(dest, sep=' ', index=True, header=False)
        print(f'wrote {len(ins):,} observation instructions to {outfile}')