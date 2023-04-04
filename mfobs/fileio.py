"""
Functions for reading and writing files
"""
from operator import index
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from mfobs.utils import get_input_arguments


def load_array(files, shape=None):
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
    if shape is not None:
        if len(files) != shape[0]:
            raise ValueError(
                'Number of files should match the first element in shape!')
    arrays = []
    for f in files:
        try:
            arr = np.loadtxt(f)
        except:
            if shape is not None:
                with open(f) as src:
                    arr = src.read().replace('\n', ' ')
                    arr = np.fromstring(arr, sep=' ')
                arr = np.reshape(arr, shape[-2:])
            else:
                raise ValueError('Text array file {} not readable by np.loadtxt; may be wrapped. '
                                 'Please supply output shape of array.')
        arrays.append(arr)
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
        instead of pandas.read_csv, by default 1e4
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
        index_col = None
        index = None
        if 'index_col' in kwargs:
            index_col = kwargs['index_col']
            index = pd.read_csv(csvfile, usecols=[index_col], index_col=0).index
            if not isinstance(index_col, int):
                index_col = new_header.index(index_col)
            del new_header[index_col]
            for l in lines:
                del l[index_col]
        kwargs = get_input_arguments(kwargs, pd.DataFrame)
        df = pd.DataFrame(lines, columns=new_header, index=index,
                          **kwargs)
    else:
        kwargs = get_input_arguments(kwargs, pd.read_csv)
        # del 'dtype' if it's in kwargs
        # pandas sniffer should apparently work for realization numbers to ints
        # and 'base' to str
        if kwargs.get('dtype') == float:
            del kwargs['dtype']
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
        

def get_insfile_observations(insfile):
    """Return observation names listed in a PEST 
    instruction file (in order)."""
    with open(insfile) as src:
        text = src.read()
        results = re.findall("(?<=\!).+?(?=\!)",text)
    return results