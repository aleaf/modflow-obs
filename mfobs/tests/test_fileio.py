"""
Tests for the fileio.py module
"""
import pandas as pd
from mfobs.fileio import read_csv


def test_read_csv(test_data_path):
    
    csvfile = test_data_path / 'meras3/meras3_1L.sfr.obs6.output.csv'

    df1 = read_csv(csvfile)
    df2 = read_csv(csvfile, col_limit=2, dtype='float64')
    # the frames will only be equal if the dtypes are right
    # and duplicate columns are handled the same as pandas
    pd.testing.assert_frame_equal(df1, df2)
