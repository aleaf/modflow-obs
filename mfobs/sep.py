"""Functions related to hydrograph separation
"""
import numpy as np
import pandas as pd
    
    
def ih_method(Qseries, block_length=5, tp=0.9, interp_semilog=True, freq='D', limit=100):
    """Baseflow separation using the Institute of Hydrology method, as documented in
    Institute of Hydrology (1980) and Wahl and Wahl (1988).

    
    Parameters
    ----------
    Qseries : pandas Series
        Pandas time series (with datetime index) containing measured streamflow values.
    block_length : int
        N parameter in IH method. Streamflow is partitioned into N-day intervals;
        a minimum flow is recorded for each interval.
    tp : float
        f parameter in IH method. For each three N-day minima, if f * the central value
        is less than the adjacent two values, the central value is considered a 
        turning point. Baseflow is interpolated between the turning points.
    interp_semilog : boolean
        If False, linear interpolation is used to compute baseflow between  turning points
        (as documented in the IH method). If True, the base-10 logs of the turning points
        are interpolated, and the interpolated values are transformed back to 
        linear space (producing a curved hydrograph). Semi-logarithmic interpolation
        as documented in Wahl and Wahl (1988), is used in the Base-Flow Index (BFI)
        fortran program. This method reassigns zero values to -2 in log space (0.01)
        for the interpolation.
    freq : str or DateOffset, default ‘D’
        Any `pandas frequency alias <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_
        Regular time interval that forms the basis for base-flow separation. Input data are
        resampled to this frequency, and block lengths represent the number of time increments
        of the frequency. By default, days ('D'), which is what all previous BFI methods
        are based on. Note that this is therefore an experimental option; it is up to the user t
        o verify any results produced by other frequencies.
    limit : int
        Maximum number of timesteps allowed during linear interploation between baseflow 
        ordinances. Must be greater than zero.

    
    Returns
    -------
    Q : pandas DataFrame
        DataFrame containing the following columns:
        minima : N-day minima
        ordinate : selected turning points
        n : block number for each N-day minima
        QB : computed baseflow
        Q : discharge values
    
    Notes
    -----
    Whereas this program only selects turning points following the methodology above, 
    the BFI fortran program adds artificial turning points at the start and end of
    each calendar year. Therefore results for datasets consisting of multiple years
    will differ from those produced by the BFI program.
    
    References
    ----------
    Institute of Hydrology, 1980b, Low flow studies report no. 3--Research report: 
    Wallingford, Oxon, United Kingdom, Institute of Hydrology Report no. 3, p. 12-19
    
    Wahl, K.L and Wahl, T.L., 1988. Effects of regional ground-water level declines
    on streamflow in the Oklahoma Panhandle. In Proceedings of the Symposium on 
    Water-Use Data for Water Resources Management, American Water Resources Association. 
    
    """
    if len(Qseries) < 2 * block_length:
        raise ValueError('Input Series must be at '
                         'least two block lengths\nblock_length: '
                         '{}\n{}'.format(block_length, Qseries))

    # convert flow values to numeric if they are objects
    # (pandas will cast column as objects if there are strings such as "ICE")
    # coerce any strings into np.nan values
    if Qseries.dtype.name == 'object':
        Qseries = pd.to_numeric(Qseries, errors='coerce')

    # convert the series to a dataframe; resample to daily values
    # missing days will be filled with nan values
    df = pd.DataFrame(Qseries).resample(freq).mean()
    df.columns = ['Q']

    # compute block numbers for grouping values on blocks
    nblocks = int(np.floor(len(df) / float(block_length)))

    # make list of ints, one per measurement, denoting the block
    # eg [1,1,1,1,1,2,2,2,2,2...] for block_length = 5
    n = []
    for i in range(nblocks):
        n += [i + 1] * block_length
    n += [np.nan] * (len(df) - len(n))  # pad any leftover values with nans
    df['n'] = n

    # compute the minimum for each block
    # create dataframe Q, which only has minimums for each block
    blocks = df[['Q', 'n']].reset_index(drop=True).dropna(axis=0).groupby('n')
    Q = blocks.min()
    Q = Q.rename(columns={'Q': 'block_Qmin'})
    Q['n'] = Q.index
    # get the index position of the minimum Q within each block
    idx_Qmins = blocks.idxmin()['Q'].values.astype(int)
    # get the date associated with each Q minimum
    Q['datetime'] = df.index[idx_Qmins]

    # compute baseflow ordinates
    Q['ordinate'] = [np.nan] * len(Q)
    Qlist = Q.block_Qmin.tolist()
    Q['Qi-1'] = [np.nan] + Qlist[:-2] + [np.nan]
    Q['Qi'] = [np.nan] + Qlist[1:-1] + [np.nan]
    Q['Qi+1'] = [np.nan] + Qlist[2:] + [np.nan]
    isordinate = tp * Q.Qi < Q[['Qi-1', 'Qi+1']].min(axis=1)
    Q.loc[isordinate, 'ordinate'] = Q.loc[isordinate, 'block_Qmin']

    # reset the index of Q to datetime
    Q.index = Q.datetime

    # expand Q dataframe back out to include row for each day
    Q = Q.dropna(subset=['datetime'], axis=0).resample(freq).mean()

    # reassign the original flow values back to Q
    Q['Q'] = df.Q.loc[Q.index]
    Q.loc[Q['Q'] == 0, 'ordinate'] = 0
    
    # interpolate between baseflow ordinates
    if interp_semilog:
        iszero = Q.ordinate.values == 0
        logQ = np.log10(Q.ordinate)
        # fill zero values for the purpose of semi-log interpolation
        # (all baseflow values coinciding with 
        #  total flow values of zero will be reset to zero)
        # use the 1% quantile for filling
        # otherwise semi-log interpolation might be unreasonable (too rapid)
        # for larger units like cubic feet per day
        fill_value = np.log10(df.Q.loc[df.Q > 0].quantile(0.01))
        logQ[iszero] = fill_value
        QB = np.power(10.0, logQ.interpolate(limit=limit).values)
    else:
        QB = Q.ordinate.interpolate(limit=limit).values
    Q['QB'] = QB

    # in places where 'Q' is zero, set 'QB' to zero as well
    # (for example, zero-flow values past the interpolation limit)
    # otherwise, an inconsistent number of flow values can be returned for a site, 
    # depending on how many zero values are simulated,
    # which for example would cause PEST to crash during parameter estimation
    Q.loc[Q['Q'] == 0, 'QB'] = 0

    # ensure that no baseflow values are > Q measured
    QBgreaterthanQ = Q.QB.values > Q.Q.values
    Q.loc[QBgreaterthanQ, 'QB'] = Q.loc[QBgreaterthanQ, 'Q']
    return Q

