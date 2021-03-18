from collections import namedtuple

import numpy as np
from scipy import signal
import pandas as pd

FtlrResult = namedtuple("FtlrResult", [
    "xf", "yf",
    "xt", "yt",
    "trend_coefs"
])


def load_data(filename):
    """Load and format data from a specified file.

    Parameters
    ----------
    filename : str or Path
        Path to a file with the data

    Returns
    -------
    pandas.DataFrame
        Loaded and formated data in the `pandas.DataFrame` format
    """
    df = pd.read_csv(filename, sep=";", header=None)
    df = df.drop([0, 1, 2, 3, 4, 6, 8, 10], axis=1)
    df.columns = ["Pin [kPa]", "Pout [kPa]", "Rate [l/min]"]
    return df


def find_peaks(data, size=200, fpargs={"height": 100, "distance": 15}):
    """Find the largest data clusters.

    Parameters
    ----------
    data : array like
        X coordinates of the data points
    size : int, optional
        A number of data subdivisions. The larger the number,
        the smaller the data clusters, by default 200
    fpargs : dict, optional
        Keyword arguments to the `scipy.signal.find_peaks` algorithm,
        by default {"height": 100, "distance": 15}

    Returns
    -------
    peaks : numpy.ndarray
        X coordinates of the largest data clusters
    ebins : numpy.ndarray
        The edges of all data clusters
    vals : numpy.ndarray
        The number of elements in each data cluster
    """
    vals, ebins = np.histogram(data, size)
    # Add zeros to include peaks on the edgegs
    vals_ = [0, *vals, 0]
    # Minus one to account for added zeros
    peaks = signal.find_peaks(vals_, **fpargs)[0] - 1
    return peaks, ebins, vals


def filter_data(xs, ys, peaks, bins, alpha_low=0.05, alpha_high=0.05):
    """Cleans the data, namely:
    * Removes the data collected during the intermediate regimes
    * Constrain vertical elongation of the desired regions

    Parameters
    ----------
    xs : numpy.ndarray
        X coordinates of the data points
    ys : numpy.ndarray
        Y coordinates of the data points
    peaks : numpy.ndarray
        X coordinates of the largest data clusters
    bins : numpy.ndarray
        The edges of all data clusters
    alpha_low : float, optional
        The lower bound of the cluster as a fraction
        from the geometric center, by default 0.05
    alpha_high : float, optional
        The upper bound of the cluster as a fraction
        from the geometric center, by default 0.05

    Returns
    -------
    numpy.ndarray
        X coordinates of the cleand data points
    numpy.ndarray
        Y coordinates of the cleand data points
    """
    xs_ = []
    ys_ = []
    bins = np.append(bins, bins[-1]*1.1)
    for pi in peaks:
        ids = (bins[pi] <= xs) & (xs <= bins[pi+1])
        temp = ys[ids]
        rates = xs[ids]
        m = np.median(temp)
        ids = (m * (1 - alpha_low) <= temp) & (temp <= m * (1 + alpha_high))
        xs_.extend(rates[ids])
        ys_.extend(temp[ids])
    return np.array(xs_), np.array(ys_)


def process_data(filename, peak_kwgs, fltr_kwgs, deg, size):
    """Processes submitted datafile and returns clean data
    and fitted trend line

    Parameters
    ----------
    filename : str or Path
        Path to a file with the data
    peak_kwgs : dict
        Parameters to the peak finding algorithm
    fltr_kwgs : dict
        Parameters to the data filtering algorithm
    deg : int
        Degree of the fitting polynomial
    size : int
        Number of points in the calculated trend line
        coordinates

    Returns
    -------
    FtlrResult
        Result of the data processing:
        * Cleaned data
        * Trend line
        * Trend line equation
    """
    df = load_data(filename)
    xs = df["Rate [l/min]"].values
    ys = (df["Pin [kPa]"] - df["Pout [kPa]"]).values
    peaks, bins, _ = find_peaks(xs, **peak_kwgs)
    xs_fltr, ys_fltr = filter_data(xs, ys, peaks, bins, **fltr_kwgs)
    ps = np.polyfit(xs_fltr, ys_fltr, deg=deg)
    xs_trnd = np.linspace(xs_fltr.min(), xs_fltr.max(), size)
    ys_trnd = np.polyval(ps, xs_trnd)
    res = FtlrResult(
        xs_fltr, ys_fltr,
        xs_trnd, ys_trnd,
        ps
    )
    return res


def compare_data(stand, fltrs, size):
    """Compare stend data with filter data and
    return 

    Parameters
    ----------
    stand : FtlrResult
        Filtering result for the stand data
    fltrs : list of FtlrResult
        Filtering results for the filters
    size : int
        Number of points in the calculated trend line
        coordinates

    Returns
    -------
    numpy.ndarray
        X coordinates of the data points
    list of numpy.ndarray
        List of arrays of Y coordinates for
        the each filter
    """
    xmin = max(map(lambda x: x.xf.min(), [stand, *fltrs]))
    xmax = min(map(lambda x: x.xf.max(), [stand, *fltrs]))
    xs = np.linspace(xmin, xmax, size)
    yts = np.polyval(stand.trend_coefs, xs)
    dyfs = [np.polyval(f.trend_coefs, xs) - yts for f in fltrs]
    return xs, dyfs
