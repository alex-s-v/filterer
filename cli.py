from configparser import ConfigParser
from pathlib import Path
from copy import deepcopy

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from filterer import (
    process_data,
    compare_data
)


def main(config):
    """Main algorithm function

    Parameters
    ----------
    config : ConfigParser
        Configuration data for the algorithm
    """
    cfg = parse_config(config)
    size = cfg["General Parameters"]["number_of_interpolation_points"]
    deg = cfg["General Parameters"]["trend_degree"]
    names = [Path(cfg["Data Files"]["stand"])]
    frs = []
    c = cfg["Stand Cleaning Parameters"]
    i = 1
    while True:
        frs.append(process_data_cfg(names[-1], c, deg, size))
        try: c = cfg[f"Filter {i} Cleaning Parameters"]
        except: break
        names.append(Path(cfg["Data Files"][f"filter_{i}"]))
        i += 1
    xs, ys = compare_data(frs[0], frs[1:], size)
    path = Path(cfg["Output Parameters"]["out_path"])
    path.mkdir(parents=True, exist_ok=True)
    save_figures = cfg["Output Parameters"]["save_figures"] == "Y"
    preview = cfg["Output Parameters"]["preview"] == "Y"
    act_names = [x.name[:-4] for x in names]
    if save_figures or preview:
        save_plots(
            xs, ys, frs, act_names, path,
            save_figures=save_figures,
            preview=preview,
            dpi=cfg["Output Parameters"]["image_dpi"],
            format=cfg["Output Parameters"]["image_format"],
            size=cfg["Output Parameters"]["image_size"],
            **cfg["Figure Parameters"]
        )
    if cfg["Output Parameters"]["save_tables"] == "Y":
        save_tables(
            frs, act_names, path,
            cdel=cfg["Output Parameters"]["col_delimiter"],
            ddel=cfg["Output Parameters"]["dec_delimiter"]
        )
    if cfg["Output Parameters"]["save_trends"] == "Y":
        save_trends(frs, act_names, path)


def save_trends(datas, names, path):
    """Saves calculated trend equations in text format

    Parameters
    ----------
    datas : list of FtlrResult
        Filtering results for the stend  and the filters
    names : list of str
        List of figure names (in the same
        order as filters in `datas` parameter)
    path : pathlib.Path
        Path to the folder for saving text files
    """
    ml = max(map(len, names))
    with (path / "trends.txt").open(mode="w") as f:
        for d, n in zip(datas, names):
            eq = ps2eq(d.trend_coefs)
            f.write(f"{n}{' '*(ml-len(n))} : y = {eq}\n")
    return None


def save_tables(datas, names, path, cdel, ddel):
    """Saves calculated data in csv table format

    Parameters
    ----------
    datas : list of FtlrResult
        Filtering results for the stend  and the filters
    names : list of str
        List of figure names (in the same
        order as filters in `datas` parameter)
    path : pathlib.Path
        Path to the folder for saving tables
    cdel : str
        String of length 1. Field delimiter for the output file
    ddel : str
        String of length 1. Floating point delimiter
    """
    p = path / "tables"
    p.mkdir(parents=True, exist_ok=True)
    for d, n in zip(datas, names):
        df = pd.DataFrame({"Rate [l/min]": d.xf, "Pd [kPa]": d.yf})
        df.to_csv(p / f"{n}.csv", sep=cdel, decimal=ddel, index=False)
    return None


def save_plots(xs, ys, datas, names, path, **kwargs):
    """Create plots to save or display or both

    Parameters
    ----------
    xs : numpy.ndarray
        X coordinates of the data points
    ys : list of numpy.ndarray
        List of arrays of Y coordinates for
        the each filter (after the comparison
        to stand data)
    datas : list of FtlrResult
        Filtering results for the stend
        (always first) and the filters
    names : list of str
        List of figure names (in the same
        order as filters in `datas` parameter)
    path : pathlib.Path
        Path to the folder for saving figures
    """
    figs = []
    figsize = (kwargs["size"], kwargs["size"])
    for i, d in enumerate(datas):
        fig = plt.figure(num=names[i], figsize=figsize)
        plt.scatter(d.xf, d.yf, alpha=0.1)
        plt.plot(d.xt, d.yt, c="k")
        ax = plt.gca()
        ax.set_aspect(1 / ax.get_data_ratio())
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel(kwargs["xlabel"])
        plt.ylabel(kwargs["ylabel"])
        plt.tight_layout()
        figs.append(fig)
    fig = plt.figure(num="comparison", figsize=figsize)
    figs.append(fig)
    for i, y in enumerate(ys, start=1):
        plt.plot(xs, y, label=kwargs[f"filter_{i}_label"])
    ax = plt.gca()
    ax.set_aspect(1 / ax.get_data_ratio())
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel(kwargs["xlabel"])
    plt.ylabel(kwargs["ylabel"])
    plt.legend()
    plt.tight_layout()
    # new
    fig = plt.figure(num="comparison2", figsize=figsize)
    figs.append(fig)
    for i, d in enumerate(datas):
        plt.plot(d.xt, d.yt, label=kwargs[f"filter_{i}_label"])
    ax = plt.gca()
    ax.set_aspect(1 / ax.get_data_ratio())
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel(kwargs["xlabel"])
    plt.ylabel(kwargs["ylabel"])
    plt.legend()
    plt.tight_layout()
    # new
    if kwargs["preview"]:
        plt.show()
    if kwargs["save_figures"]:
        p = (path / "plots")
        p.mkdir(parents=True, exist_ok=True)
        for fig in figs:
            fig.savefig(
                p / f"{fig._label}.{kwargs['format']}",
                dpi=kwargs["dpi"],
                format=kwargs["format"]
            )
    return None


def ps2eq(ps):
    """Convert polynomial coefficients to text

    Parameters
    ----------
    ps : list of float
        List of polynomial coefficients

    Returns
    -------
    str
        String representation of the polynomial
    """
    n = len(ps)
    f = "".join(f"{p:+} * x^{n-i}" for i, p in enumerate(ps, start=1))
    if "+" == f[0]: f = f[1:]
    f = f.replace("-", " - ")
    f = f.replace("+", " + ")
    # -6 to cut the ` * x^0` at the end
    return f[:-6]


def process_data_cfg(filename, cfg, deg, size):
    """Process data using specified configuration

    Parameters
    ----------
    filename : str or  Path
        Path to a file with the data
    cfg : dict
        Configuration dictionary
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
    res = process_data(
        filename,
        {
            "size": cfg["number_of_bins"],
            "fpargs": {
                "height": cfg["height"], "distance": cfg["distance"]
            }
        },
        {"alpha_low": cfg["alpha_low"], "alpha_high": cfg["alpha_high"]},
        deg, size
    )
    return res


def parse_config(cfg):
    """Parse configuration dictionary

    Parameters
    ----------
    cfg : dict
        Raw configuration dictionary

    Returns
    -------
    dict
        Parsed configuration file
    """
    cfg_ = dict()
    for k1 in cfg:
        cfg_[k1] = {}
        for k2 in cfg[k1]:
            try: cfg_[k1][k2] = int(cfg[k1][k2])
            except:
                try: cfg_[k1][k2] = float(cfg[k1][k2])
                except: cfg_[k1][k2] = cfg[k1][k2]
    return cfg_


if __name__ == "__main__":
    mpl.style.use("seaborn-whitegrid")
    config = ConfigParser()
    config.read("settings.ini")
    main(config)
