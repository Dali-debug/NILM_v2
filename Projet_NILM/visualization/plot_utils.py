"""
plot_utils.py
-------------
Shared matplotlib helpers for the REFIT NILM pipeline.

Used by pipeline.preprocessing and pipeline.disaggregate to produce
consistent, memory-safe figures without duplicating boilerplate.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import ensure_dir, get_logger

logger = get_logger(__name__)


def save_fig(fig: plt.Figure, path: str, dpi: int = 120) -> str:
    """Save *fig* to *path*, close it, and return *path*.

    Always closes the figure after saving to prevent memory leaks in
    batch runs that generate many plots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Full output path including filename and extension.
    dpi : int
        Resolution in dots-per-inch.

    Returns
    -------
    str
        The path the figure was saved to.
    """
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


def style_axes(ax: plt.Axes,
               title: str = "",
               xlabel: str = "",
               ylabel: str = "",
               grid_alpha: float = 0.35) -> plt.Axes:
    """Apply consistent styling to *ax* and return it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    title : str
    xlabel : str
    ylabel : str
    grid_alpha : float
        Transparency of the background grid.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=grid_alpha)
    return ax
