#!/usr/bin/env python
"""Provide streamlined plotting functions."""

# Imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from munch import Munch, munchify


# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


# Functions
def plot_scatter_pairs(data, variables=None, palette='bright', data_labels=None, label_colors=None, diag='kde', diag_kws=None):
    """Plot scatter-plots below the diagonal and density plots on the diagonal.

    data (DataFrame):
    variables (list):
    palette (?):
    data_labels (Series):
    label_colors (dict):
    diag (str):
    diag_kws (dict):


    label_colors = {'label1':'g',
                    'label2':'r',
                    'label3':'b'
                    }
    """
    diag_dict = {'kde': (sns.kdeplot, {'cut': 2, 'shade':True}),
                 'rug': (sns.rugplot, {'height': 0.25}),
                 'hist': (plt.hist, {})}

    # Setup and Validation
    if data_labels is None:
        data_labels = pd.Series(["Unlabeled"] * data.shape[0], name="Labels")

    if diag not in diag_dict.keys():
        msg = """"{diag}" is not in list of valid options: {opts}.""".format(diag=diag,opts=diag_dict.keys())
        raise ValueError(msg)

    if diag_kws is None:
        diag_kws = {}

    if label_colors is None:
        label_colors = {}

    if variables is not None:
        if not isinstance(variables, list):
            raise TypeError("`variables` should be a list.")

        plt_data = pd.concat([data_labels,data[variables]], axis=1)
    else:
        plt_data = pd.concat([data_labels,data], axis=1)

    # Begin Plotting
    sns.set_palette(palette=palette, n_colors=None, desat=None, color_codes=True)

    with sns.axes_style("white"):

        g = sns.PairGrid(plt_data, hue=data_labels.name, palette=label_colors, diag_sharey=False,)

        try:
            g.map_diag(diag_dict[diag][0], **diag_dict[diag][1], **diag_kws)

        except ZeroDivisionError:
            print("Something went wrong with {diag}. Using rugplot for diagonal.".format(diag=diag))
            g.map_diag(diag_dict['rug'][0], **diag_dict['rug'][1])
        except:
            raise

        g.map_lower(plt.scatter)

        g.add_legend()

        return g

def conditional_violin_swarm_facets(data, row, col, x, y, label_colors=None, highlight_meds=False, layout_pad=None, size=5, **kwargs):
    """Plot FacetGrid violin/swarm plots of 'long' data 'conditioned' on `row`, `col`, and `x`.

    data (DataFrame): 'longform' or 'melted' data with at least three id_vars.
    row (str): string name of an id_var in `data`
    col (str): string name of an id_var in `data`
    x (str): string name of an id_var in `data`
    y (str): string name of an value_name in `data`
    label_colors (dict):
    highlight_meds (Bool):
    """
    g = sns.FacetGrid(data, row=row, col=col, size=size, **kwargs)
    g.map(sns.violinplot, x, y, color="#C5C5C5", cut=0)
    g.map(sns.swarmplot, x, y, color='#D68235')

    if highlight_meds:
        g.map(group_med_line, x, y, colors=label_colors)


    plt.tight_layout(w_pad=layout_pad)

    return g


def conditional_barplot_facets(data, row, col, x, y, label_colors=None, color=None, facet_color=None, palette=None, facet_hue=None, highlight_meds=False, layout_pad=None, size=5, rot_x=None, rot_y=None, **kwargs):
    """Plot FacetGrid barplots of 'long' data 'conditioned' on `row`, `col`, and `x`.

    data (DataFrame): 'longform' or 'melted' data with at least three id_vars.
    row (str): string name of an id_var in `data`
    col (str): string name of an id_var in `data`
    x (str): string name of an id_var in `data`
    y (str): string name of an value_name in `data`
    label_colors (dict):
    highlight_meds (Bool):
    """


    g = sns.FacetGrid(data, row=row, col=col, size=size, **kwargs)
    g.map(sns.barplot, x, y, palette=palette, color=facet_color, hue=facet_hue)


    for ax in g.axes.flat:

        if rot_x is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(rot_x)

        if rot_y is not None:
            for label in ax.get_yticklabels():
                label.set_rotation(rot_y)


    plt.tight_layout(w_pad=layout_pad)

    return g


# FacetGrid Mappables
def group_med_line(x, y, **kwargs):
    opts = Munch(kwargs)

    y.index = x.values
    data = pd.DataFrame(y)
    meds = dict(data.reset_index().groupby('index').agg(np.median).iloc[:,0])

    if 'colors' not in opts.keys():
        opts.colors = {name: 'k' for name in meds.keys()}

    for name, val in meds.items():
        plt.axhline(y=val, linewidth=2, color=opts.colors[name], ls='solid', label=name, alpha=1)
