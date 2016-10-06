"""Provide functions that help quickly explore datasets with sklearn."""
import munch
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from crunchers.pandas_helpers.transformations import apply_pairwise
from crunchers.sklearn_helpers.misc import repandasify

class PCAReport(object):
    """docstring for PCAReport"""
    def __init__(self, data, pca=None, n_components=None, data_labels=None, color_palette=None, label_colors=None):
        """Set critical information."""
        if pca is None:
            pca = PCA()

        if data_labels is None:
            data_labels = pd.Series(["Unlabeled"] * data.shape[0], name="Labels")
        else:
            data_labels.name = "Labels"

        if color_palette is None:
            color_palette = 'bright'

        self.data = data.copy()
        self.pca = pca

        if n_components is None:
            self.n_components = self.pca.n_components
        else:
            self.n_components = n_components

        self.data_labels = data_labels
        self.pcs = None
        self.pc_plot = None
        self.var_decay_plot = None
        self.color_palette = color_palette
        self.label_colors = label_colors

    @property
    def n_components(self):
      """Provide access to the number of PCs."""
      return self.pca.n_components

    @n_components.setter
    def n_components(self, n_components):
      assert isinstance(n_components, int)
      self.pca.n_components = n_components

    def get_pcs(self):
        """Fit and Transform via our local PCA object; store results in `self.pcs`."""
        pft = self.pca.fit_transform(self.data)
        self.pcs = repandasify(array=pft,
                               y_names=self.data.index.values,
                               X_names=['PC {v_}'.format(v_=v+1) for v in range(len(pft[0]))])

        var_ratios = self.pca.explained_variance_ratio_
        self.var_ratios = pd.Series(var_ratios, index=range(1,1+len(var_ratios)))

    def plot_variance_decay(self, thresh=6):
        sns.set_palette(palette=self.color_palette, n_colors=None, desat=None, color_codes=True)

        if self.pcs is None:
            self.get_pcs()

        plt = self.var_ratios[:thresh].plot(kind='line', figsize=(10,10))
        plt.set_xlabel("PCs")
        plt.set_ylabel("Fraction of Variance")

        var = round(self.var_ratios[:thresh].sum()*100,1)

        plt.text(x=thresh*0.6,y=self.var_ratios.max()*0.8,
                s="{var}% of variance represented.".format(var=var),
                fontsize=14
               )
        self.var_decay_plot = plt

    def plot_pcs(self, components=None, label_colors=None):
        """Plot scatter-plots below the diagonal and density plots on the diagonal.

        label_colors = {'label1':'g',
                        'label2':'r',
                        'label3':'b'
                        }
        """
        sns.set_palette(palette=self.color_palette, n_colors=None, desat=None, color_codes=True)

        if self.pcs is None:
            self.get_pcs()

        if label_colors is None:
            label_colors = self.label_colors

        if components is not None:
            if not isinstance(components, list):
                raise TypeError("`components` should be a list.")
            components = self._pc_numbers_to_names(components)
            plt_data = pd.concat([self.data_labels,self.pcs[components]], axis=1)
        else:
            plt_data = pd.concat([self.data_labels,self.pcs], axis=1)

        with sns.axes_style("white"):
            g = sns.PairGrid(plt_data, hue=self.data_labels.name, palette=label_colors)
            g.map_diag(sns.kdeplot)
            g.map_lower(plt.scatter)
            g.add_legend()
            self.pc_plot = g

    def _pc_numbers_to_names(self, pcs):
        """Take pcs = [1,4,6]; return ['PC 1','PC 4','PC 6']."""
        return ['PC {n}'.format(n=n) for n in pcs]
