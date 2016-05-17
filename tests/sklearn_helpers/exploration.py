"""Provide functions that help quickly explore datasets with sklearn."""
import munch
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from crunchers.pandas_helpers.transformations import apply_pairwise
from crunchers.sklearn_helpers.misc import repandasify

def pca_and_report(data, plot_comps=[1,2,3,4], verbose=True, pca=PCA(), data_labels=None):
    """Generate figures and tables to provide insight into PCA results."""
    r = munch.Munch()

    if data_labels is None:
        data_labels = pd.Series(["unlabeled"] * data.shape[0], name="Labels")

    r.data_labels = data_labels

    r.pft = pca.fit_transform(data)

    var_ratios = pca.explained_variance_ratio_
    r.var_ratios = pd.Series(var_ratios, index=range(1,1+len(var_ratios)))

    r.pcs = repandasify(array=r.pft, y_names=data.index.values, X_names=['PC {v_}'.format(v_=v+1) for v in range(len(r.pft[0]))])

    r.var_pairs = apply_pairwise(series=r.var_ratios, func=np.sum)

    with sns.color_palette(sns.color_palette("hls", 2)):
        with sns.axes_style("white"):
            g = sns.PairGrid(pd.concat([r.data_labels,r.pcs], axis=1), hue=data_labels.name)
            g.map_diag(plt.hist)
            g.map_lower(plt.scatter)
#             g.map_lower(sns.kdeplot, cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True), shade=True)
            g.add_legend()
            r.g = g

    return r
