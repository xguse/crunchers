"""Provide functions that help quickly explore datasets with sklearn."""
import munch
import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from scipy.stats import spearmanr
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns


from crunchers.pandas_helpers.transformations import apply_pairwise
from crunchers.sklearn_helpers.misc import repandasify


class KMeansReport(object):

    """Manage KMeans Clustering and exploration of results."""

    def __init__(self, data, n_clusters, seed=None, n_jobs=-1, palette='deep'):
        """Initialize instance but don't run any analysis.

        Args:
            n_clusters: (int or list of ints) used to set value of `k`
            seed: (int) set random seed.
        """
        self.state = None
        self.X = data
        self.seed = seed
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.palette = palette
        self.estimators = self.init_estimators()
        self.silhouette_results = {}
        self.silhouette_plots = {}

    def cluster(self):
        """Fit each estimator."""
        if self.state != 'fresh_init':
            self.init_estimators()

        for k, est in self.estimators.items():
            est.fit(self.X)

        self.state = 'fit'

    def eval_silhouette(self, verbose=True):
        """Evaluate each estimator via silhouette score."""
        for k in self.n_clusters:
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            cluster_labels = self.estimators[k].labels_

            results = munch.Munch()

            results.silhouette_avg = silhouette_score(self.X, cluster_labels)
            if verbose:
                print("For n_clusters = {k} The average silhouette_score is : {avg_sil}".format(k=k, avg_sil=results.silhouette_avg))

            # Compute the silhouette scores for each sample
            results.sample_silhouette_values = silhouette_samples(self.X, cluster_labels)
            self.silhouette_results[k] = results

    def plot_silhouette_results(self, feature_names=None, feature_space=None):
        """Perform plotting similar to that from sklearn link below.

        http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        """
        use_external_features = False

        if (feature_space is None) and (feature_names is None):
            # Choose the to features with the most variance.
            feature_names = list(self.X.var().sort_values(ascending=False).iloc[:2].index.values)
            feature_locs = (self.X.columns.get_loc(feature_names[0]),
                            self.X.columns.get_loc(feature_names[1]))

        elif (feature_space is None) and (feature_names is not None):
            feature_locs = (self.X.columns.get_loc(feature_names[0]),
                            self.X.columns.get_loc(feature_names[1]))

        elif (feature_space is not None):
            use_external_features = True

            if (feature_names is not None):
                try:
                    feature_locs = (feature_space.columns.get_loc(feature_names[0]),
                                    feature_space.columns.get_loc(feature_names[1]))
                except AttributeError as exc:
                    if "object has no attribute 'columns'" in exc.args[0]:
                        msg = "Argument `feature_space` is expected to be a Dataframe, not '{type}'".format(type=type(feature_space))
                        raise TypeError(msg)

            elif (feature_names is None):
                feature_names = list(feature_space.var().sort_values(ascending=False).iloc[:2].index.values)
                feature_locs = (feature_space.columns.get_loc(feature_names[0]),
                                feature_space.columns.get_loc(feature_names[1]))

        if not self.silhouette_results:
            self.eval_silhouette(verbose=True)

        X = np.array(self.X)

        for k in self.n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (k+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (k + 1) * 10])

            y_lower = 10
            for cluster_i in range(k):
                cluster_name = cluster_i + 1

                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = self.silhouette_results[k].sample_silhouette_values[self.estimators[k].labels_ == cluster_i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # Create our colormap and colorlist
                custom_cmap = ListedColormap(sns.palettes.color_palette(self.palette))
                color = custom_cmap(float(cluster_i) / k)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_name))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=self.silhouette_results[k].silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = custom_cmap(self.estimators[k].labels_.astype(float) / k)

            if not use_external_features:
                ax2.scatter(X[:, feature_locs[0]], X[:, feature_locs[1]], marker='.', s=30, lw=0, alpha=0.7, c=colors)
            else:
                ax2.scatter(X[:, feature_locs[0]], X[:, feature_locs[1]], marker='.', s=30, lw=0, alpha=0.7, c=colors)

            # Labeling the clusters
            centers = self.estimators[k].cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200)

            for cluster_i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='${cluster_name}$'.format(cluster_name=cluster_i+1), alpha=1, s=50)

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for {feature}".format(feature=feature_names[0]))
            ax2.set_ylabel("Feature space for {feature}".format(feature=feature_names[1]))

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % k),
                         fontsize=14, fontweight='bold')

            plt.show()

    def init_estimators(self):
        """Set up and return dictionary of estimators with key = `n_clusters`."""
        if isinstance(self.n_clusters, int):
            self.n_clusters = [self.n_clusters]

        estimators = {}

        for k in self.n_clusters:
            estimators[k] = KMeans(n_clusters=k,
                                   init='k-means++',
                                   n_init=20, max_iter=1000,
                                   precompute_distances='auto',
                                   verbose=0, random_state=self.seed,
                                   n_jobs=self.n_jobs)

        self.state = 'fresh_init'

        return estimators


class PCAReport(object):

    """Manage PCA and exploration of results."""

    def __init__(self, data, pca=None, n_components=None, data_labels=None, color_palette=None, label_colors=None, name=None):
        """Set critical information.

        data: (pandas.DataFrame) the data, duh.
        n_components: (int) number of PCs to calculate.
        data_labels: (pandas.Series) with **SAME INDEX** (or MultiIndex) as `data` containing group labels.
        color_palette: (str or dict?) name of color palette to use.
        label_colors: (dict) dictionary mapping data labels to the color short-cut you want assigned to the label.
        name: (str) used for legend labels.
        """
        if pca is None:
            pca = PCA()

        if data_labels is None:
            data_labels = pd.Series(["Unlabeled"] * data.shape[0], name="Labels")
        else:
            data_labels.name = "Labels"

        if color_palette is None:
            color_palette = 'bright'

        self.name = name
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

        self._correlation = {'pearsonr': pearsonr,
                             'spearmanr': spearmanr}

        self.loadings = munch.Munch({kind: None for kind in self._correlation.keys()})

    @property
    def n_components(self):
        """Provide access to the number of PCs."""
        return self.pca.n_components

    @n_components.setter
    def n_components(self, n_components):
        assert isinstance(n_components, int)
        self.pca.n_components = n_components

    def get_pcs(self, rerun=True):
        """Fit and Transform via our local PCA object; store results in `self.pcs`."""
        if rerun:
            pft = self.pca.fit_transform(self.data)
        else:
            pft = self.pca.transform(self.data)

        self.pcs = repandasify(array=pft,
                               y_names=self.data.index,
                               X_names=['PC {v_}'.format(v_=v+1) for v in range(len(pft[0]))])

        var_ratios = self.pca.explained_variance_ratio_
        self.var_ratios = pd.Series(var_ratios, index=range(1, 1+len(var_ratios)))

        for kind in self._correlation.keys():
            self.loadings[kind] = self.get_loading_corr(kind=kind)

    def get_loading_corr(self, kind='pearsonr'):
        """Return dataframe of correlation based "loadings" repective of `kind`."""
        self._validate_correlation_kind(kind)

        corr = OrderedDict()

        for pc in self.pcs.columns.values:
            correlation = self._correlation[kind]
            r, p = list(zip(*self.data.apply(lambda col: correlation(col, self.pcs[pc]), axis=0)))
            corr[pc] = r

        return pd.DataFrame(corr, index=self.data.columns.values)

    # def get_strongest_loadings(self, kind, corr_val=0.7):
    #     """Return mapping of PC to set of variable names with the strongest loading correlations."""
    #     self._validate_correlation_kind(kind)
    #
    #     loadings = self.loadings[kind]
    #     strongest = loadings.abs() >= corr_val
    #
    #     m = {}
    #
    #     for pc in loadings.columns.values:
    #         m[pc] = set(loadings[strongest[pc]].index.values)
    #
    #     return m

    def filter_by_loadings(self, kind, column, hi_thresh, lo_thresh):
        """Return index of row names.

        kind (str): either ['pearsonr','spearmanr']
        column (str): which PC column to filter
        hi_thresh (float): retain rows with >= hi_thresh
        lo_thresh (float): retain rows with <= lo_thresh
        """
        loadings = self.loadings[kind]

        hi = loadings[column] >= hi_thresh
        lo = loadings[column] <= lo_thresh

        hits = loadings[column][hi | lo]

        return hits.index

    def plot_variance_accumulation(self, thresh=6, verbose=False):
        """Plot variance accumulation over PCs."""
        sns.set_palette(palette=self.color_palette, n_colors=None, desat=None, color_codes=True)

        if self.pcs is None:
            self.get_pcs()

        var_accum = self.var_ratios.cumsum()

        ax = var_accum[:thresh].plot(kind='line', figsize=(10, 10), label=self.name)
        ax.set_xlabel("PCs")
        ax.set_ylabel("Fraction of Variance Accumulated")

        var = round(self.var_ratios[:thresh].sum()*100, 1)

        if verbose:
            plt.text(x=1.5, y=var_accum.max()*0.8,
                     s="{var}% of variance represented.".format(var=var),
                     fontsize=14)

        self.var_accum_plot = ax

    def plot_variance_decay(self, thresh=6, verbose=False):
        """Plot variance decay over PCs."""
        sns.set_palette(palette=self.color_palette, n_colors=None, desat=None, color_codes=True)

        if self.pcs is None:
            self.get_pcs()

        ax = self.var_ratios[:thresh].plot(kind='line', figsize=(10, 10), label=self.name)
        ax.set_xlabel("PCs")
        ax.set_ylabel("Fraction of Variance")

        var = round(self.var_ratios[:thresh].sum()*100, 1)

        if verbose:
            plt.text(x=thresh*0.6, y=self.var_ratios.max()*0.8,
                     s="{var}% of variance represented.".format(var=var),
                     fontsize=14)

        self.var_decay_plot = ax

    def plot_pcs(self, components=None, label_colors=None, diag='kde', diag_kws=None, **kwargs):
        """Plot scatter-plots below the diagonal and density plots on the diagonal.

        components (list): list of components to plot

        label_colors = {'label1':'g',
                        'label2':'r',
                        'label3':'b'
                        }
        """
        diag_dict = {'kde': (sns.kdeplot, {'cut': 2, 'shade': True}),
                     'rug': (sns.rugplot, {'height': 0.25}),
                     'hist': (plt.hist, {})}

        if diag not in diag_dict.keys():
            msg = """"{diag}" is not in list of valid options: {opts}.""".format(diag=diag, opts=diag_dict.keys())
            raise ValueError(msg)

        if diag_kws is None:
            diag_kws = {}

        sns.set_palette(palette=self.color_palette, n_colors=None, desat=None, color_codes=True)

        if self.pcs is None:
            self.get_pcs()

        if label_colors is None:
            label_colors = self.label_colors

        if components is not None:
            if not isinstance(components, list):
                raise TypeError("`components` should be a list.")
            components = self._pc_numbers_to_names(components)
            plt_data = pd.concat([self.data_labels, self.pcs[components]], axis=1)
        else:
            plt_data = pd.concat([self.data_labels, self.pcs], axis=1)

        with sns.axes_style("white"):

            g = sns.PairGrid(plt_data, hue=self.data_labels.name, palette=label_colors, diag_sharey=False,)

            try:
                g.map_diag(diag_dict[diag][0], **diag_dict[diag][1], **diag_kws)
            except ZeroDivisionError:
                print("Something went wrong with {diag}. Using rugplot for diagonal.".format(diag=diag))
                g.map_diag(diag_dict['rug'][0], **diag_dict['rug'][1])
            except:
                raise

            g.map_lower(plt.scatter, **kwargs)
            g.add_legend()
            self.pc_plot = g

    def _pc_numbers_to_names(self, pcs):
        """Take pcs = [1,4,6]; return ['PC 1','PC 4','PC 6']."""
        return ['PC {n}'.format(n=n) for n in pcs]

    def _validate_correlation_kind(self, kind):
        correlation = self._correlation
        if kind not in correlation.keys():
            m = "The correlation kind provided ({provided}) is not a valid value: {valid}.".format(provided=kind,
                                                                                                   valid=list(correlation.keys()))
            raise ValueError(m)
        else:
            pass
