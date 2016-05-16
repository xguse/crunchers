"""Provide helper functions for working with scikit-learn based objects."""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ..sklearn_helpers.misc import repandasify

def confusion_matrix_to_pandas(cm, labels):
    """Return the confusion matrix as a pandas dataframe.

    It is created from the confusion matrix stored in `cm` with rows and columns
    labeled with `labels`.
    """
    return pd.DataFrame(data=cm, index=labels, columns=labels)


def normalize_confusion_matrix(cm):
    """Return confusion matrix with values as fractions of outcomes instead of specific cases."""
    try:
        return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    except ValueError as exc:
        if "Shape of passed values is" in exc:
            raise Exception()


def plot_confusion_matrix(cm, labels=None, cmap='Blues', title=None, norm=False, context=None, annot=True):
    """Plot and return the confusion matrix heatmap figure."""
    if labels is None:
        labels = True

    if isinstance(labels, collections.Iterable) and not isinstance(labels,str):
        labels = [label.title() for label in labels]

    if norm:
        cm = normalize_confusion_matrix(cm)

    if title is None:
        if norm:
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix"

    if context is None:
        context = sns.plotting_context("notebook", font_scale=1.5)

    with context:
        ax = sns.heatmap(cm,
                         xticklabels=labels,
                         yticklabels=labels,
                         cmap=cmap,
                         annot=annot
                        )
        ax.set_title(title)

        return ax
