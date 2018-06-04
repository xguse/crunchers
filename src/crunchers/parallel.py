#!/usr/bin/env python
"""Provide a directly importable home for things needed to work with concurrent.futures."""

# Imports


from collections import defaultdict, Sequence, namedtuple


# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


# Functions
GridKey = namedtuple(typename='GridKey', field_names=['X_hyps','ctrl_coefs','outcomes'])
GridValues = namedtuple(typename='GridValues', field_names=['X_hyps','ctrl_coefs','outcomes'])