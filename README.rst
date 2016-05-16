=========
Crunchers
=========

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        | |codecov|
    * - package
      - |version| |downloads|

.. |docs| image:: https://readthedocs.org/projects/crunchers/badge/?style=flat
    :target: https://readthedocs.org/projects/crunchers
    :alt: Documentation Status

.. |travis| image:: https://img.shields.io/travis/xguse/crunchers/master.svg?style=flat&label=Travis
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/xguse/crunchers

.. |appveyor| image:: https://img.shields.io/appveyor/ci/xguse/crunchers/master.svg?style=flat&label=AppVeyor
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/xguse/crunchers


.. |codecov| image:: https://img.shields.io/codecov/c/github/xguse/crunchers/master.svg?style=flat&label=Codecov
    :alt: Coverage Status
    :target: https://codecov.io/github/xguse/crunchers




.. |version| image:: https://img.shields.io/pypi/v/crunchers.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/crunchers

.. |downloads| image:: https://img.shields.io/pypi/dm/crunchers.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/crunchers

A library that provides a set of helper functions etc that I tend to use a lot when crunching data with scikit-learn, pandas, et al.

* Free software: BSD license

Installation
============

::

    pip install crunchers

Documentation
=============

https://crunchers.readthedocs.org/

Development
===========

To run the all tests run::

    tox
