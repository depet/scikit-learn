#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Dejan Petelin <dejan [five] elin [at] gmail [dot] com>
#         (based on GPML MATLAB Toolbox, version 3.2)
# Licence: BSD 3 clause

"""
The :mod:`sklearn.gpml` module implements Gaussian Process
based predictions.
"""

from .gp import GP
from . import cov
from . import inf
from . import lik
from . import mean
from . import util

__all__ = ['GP', 'cov', 'inf', 'lik', 'mean', 'util']
