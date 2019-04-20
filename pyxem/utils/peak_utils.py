# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import math

def mapping_indeces_dictionary(dictionary, key):
    """It allows to index through dictionaries as BaseSignal data.
    Parameters
    ----------
    dictionary : np.array([dictionary])
        A BaseSignal data point, containing a dictionary.
    key: string
        The dictionary key to be indexed.
    Returns
    -------
    norms : np.array([indexed dictionary])
        An BaseSignal data point, containg the indexed value from the dictionary.
    """
    return dictionary[0][key]