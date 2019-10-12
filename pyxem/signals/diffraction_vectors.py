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

from hyperspy.signals import BaseSignal, Signal1D

from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

from warnings import warn

from pyxem.signals import push_metadata_through
from pyxem.utils.vector_utils import calculate_norms, calculate_norms_ragged
from pyxem.utils.vector_utils import get_npeaks


"""
Base Signal class for n-dimensional diffraction vectors.

There are two cases that are supported:

1. A map of diffraction vectors, which will in general be a ragged signal of
signals. It the navigation dimensions of the map and contains a signal for each
peak at every position.

2. A list of diffraction vectors with dimensions < x | n > where x is the
number of vectors and n is the dimensionality of the vectors.
"""


class BaseDiffractionVectors(BaseSignal):
    """Two-dimensional diffraction vectors in reciprocal Angstrom units.

    Attributes
    ----------
    detector_coordinates : DetectorCoordinates2D
        Array of 2-vectors describing detector coordinates associated with each
        diffraction vector.
    """
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.detector_coordinates = None

    def get_npeaks_map(self, binary=False):
        """Map of the number of vectors at each navigation position.

        Parameters
        ----------
        binary : boolean
            If True a binary image with diffracting pixels taking value == 1 is
            returned.

        Returns
        -------
        crystim : Signal2D
            2D map of diffracting pixels.
        """
        crystim = self.map(get_npeaks, inplace=False).as_signal2D((0, 1))

        if binary == True:
            crystim = crystim == 1

        crystim.change_dtype('float')

        # Set calibration to same as signal
        x = crystim.axes_manager.signal_axes[0]
        y = crystim.axes_manager.signal_axes[1]

        x.name = 'x'
        x.scale = self.axes_manager.navigation_axes[0].scale
        x.units = 'nm'

        y.name = 'y'
        y.scale = self.axes_manager.navigation_axes[0].scale
        y.units = 'nm'

        return crystim

    def get_magnitudes(self, *args, **kwargs):
        """Calculate the magnitude of diffraction vectors.

        Parameters
        ----------
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to map().

        Returns
        -------
        magnitudes : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containging an array of gvector magnitudes at each
            navigation position.

        """
        # If ragged the signal axes will not be defined
        if len(self.axes_manager.signal_axes) == 0:
            magnitudes = self.map(calculate_norms_ragged,
                                  inplace=False,
                                  *args, **kwargs)
        # Otherwise easier to calculate.
        else:
            magnitudes = BaseSignal(calculate_norms(self))
            magnitudes.axes_manager.set_signal_dimension(0)

        return magnitudes

    def get_magnitude_histogram(self, bins, *args, **kwargs):
        """Obtain a histogram of gvector magnitudes.

        Parameters
        ----------
        bins : numpy array
            The bins to be used to generate the histogram.
        *args:
            Arguments to get_magnitudes().
        **kwargs:
            Keyword arguments to get_magnitudes().

        Returns
        -------
        ghis : Signal1D
            Histogram of gvector magnitudes.

        """
        gmags = self.get_magnitudes(*args, **kwargs)

        if len(self.axes_manager.signal_axes) == 0:
            glist = []
            for i in gmags._iterate_signal():
                for j in np.arange(len(i[0])):
                    glist.append(i[0][j])
            gs = np.asarray(glist)
            gsig = Signal1D(gs)
            ghis = gsig.get_histogram(bins=bins)

        else:
            ghis = gmags.get_histogram(bins=bins)

        ghis.axes_manager.signal_axes[0].name = 'k'
        ghis.axes_manager.signal_axes[0].units = '$A^{-1}$'

        return ghis

    def get_unique_vectors(self, distance_threshold=0.01,
                           method='distance_comparison', min_samples=1,
                           return_clusters=False):
        """Returns diffraction vectors considered unique by:
        strict comparison, distance comparison with a specified
        threshold, or by clustering using DBSCAN [1].

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between diffraction vectors for them to
            be considered unique diffraction vectors. If
            distance_threshold==0, the unique vectors will be determined
            by strict comparison.
        method : str
            The method to use to determine unique vectors. Valid methods
            are 'strict', 'distance_comparison' and 'DBSCAN'.
            'strict' returns all vectors that are strictly unique and
            corresponds to distance_threshold=0.
            'distance_comparison' checks the distance between vectors to
            determine if some should belong to the same unique vector,
            and if so, the unique vector is iteratively updated to the
            average value.
            'DBSCAN' relies on the DBSCAN [1] clustering algorithm, and
            uses the Eucledian distance metric.
        min_samples : int, optional
            The minimum number of not strictly identical vectors within
            one cluster for the cluster to be considered a core sample,
            i.e. to not be considered noise. Only used for method='DBSCAN'.
        return_clusters : bool, optional
            If True (False is default), the DBSCAN clustering result is
            returned. Only used for method='DBSCAN'.

        References
        ----------
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.
            cluster.DBSCAN.html

        Returns
        -------
        unique_peaks : DiffractionVectors2D
            The unique diffraction vectors.
        clusters : DBSCAN
            The results from the clustering, given as class DBSCAN.
            Only returned if method='DBSCAN' and return_clusters=True.
        """
        # Flatten the array of peaks to reach dimension (n, 2), where n
        # is the number of peaks.
        peaks_all = np.concatenate([
            peaks.ravel() for peaks in self.data.flat]).reshape(-1, 2)

        # A distance_threshold of 0 implies a strict comparison. So in that
        # case, a warning is raised unless the specified method is 'strict'.
        if distance_threshold == 0:
            if method is not 'strict':
                warn(message='distance_threshold=0 was given, and therefore ' +
                     'a strict comparison is used, even though the ' +
                     'specified method was ' + method + '.')
                method = 'strict'

        if method == 'strict':
            unique_peaks = np.unique(peaks_all, axis=0)

        elif method == 'distance_comparison':
            unique_vectors, unique_counts = np.unique(
                peaks_all, axis=0, return_counts=True)

            unique_peaks = np.array([[0, 0]])
            unique_peaks_counts = np.array([0])

            while unique_vectors.shape[0] > 0:
                unique_vector = unique_vectors[0]
                distances = distance_matrix(
                    np.array([unique_vector]), unique_vectors)
                indices = np.where(distances < distance_threshold)[1]

                new_count = indices.size
                new_unique_peak = np.array([np.average(
                    unique_vectors[indices], weights=unique_counts[indices],
                    axis=0)])

                unique_peaks = np.append(unique_peaks, new_unique_peak,
                                         axis=0)

                unique_peaks_counts = np.append(unique_peaks_counts,
                                                new_count)
                unique_vectors = np.delete(unique_vectors, indices, axis=0)
                unique_counts = np.delete(unique_counts, indices, axis=0)
            unique_peaks = np.delete(unique_peaks, [0], axis=0)

        elif method == 'DBSCAN':
            # All peaks are clustered by DBSCAN so that peaks within
            # one cluster are separated by distance_threshold or less.
            unique_vectors, unique_vectors_counts = np.unique(
                peaks_all, axis=0, return_counts=True)
            clusters = DBSCAN(
                eps=distance_threshold, min_samples=min_samples,
                metric='euclidean').fit(
                unique_vectors, sample_weight=unique_vectors_counts)
            unique_labels, unique_labels_count = np.unique(
                clusters.labels_, return_counts=True)
            unique_peaks = np.zeros((unique_labels.max() + 1, 2))
            # For each cluster, a center of mass is calculated based
            # on all the peaks within the cluster, and the center of
            # mass is taken as the final unique vector position.
            for n in np.arange(unique_labels.max() + 1):
                peaks_n_temp = unique_vectors[clusters.labels_ == n]
                peaks_n_counts_temp = unique_vectors_counts[
                    clusters.labels_ == n]
                unique_peaks[n] = np.average(
                    peaks_n_temp, weights=peaks_n_counts_temp,
                    axis=0)
        # Manipulate into DiffractionVectors2D class
        if unique_peaks.size > 0:
            unique_peaks = DiffractionVectors2D(unique_peaks)
            unique_peaks.axes_manager.set_signal_dimension(1)
        if return_clusters and method == 'DBSCAN':
            return unique_peaks, clusters
        else:
            return unique_peaks
