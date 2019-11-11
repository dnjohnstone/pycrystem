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
import dask.array as da
import pytest

from pyxem.signals.xray_diffraction1d import XrayDiffraction1D
from pyxem.signals.xray_diffraction1d import LazyXrayDiffraction1D


class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self, electron_diffraction1d):
        xray_diffraction1d = XrayDiffraction1D(electron_diffraction1d)
        xray_diffraction1d.set_experimental_parameters(beam_energy=3,
                                                       camera_length=3,
                                                       exposure_time=1)
        assert isinstance(xray_diffraction1d, XrayDiffraction1D)

    def test_set_scan_calibration(self, electron_diffraction1d):
        xray_diffraction1d = XrayDiffraction1D(electron_diffraction1d)
        xray_diffraction1d.set_scan_calibration(19)
        assert isinstance(electron_diffraction1d, XrayDiffraction1D)

    @pytest.mark.parametrize('calibration', [1, 0.017, 0.5, ])
    def test_set_diffraction_calibration(self,
                                         electron_diffraction1d,
                                         calibration):
        xray_diffraction1d = XrayDiffraction1D(electron_diffraction1d)
        xray_diffraction1d.set_diffraction_calibration(calibration)
        dx = xray_diffraction1d.axes_manager.signal_axes[0]
        assert dx.scale == calibration


class TestComputeAndAsLazyXray1D:

    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyXrayDiffraction1D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == XrayDiffraction1D
        assert not hasattr(s.data, 'compute')
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_3d_data_compute(self):
        dask_array = da.random.random((4, 10, 15),
                                      chunks=(1, 10, 15))
        s = LazyXrayDiffraction1D(dask_array)
        s.compute()
        assert s.__class__ == XrayDiffraction1D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = XrayDiffraction1D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyXrayDiffraction1D
        assert hasattr(s_lazy.data, 'compute')
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_3d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = XrayDiffraction1D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyXrayDiffraction1D
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_class_assignment(self, xray_diffraction1d):
        xray_diffraction1d = XrayDiffraction1D(electron_diffraction1d)
        xray_diffraction1d.decomposition()
        assert isinstance(xray_diffraction1d, XrayDiffraction1D)