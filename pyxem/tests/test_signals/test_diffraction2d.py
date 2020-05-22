# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import pytest
import numpy as np
import dask.array as da
<<<<<<< HEAD
import hyperspy.api as hs
=======
import pyxem as pxm

from hyperspy.signals import Signal1D, Signal2D
>>>>>>> 56aa0b1780fc6379e6e85e4fc725db34e4b028c8

from pyxem.signals.diffraction2d import Diffraction2D, LazyDiffraction2D
from pyxem.signals.polar_diffraction2d import PolarDiffraction2D
from pyxem.detectors.generic_flat_detector import GenericFlatDetector
from pyxem.signals.diffraction1d import Diffraction1D


<<<<<<< HEAD
class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyDiffraction2D(dask_array)
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Diffraction2D(data)
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert data.shape == s_lazy.data.shape
=======
class TestSimpleMaps:
    # Confirms that maps run without error.

    def test_center_direct_beam_cross_correlate(self, diffraction2d):
        assert isinstance(diffraction2d, Diffraction2D)
        diffraction2d.center_direct_beam(method='cross_correlate', radius_start=1, radius_finish=3)
        assert isinstance(diffraction2d, Diffraction2D)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_center_direct_beam_fail(self, diffraction2d):
        diffraction2d.center_direct_beam(method="Invalid value")

    def test_center_direct_beam_in_small_region(self, diffraction2d):
        assert isinstance(diffraction2d, Diffraction2D)
        diffraction2d.center_direct_beam(method='blur',
                                               sigma=5,
                                               square_width=3)
        assert isinstance(diffraction2d, Diffraction2D)

    def test_apply_affine_transformation(self, diffraction2d):
        diffraction2d.apply_affine_transformation(
            D=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]]))
        assert isinstance(diffraction2d, Diffraction2D)
>>>>>>> 56aa0b1780fc6379e6e85e4fc725db34e4b028c8

    def test_apply_affine_transforms_paths(self, diffraction2d):
        D = np.array([[1., 0.9, 0.],
                      [1.1, 1., 0.],
                      [0., 0., 1.]])
        s = Signal2D(np.asarray([[D, D], [D, D]]))
        static = diffraction2d.apply_affine_transformation(D, inplace=False)
        dynamic = diffraction2d.apply_affine_transformation(s, inplace=False)
        assert np.allclose(static.data, dynamic.data, atol=1e-3)

    def test_apply_affine_transformation_with_casting(self, diffraction2d):
        diffraction2d.change_dtype('uint8')
        transformed_dp = Diffraction2D(diffraction2d).apply_affine_transformation(
            D=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.2]]), order=2, keep_dtype=True, inplace=False)
        assert transformed_dp.data.dtype == 'uint8'

    methods = ['average', 'nan']

    @pytest.mark.parametrize('method', methods)
    def test_remove_dead_pixels(self, diffraction2d, method):
        dpr = diffraction2d.remove_deadpixels([[1, 2], [5, 6]], method,
                                                    inplace=False)
        assert isinstance(dpr, Diffraction2D)


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_interactive_virtual_image(self, diffraction2d):
        roi = pxm.roi.CircleROI(3, 3, 5)
        diffraction2d.plot_interactive_virtual_image(roi)

    def test_get_virtual_image(self, diffraction2d):
        roi = pxm.roi.CircleROI(3, 3, 5)
        diffraction2d.get_virtual_image(roi)


class TestDirectBeamMethods:

    @pytest.mark.parametrize('mask_expected', [
        (np.array([
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, True, True, False, False, False],
            [False, False, True, True, True, True, False, False],
            [False, False, True, True, True, True, False, False],
            [False, False, False, True, True, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]]),),
    ])
    def test_get_direct_beam_mask(self, diffraction2d, mask_expected):
        mask_calculated = diffraction2d.get_direct_beam_mask(2)
        assert isinstance(mask_calculated, Signal2D)
        assert np.equal(mask_calculated, mask_expected)


class TestGainNormalisation:

    @pytest.mark.parametrize('dark_reference, bright_reference', [
        (-1, 1),
        (0, 1),
        (0, 256),
    ])
    def test_apply_gain_normalisation(self, diffraction2d,
                                      dark_reference, bright_reference):
        dpr = diffraction2d.apply_gain_normalisation(
            dark_reference=dark_reference, bright_reference=bright_reference,
            inplace=False)
        assert dpr.max() == bright_reference
        assert dpr.min() == dark_reference


class TestPeakFinding:
    # This is assertion free testing

    @pytest.fixture
    def ragged_peak(self):
        """
        A small selection of peaks in an ElectronDiffraction2D, to allow
        flexibilty of test building here.
        """
        pattern = np.zeros((2, 2, 128, 128))
        pattern[:, :, 40:42, 45] = 1
        pattern[:, :, 110, 30:32] = 1
        pattern[1, 0, 71:73, 21:23] = 1
        dp = Diffraction2D(pattern)

        return dp

    methods = ['zaefferer', 'laplacian_of_gaussians', 'difference_of_gaussians', 'stat', 'xc']

    @pytest.mark.parametrize('method', methods)
    # skimage internals
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_findpeaks_ragged(self, ragged_peak, method):
        if method == 'xc':
            disc = np.ones((2, 2))
            output = ragged_peak.find_peaks(method='xc',
                                            disc_image=disc,
                                            min_distance=3)
        else:
            output = ragged_peak.find_peaks(method=method,
                                            show_progressbar=False)


class TestAzimuthalIntegral1d:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    def test_unit(self, ones):
        dif = Diffraction2D(data=[[1, 1], [1, 1]])
        assert dif.unit is None
        dif.unit = "!23"
        assert dif.unit is None

    def test_unit_set(self, ones):
        assert ones.unit == "2th_deg"

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_azimuthal_integral_fast_2th_units(self, ones, unit):
        ones.unit = unit
        az = ones.get_azimuthal_integral1d(
            npt_rad=10, wavelength=1e-9, correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8], np.ones(8))

    def test_1d_azimuthal_integral_inplace(self, ones):
        az = ones.get_azimuthal_integral1d(
            npt_rad=10, correctSolidAngle=False, inplace=True,
        )
        assert isinstance(ones, Diffraction1D)
        np.testing.assert_array_equal(ones.data[0:8], np.ones((8)))
        assert az is None

    def test_1d_azimuthal_integral_fast_slicing(self, ones):
        ones.unit = "2th_rad"
        az = ones.get_azimuthal_integral1d(
            npt_rad=10,
            center=(5.5, 5.5),
            method="BBox",
            correctSolidAngle=False,
            radial_range=[0.0, 1.0],
        )
        np.testing.assert_array_equal(az.data[0:7], np.ones(7))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_axes_continuity(self, ones, unit):
        ones.unit = unit
        az1 = ones.get_azimuthal_integral1d(
            wavelength=1e-9,
            npt_rad=10,
            center=(5.5, 5.5),
            radial_range=[0.0, 1.0],
            method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[0].scale, 0.1)

    @pytest.mark.parametrize("radial_range", [None, [0.0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, np.pi]])
    @pytest.mark.parametrize("center", [None, [9, 9]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_1d_integration(
        self, ones, radial_range, azimuth_range, center, affine,
    ):
        az = ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            center=center,
            affine=affine,
            correctSolidAngle=False,
            inplace=False,
        )
        assert isinstance(az, Diffraction1D)

    def test_1d_azimuthal_integral_slow(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
        )

    def test_1d_azimuthal_integral_slow_shifted_center(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
        )

    def test_1d_azimuthal_integral_slow_mask(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
            mask=mask_bs,
        )

    def test_1d_azimuthal_integral_pyfai(self, ones):
        from pyFAI.detectors import Detector

        d = Detector(pixel1=1e-4, pixel2=1e-4)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            detector=d,
            detector_dist=1,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            unit="q_nm^-1",
        )

    def test_1d_azimuthal_integral_failure(self, ones):
        ones.unit = "k_nm^-1"
        integration = ones.get_azimuthal_integral1d(npt_rad=10)
        assert integration is None


class TestAzimuthalIntegral2d:
    @pytest.fixture
<<<<<<< HEAD
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    def test_2d_azimuthal_integral_fast(self, ones):
        az = ones.get_azimuthal_integral2d(
            npt_rad=10, npt_azim=10, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8, :], np.ones((8, 10)))

    def test_2d_azimuthal_integral_fast_slicing(self, ones):
        az1 = ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            center=(5.5, 5.5),
            radial_range=[0.0, 1.0],
            method="splitpixel",
            correctSolidAngle=False,
        )
        np.testing.assert_array_equal(az1.data[8:, :], np.zeros(shape=(2, 10)))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_2d_axes_continuity(self, ones, unit):
        ones.unit = unit
        az1 = ones.get_azimuthal_integral2d(
            wavelength=1e-9,
            npt_rad=10,
            npt_azim=20,
            center=(5.5, 5.5),
            radial_range=[0.0, 1.0],
            method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[1].scale, 0.1)

    def test_2d_azimuthal_integral_inplace(self, ones):
        az = ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            correctSolidAngle=False,
            inplace=True,
            method="BBox",
        )
        assert isinstance(ones, PolarDiffraction2D)
        np.testing.assert_array_equal(ones.data[0:8, :], np.ones((8, 10)))
        assert az is None

    @pytest.mark.parametrize("radial_range", [None, [0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, 0]])
    @pytest.mark.parametrize("correctSolidAngle", [True, False])
    @pytest.mark.parametrize("center", [None, [7, 7]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_2d_azimuthal_integral_fast_params(
        self, ones, radial_range, azimuth_range, correctSolidAngle, center, affine
    ):
        az = ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            wavelength=1e-9,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            correctSolidAngle=correctSolidAngle,
            center=center,
            affine=affine,
            method="BBox",
        )
        assert isinstance(az, PolarDiffraction2D)

    def test_2d_azimuthal_integral_failure(self, ones):
        ones.unit = "k_nm^-1"
        integration = ones.get_azimuthal_integral2d(npt_rad=10)
        assert integration is None

    def test_2d_azimuthal_integral_slow(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
        )

    def test_2d_azimuthal_integral_slow_shifted_center(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
        )

    def test_2d_azimuthal_integral_slow_mask(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
            mask=mask_bs,
        )

    def test_2d_azimuthal_integral_pyfai(self, ones):
        from pyFAI.detectors import Detector

        d = Detector(pixel1=1e-4, pixel2=1e-4)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            detector=d,
            detector_dist=1,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            unit="q_nm^-1",
        )


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    @pytest.mark.parametrize("stack", [True, False])
    def test_plot_integrated_intensity(self, stack, diffraction_pattern):
        if stack:
            diffraction_pattern = hs.stack([diffraction_pattern] * 3)
        roi = hs.roi.CircleROI(3, 3, 5)
        diffraction_pattern.plot_integrated_intensity(roi)

    def test_get_integrated_intensity(self, diffraction_pattern):
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = diffraction_pattern.get_integrated_intensity(roi)
        assert vi.data.shape == (2, 2)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 0

    @pytest.mark.parametrize("out_signal_axes", [None, (0, 1), (1, 2), ("x", "y")])
    def test_get_integrated_intensity_stack(self, diffraction_pattern, out_signal_axes):
        s = hs.stack([diffraction_pattern] * 3)
        s.axes_manager.navigation_axes[0].name = "x"
        s.axes_manager.navigation_axes[1].name = "y"

        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi, out_signal_axes)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 1
        if out_signal_axes == (1, 2):
            assert vi.data.shape == (2, 3, 2)
            assert vi.axes_manager.navigation_size == 2
            assert vi.axes_manager.signal_shape == (2, 3)
        else:
            assert vi.data.shape == (3, 2, 2)
            assert vi.axes_manager.navigation_size == 3
            assert vi.axes_manager.signal_shape == (2, 2)

    def test_get_integrated_intensity_out_signal_axes(self, diffraction_pattern):
        s = hs.stack([diffraction_pattern] * 3)
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi, out_signal_axes=(0, 1, 2))
        assert vi.axes_manager.signal_dimension == 3
        assert vi.axes_manager.navigation_dimension == 0
        assert vi.metadata.General.title == "Integrated intensity"
        assert (
            vi.metadata.Diffraction.intergrated_range
            == "CircleROI(cx=3, cy=3, r=5) of Stack of "
        )

    def test_get_integrated_intensity_error(
        self, diffraction_pattern, out_signal_axes=(0, 1, 2)
    ):
        roi = hs.roi.CircleROI(3, 3, 5)
        with pytest.raises(ValueError):
            _ = diffraction_pattern.get_integrated_intensity(roi, out_signal_axes)

    def test_get_integrated_intensity_linescan(self, diffraction_pattern):
        s = diffraction_pattern.inav[0, :]
        s.metadata.General.title = ""
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi)
        assert vi.data.shape == (2,)
        assert vi.axes_manager.signal_dimension == 1
        assert vi.axes_manager.navigation_dimension == 0
        assert vi.metadata.Diffraction.intergrated_range == "CircleROI(cx=3, cy=3, r=5)"
=======
    def diffraction_pattern_for_origin_variation(self):
        """
        Two diffraction patterns with easy to see radial profiles, wrapped
        in Diffraction2D  <2,2|3,3>
        """
        dp = Diffraction2D(np.zeros((2, 2, 4, 4)))
        dp.data = np.array(
            [[[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]],
                [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]])
        return dp

    def test_azimuthal_integral_slow(self,
                                     diffraction_pattern_for_origin_variation):
        origin = np.array([[[0, 0], [1, 1]], [[1.5, 1.5], [2, 3]]])
        detector = GenericFlatDetector(4, 4)
        ap = diffraction_pattern_for_origin_variation.get_azimuthal_integral(
            origin,
            detector=detector,
            detector_distance=1e9,
            wavelength=1, size_1d=4)
        expected = np.array([[[1.01127149e-07, 4.08790171e-01, 2.93595970e-01, 0.00000000e+00],
                              [2.80096084e-01, 4.43606853e-01, 1.14749573e-01, 0.00000000e+00]],
                             [[6.20952725e-01, 2.99225271e-01, 4.63002026e-02, 0.00000000e+00],
                              [5.00000000e-01, 3.43071640e-01, 1.27089232e-01, 0.00000000e+00]]])
        assert np.allclose(ap.data, expected, atol=1e-5)


class TestRadialProfile:

    @pytest.fixture
    def diffraction_pattern_for_radial(self):
        """
        Two diffraction patterns with easy to see radial profiles, wrapped
        in Diffraction2D  <2|8,8>
        """
        dp = Diffraction2D(np.zeros((2, 8, 8)))
        dp.data[0] = np.array([[0., 0., 2., 2., 2., 2., 0., 0.],
                               [0., 2., 3., 3., 3., 3., 2., 0.],
                               [2., 3., 3., 4., 4., 3., 3., 2.],
                               [2., 3., 4., 5., 5., 4., 3., 2.],
                               [2., 3., 4., 5., 5., 4., 3., 2.],
                               [2., 3., 3., 4., 4., 3., 3., 2.],
                               [0., 2., 3., 3., 3., 3., 2., 0.],
                               [0., 0., 2., 2., 2., 2., 0., 0.]])

        dp.data[1] = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [1., 1., 1., 1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1., 1., 1., 1.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.]])

        return dp

    @pytest.fixture
    def mask_for_radial(self):
        """
        An 8x8 mask array, to test that part of the radial average code.
        """
        mask = np.array([[0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [1., 0., 1., 1., 1., 1., 1., 1.],
                         [1., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.]])

        return mask

    def test_radial_profile_signal_type(self, diffraction_pattern_for_radial,
                                        mask_for_radial):
        rp = diffraction_pattern_for_radial.get_radial_profile()
        rp_mask = diffraction_pattern_for_radial.get_radial_profile(
            mask_array=mask_for_radial)

        assert isinstance(rp, Signal1D)
        assert isinstance(rp_mask, Signal1D)

    @pytest.fixture
    def axes_test_dp(self):
        dp_data = np.random.randint(0, 10, (2, 2, 10, 10))
        dp = Diffraction2D(dp_data)
        return dp

    def test_radial_profile_axes(self, axes_test_dp):
        n_scale = 0.5
        axes_test_dp.axes_manager.navigation_axes[0].scale = n_scale
        axes_test_dp.axes_manager.navigation_axes[1].scale = 2 * n_scale
        name = 'real_space'
        axes_test_dp.axes_manager.navigation_axes[0].name = name
        axes_test_dp.axes_manager.navigation_axes[1].units = name
        units = 'um'
        axes_test_dp.axes_manager.navigation_axes[1].name = units
        axes_test_dp.axes_manager.navigation_axes[0].units = units

        rp = axes_test_dp.get_radial_profile()
        rp_scale_x = rp.axes_manager.navigation_axes[0].scale
        rp_scale_y = rp.axes_manager.navigation_axes[1].scale
        rp_units_x = rp.axes_manager.navigation_axes[0].units
        rp_name_x = rp.axes_manager.navigation_axes[0].name
        rp_units_y = rp.axes_manager.navigation_axes[1].units
        rp_name_y = rp.axes_manager.navigation_axes[1].name

        assert n_scale == rp_scale_x
        assert 2 * n_scale == rp_scale_y
        assert units == rp_units_x
        assert name == rp_name_x
        assert name == rp_units_y
        assert units == rp_name_y

    @pytest.mark.parametrize('expected', [
        (np.array(
            [[5., 4., 3., 2., 0.],
             [1., 0.5, 0.2, 0.2, 0.]]
        ))])
    @pytest.mark.parametrize('expected_mask', [
        (np.array(
            [[5., 4., 3., 2., 0.],
             [1., 0.5, 0.125, 0.25, 0.]]
        ))])
    def test_radial_profile(self, diffraction_pattern_for_radial, expected,
                            mask_for_radial, expected_mask):
        rp = diffraction_pattern_for_radial.get_radial_profile()
        rp_mask = diffraction_pattern_for_radial.get_radial_profile(
            mask_array=mask_for_radial)
        assert np.allclose(rp.data, expected, atol=1e-3)
        assert np.allclose(rp_mask.data, expected_mask, atol=1e-3)


class TestBackgroundMethods:

    @pytest.mark.parametrize('method, kwargs', [
        ('h-dome', {'h': 1, }),
        ('gaussian_difference', {'sigma_min': 0.5, 'sigma_max': 1, }),
        ('median', {'footprint': 4, }),
        ('reference_pattern', {'bg': np.ones((8, 8)), })
    ])
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_remove_background(self, diffraction2d,
                               method, kwargs):
        bgr = diffraction2d.remove_background(method=method, **kwargs)
        assert bgr.data.shape == diffraction2d.data.shape
        assert bgr.max() <= diffraction2d.max()

    @pytest.mark.xfail(raises=TypeError)
    def test_no_kwarg(self, diffraction2d):
        bgr = diffraction2d.remove_background(method='h-dome')


class TestsAssertionless:

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    # we don't want to use xc in this bit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_find_peaks_interactive(self, diffraction2d):
        from matplotlib import pyplot as plt
        plt.ion()  # to make plotting non-blocking
        diffraction2d.find_peaks_interactive()
        plt.close('all')


@pytest.mark.xfail(raises=NotImplementedError)
class TestNotImplemented():
    def test_failing_run(self, diffraction2d):
        diffraction2d.find_peaks(method='no_such_method_exists')

    def test_remove_dead_pixels_failing(self, diffraction2d):
        dpr = diffraction2d.remove_deadpixels(
            [[1, 2], [5, 6]], 'fake_method', inplace=False, progress_bar=False)

    def test_remove_background_fake_method(self, diffraction2d):
        bgr = diffraction2d.remove_background(method='fake_method')


class TestComputeAndAsLazy2D:

    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == Diffraction2D
        assert not hasattr(s.data, 'compute')
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15),
                                      chunks=(1, 1, 10, 15))
        s = LazyDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == Diffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyDiffraction2D
        assert hasattr(s_lazy.data, 'compute')
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Diffraction2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyDiffraction2D
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, Diffraction2D)
>>>>>>> 56aa0b1780fc6379e6e85e4fc725db34e4b028c8
