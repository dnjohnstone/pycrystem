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
"""Signal class for Electron Diffraction radial profiles

"""
from hyperspy._signals.lazy import LazySignal

from pyxem.signals import push_metadata_through
from pyxem.signals.diffraction1d import Diffraction1D


class XrayDiffraction1D(Diffraction1D):
    _signal_type = "xray_diffraction1d"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        # Set default attributes
        if 'Acquisition_instrument' in self.metadata.as_dictionary():
            if 'SEM' in self.metadata.as_dictionary()['Acquisition_instrument']:
                self.metadata.set_item(
                    "Acquisition_instrument.I14",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
            if 'REM' in self.metadata.as_dictionary()['Acquisition_instrument']:
                self.metadata.set_item(
                    "Acquisition_instrument.I14",
                    self.metadata.Acquisition_instrument.TEM)
                del self.metadata.Acquisition_instrument.TEM
        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def set_experimental_parameters(self,
                                    beam_energy=None,
                                    exposure_time=None):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        beam_energy : float
            Beam energy in kV
        camera_length: float
            Camera length in cm
        exposure_time : float
            Exposure time in ms.
        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.I14.beam_energy",
                        accelerating_voltage)
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.I14.Detector.Diffraction.camera_length",
                camera_length)
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.I14.Detector.Diffraction.exposure_time",
                exposure_time)

    def set_diffraction_calibration(self, calibration):
        """Set diffraction profile channel size in reciprocal Angstroms.

        Parameters
        ----------
        calibration : float
            Diffraction profile calibration in reciprocal Angstroms per pixel.
        """
        pass

    def set_scan_calibration(self, calibration):
        """Set scan pixel size in nanometres.

        Parameters
        ----------
        calibration: float
            Scan calibration in nanometres per pixel.
        """
        x = self.axes_manager.navigation_axes[0]
        y = self.axes_manager.navigation_axes[1]

        x.name = 'x'
        x.scale = calibration
        x.units = 'nm'

        y.name = 'y'
        y.scale = calibration
        y.units = 'nm'

    def as_lazy(self, *args, **kwargs):
        """Create a copy of the XrayDiffraction1D object as a
        :py:class:`~pyxem.signals.xray_diffraction1d.LazyXrayDiffraction1D`.

        Parameters
        ----------
        copy_variance : bool
            If True variance from the original XrayDiffraction1D object is
            copied to the new LazyXrayDiffraction1D object.

        Returns
        -------
        res : :py:class:`~pyxem.signals.xray_diffraction1d.LazyXrayDiffraction1D`.
            The lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyXrayDiffraction1D
        res.__init__(**res._to_dictionary())
        return res

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = XrayDiffraction1D


class LazyXrayDiffraction1D(LazySignal, XrayDiffraction1D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)
        self.__class__ = XrayDiffraction1D
        self.__init__(**self._to_dictionary())

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyXrayDiffraction1D