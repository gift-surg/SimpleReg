# \file WrapperRegistration.py
# \brief Class to provide basis to wrap command line registration tools
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017

# Import libraries
import os
import sys
import SimpleITK as sitk

# Import modules from src-folder
import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

from abc import ABCMeta, abstractmethod


##
# Abstract class to wrap registration methods from SimpleITK objects
# \date       2017-08-08 17:27:44+0100
#
class WrapperRegistration(object):
    __metaclass__ = ABCMeta

    ##
    # Store information which are considered as basic for all registration
    # tools
    # \date       2017-08-08 17:28:02+0100
    #
    # \param      self              The object
    # \param      fixed_sitk        Fixed image as sitk.Image object
    # \param      moving_sitk       Moving image as sitk.Image object
    # \param      fixed_sitk_mask   Fixed image mask as sitk.Image object
    # \param      moving_sitk_mask  Moving image mask as sitk.Image object
    # \param      options           Options to add for command line tool;
    #                               string
    #
    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask=None,
                 moving_sitk_mask=None,
                 options="",
                 ):

        self._fixed_sitk = fixed_sitk
        self._moving_sitk = moving_sitk
        self._fixed_sitk_mask = fixed_sitk_mask
        self._moving_sitk_mask = moving_sitk_mask
        self._options = options

        self._registration_transform_sitk = None
        self._moving_warped_sitk = None
        self._computational_time = ph.get_zero_time()

    ##
    # Sets the fixed image
    # \date       2017-08-08 16:45:45+0100
    #
    # \param      self        The object
    # \param      fixed_sitk  The fixed image as sitk object
    #
    def set_fixed_sitk(self, fixed_sitk):
        self._fixed_sitk = fixed_sitk

    ##
    # Gets the fixed image
    # \date       2017-08-08 16:45:58+0100
    #
    # \param      self  The object
    #
    # \return     The fixed image as sitk object.
    #
    def get_fixed_sitk(self):
        return self._fixed_sitk

    ##
    # Sets the moving image
    # \date       2017-08-08 16:45:45+0100
    #
    # \param      self        The object
    # \param      moving_sitk  The moving image as sitk object
    #
    def set_moving_sitk(self, moving_sitk):
        self._moving_sitk = moving_sitk

    ##
    # Gets the moving image
    # \date       2017-08-08 16:45:58+0100
    #
    # \param      self  The object
    #
    # \return     The moving image as sitk object.
    #
    def get_moving_sitk(self):
        return self._moving_sitk

    ##
    # Sets the fixed image mask
    # \date       2017-08-08 16:45:45+0100
    #
    # \param      self             The object
    # \param      fixed_sitk_mask  The fixed image as sitk object
    #
    def set_fixed_sitk_mask(self, fixed_sitk_mask):
        self._fixed_sitk_mask = fixed_sitk_mask

    ##
    # Gets the fixed image mask
    # \date       2017-08-08 16:45:58+0100
    #
    # \param      self  The object
    #
    # \return     The fixed image as sitk object.
    #
    def get_fixed_sitk_mask(self):
        return self._fixed_sitk_mask

    ##
    # Sets the moving image mask
    # \date       2017-08-08 16:45:45+0100
    #
    # \param      self             The object
    # \param      moving_sitk_mask  The moving image as sitk object
    #
    def set_moving_sitk_mask(self, moving_sitk_mask):
        self._moving_sitk_mask = moving_sitk_mask

    ##
    # Gets the moving image mask
    # \date       2017-08-08 16:45:58+0100
    #
    # \param      self  The object
    #
    # \return     The moving image as sitk object.
    #
    def get_moving_sitk_mask(self):
        return self._moving_sitk_mask

    ##
    # Sets the options of the registration method.
    # \date       2017-08-08 17:26:45+0100
    #
    # \param      self     The object
    # \param      options  The options as string
    #
    def set_options(self, options):
        self._options = options

    ##
    # Gets the options.
    # \date       2017-08-08 17:27:07+0100
    #
    # \param      self  The object
    #
    # \return     The options as string
    #
    def get_options(self):
        return self._options

    ##
    # Gets the obtained registration transform.
    # \date       2017-08-08 16:52:36+0100
    #
    # \param      self  The object
    #
    # \return     The registration transform as sitk object.
    #
    def get_registration_transform_sitk(self):
        if self._registration_transform_sitk is None:
            raise UnboundLocalError("Execute 'run_registration' first.")

        return self._registration_transform_sitk

    ##
    # Gets the warped moving image, i.e. moving image warped and resampled
    # to the fixed grid
    # \date       2017-08-08 16:58:30+0100
    #
    # \param      self  The object
    #
    # \return     The warped moving image as sitk.Image object
    #
    def get_warped_moving_sitk(self):
        return sitk.Image(self._moving_warped_sitk)

    ##
    # Gets the computational time it took to perform the registration
    # \date       2017-08-08 16:59:45+0100
    #
    # \param      self  The object
    #
    # \return     The computational time.
    #
    def get_computational_time(self):
        return self._computational_time

    ##
    # Run the registration method
    # \date       2017-08-08 17:01:01+0100
    #
    # \param      self  The object
    #
    def run(self):

        if self._fixed_sitk is None:
            raise ValueError("Fixed image must be specified")

        if self._moving_sitk is None:
            raise ValueError("Mobing image must be specified")

        time_start = ph.start_timing()

        # Execute registration method
        self._run()

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)

    @abstractmethod
    def _run(self):
        pass
