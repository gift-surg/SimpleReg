# \file SimpleItkRegistrationBase.py
# \brief Base class for registration methods
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017

# Import libraries
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from abc import ABCMeta, abstractmethod


##
# Abstract class for registration methods
# \date       2017-08-08 17:27:44+0100
#
class SimpleItkRegistrationBase(object):
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
    # \param      options           Options for registration method
    #
    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask,
                 moving_sitk_mask,
                 ):

        self._fixed_sitk = fixed_sitk
        self._moving_sitk = moving_sitk
        self._fixed_sitk_mask = fixed_sitk_mask
        self._moving_sitk_mask = moving_sitk_mask

        self._registration_transform_sitk = None
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
    # Gets the obtained registration transform.
    # \date       2017-08-08 16:52:36+0100
    #
    # \param      self  The object
    #
    # \return     The registration transform as sitk object.
    #
    def get_registration_transform_sitk(self):
        if self._registration_transform_sitk is None:
            raise UnboundLocalError("Execute 'run' first.")

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
        return self._get_warped_moving_sitk()

    ##
    # Gets the warped moving image mask, i.e. moving image mask warped and
    # resampled to fixed grid
    # \date       2017-08-09 11:13:01+0100
    #
    # \param      self  The object
    #
    # \return     The warped moving image mask as sitk.Image object
    #
    def get_warped_moving_sitk_mask(self):
        if self._moving_sitk_mask is None:
            raise ValueError("Moving mask is not provided.")

        return self._get_warped_moving_sitk_mask()

    ##
    # Gets the fixed image transformed by the obtained registration transform.
    #
    # The returned image will align the fixed image with the moving image as
    # found during the registration.
    # \date       2017-08-08 16:53:21+0100
    #
    # \param      self  The object
    #
    # \return     The transformed fixed as sitk.Image object
    #
    def get_transformed_fixed_sitk(self):
        return self._get_transformed_fixed_sitk()

    ##
    # Gets the fixed image mask transformed by the obtained registration
    # transform.
    #
    # The returned image will align the fixed image mask with the moving image
    # as found during the registration.
    # \date       2017-08-08 16:53:21+0100
    #
    # \param      self  The object
    #
    # \return     The transformed fixed mask as sitk.Image object
    #
    def get_transformed_fixed_sitk_mask(self):
        if self._fixed_sitk_mask is None:
            raise ValueError("Fixed mask is not provided.")

        return self._get_transformed_fixed_sitk_mask()

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

        if not isinstance(self._fixed_sitk, sitk.Image):
            raise ValueError("Fixed image must be of type SimpleITK.Image")

        if not isinstance(self._moving_sitk, sitk.Image):
            raise ValueError("Moving image must be of type SimpleITK.Image")

        if self._fixed_sitk_mask is not None and \
                not isinstance(self._fixed_sitk_mask, sitk.Image):
            raise ValueError(
                "Fixed image mask must be of type SimpleITK.Image")

        if self._moving_sitk_mask is not None and \
                not isinstance(self._moving_sitk_mask, sitk.Image):
            raise ValueError(
                "Moving image mask must be of type SimpleITK.Image")

        time_start = ph.start_timing()

        # Execute registration method
        self._run()

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)

    ##
    # Execute registration method
    # \date       2017-08-09 12:08:38+0100
    #
    # \param      self  The object
    #
    @abstractmethod
    def _run(self):
        pass

    ##
    # Gets the warped moving image.
    # \date       2017-08-09 12:08:50+0100
    #
    # \param      self  The object
    #
    # \return     The warped moving image as sitk.Image object.
    #
    @abstractmethod
    def _get_warped_moving_sitk(self):
        pass

    ##
    # Gets the warped moving image mask.
    # \date       2017-08-09 12:08:50+0100
    #
    # \param      self  The object
    #
    # \return     The warped moving image mask as sitk.Image object.
    #
    @abstractmethod
    def _get_warped_moving_sitk_mask(self):
        pass

    ##
    # Gets the transformed fixed image.
    # \date       2017-08-09 12:08:50+0100
    #
    # \param      self  The object
    #
    # \return     The transformed fixed mask as sitk.Image.
    #
    @abstractmethod
    def _get_transformed_fixed_sitk(self):
        pass

    ##
    # Gets the transformed fixed mask.
    # \date       2017-08-09 12:08:50+0100
    #
    # \param      self  The object
    #
    # \return     The transformed fixed mask as sitk.Image.
    #
    @abstractmethod
    def _get_transformed_fixed_sitk_mask(self):
        pass
