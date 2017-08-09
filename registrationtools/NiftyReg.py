# \file NiftyReg.py
# \brief      This class makes NiftyReg accessible via Python
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017

# Import libraries
import os
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

from registrationtools.definitions import DIR_TMP
from registrationtools.definitions import REG_ALADIN_EXE
from registrationtools.definitions import REG_F3D_EXE
from registrationtools.definitions import REG_RESAMPLE_EXE
from registrationtools.definitions import REG_TRANSFORM_EXE

from registrationtools.WrapperRegistration import WrapperRegistration


class NiftyReg(WrapperRegistration):
    __metaclass__ = ABCMeta

    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask,
                 moving_sitk_mask,
                 options,
                 subfolder):

        WrapperRegistration.__init__(self,
                                     fixed_sitk=fixed_sitk,
                                     moving_sitk=moving_sitk,
                                     fixed_sitk_mask=fixed_sitk_mask,
                                     moving_sitk_mask=moving_sitk_mask,
                                     options=options,
                                     )

        # Subfolder within DIR_TMP where results will be stored temporarily
        self._dir_tmp = os.path.join(DIR_TMP, subfolder)

        self._fixed_str = os.path.join(self._dir_tmp, "fixed.nii.gz")
        self._moving_str = os.path.join(self._dir_tmp, "moving.nii.gz")
        self._warped_moving_str = os.path.join(
            self._dir_tmp, "warped_moving.nii.gz")

        self._fixed_mask_str = os.path.join(
            self._dir_tmp, "fixed_mask.nii.gz")
        self._moving_mask_str = os.path.join(
            self._dir_tmp, "moving_mask.nii.gz")
        self._warped_moving_mask_str = os.path.join(
            self._dir_tmp, "warped_mask.nii.gz")

    def _run(self):

        # Create and delete all possibly existing files in the directory
        ph.create_directory(self._dir_tmp, delete_files=True)

        sitk.WriteImage(self._fixed_sitk, self._fixed_str)
        sitk.WriteImage(self._moving_sitk, self._moving_str)

        if self._fixed_sitk_mask is not None:
            sitk.WriteImage(self._fixed_sitk_mask, self._fixed_mask_str)

        if self._moving_sitk_mask is not None:
            sitk.WriteImage(self._moving_sitk_mask, self._moving_mask_str)


class RegAladin(NiftyReg):

    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask=None,
                 moving_sitk_mask=None,
                 options="",
                 subfolder="RegAladin"):

        NiftyReg.__init__(self,
                          fixed_sitk=fixed_sitk,
                          moving_sitk=moving_sitk,
                          fixed_sitk_mask=fixed_sitk_mask,
                          moving_sitk_mask=moving_sitk_mask,
                          options=options,
                          subfolder=subfolder,
                          )

        self._registration_transform_str = os.path.join(
            self._dir_tmp, "registration_transform.txt")

    def _run(self, debug=0, endl=" \\\n"):

        super(RegAladin, self)._run()

        cmd = REG_ALADIN_EXE + endl
        cmd += "-ref " + self._fixed_str + endl
        cmd += "-flo " + self._moving_str + endl

        if self._fixed_sitk_mask is not None:
            cmd += "-rmask " + self._fixed_mask_str + endl

        if self._moving_sitk_mask is not None:
            cmd += "-fmask " + self._moving_mask_str + endl

        cmd += "-res " + self._warped_moving_str + endl
        cmd += "-aff " + self._registration_transform_str + endl

        cmd += self._options

        # Execute registration
        ph.execute_command(cmd, verbose=debug)

        # Read warped image
        self._warped_moving_sitk = sitk.ReadImage(self._warped_moving_str)

        # Convert to sitk affine transform
        self._registration_transform_sitk = self._convert_to_sitk_transform()

    ##
    # Convert RegAladin transform to sitk object
    #
    # Note, not tested for 2D
    # \date       2017-08-08 18:41:40+0100
    #
    # \param      self     The object
    #
    # \return     Registration transform as sitk object
    #
    def _convert_to_sitk_transform(self):

        dimension = self._fixed_sitk.GetDimension()

        # Read trafo and invert such that format fits within SimpleITK
        # structure
        matrix = np.loadtxt(self._registration_transform_str)
        A = matrix[0:dimension, 0:dimension]
        t = matrix[0:dimension, -1]

        # Convert to SimpleITK physical coordinate system
        R = np.eye(dimension)
        R[0, 0] = -1
        R[1, 1] = -1
        A = R.dot(A).dot(R)
        t = R.dot(t)

        registration_transform_sitk = sitk.AffineTransform(A.flatten(), t)

        return registration_transform_sitk

    def _get_transformed_fixed_sitk(self):
        return sitkh.get_transformed_sitk_image(
            self._fixed_sitk, self.get_registration_transform_sitk())

    def _get_transformed_fixed_sitk_mask(self):
        return sitkh.get_transformed_sitk_image(
            self._fixed_sitk_mask, self.get_registration_transform_sitk())

    def _get_warped_moving_sitk(self):
        return self._warped_moving_sitk

    def _get_warped_moving_sitk_mask(self):

        warped_moving_sitk_mask = sitk.Resample(
            self._moving_sitk_mask,
            self._fixed_sitk,
            self.get_registration_transform_sitk(),
            sitk.sitkNearestNeighbor,
            0,
            self._moving_sitk_mask.GetPixelIDValue(),
        )

        return warped_moving_sitk_mask


class RegF3D(NiftyReg):

    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask=None,
                 moving_sitk_mask=None,
                 options="",
                 subfolder="RegF3D"):

        NiftyReg.__init__(self,
                          fixed_sitk=fixed_sitk,
                          moving_sitk=moving_sitk,
                          fixed_sitk_mask=fixed_sitk_mask,
                          moving_sitk_mask=moving_sitk_mask,
                          options=options,
                          subfolder=subfolder,
                          )

        self._registration_control_point_grid_str = os.path.join(
            self._dir_tmp, "registration_cpp.nii.gz")

    def _run(self, debug=0, endl=" \\\n"):

        super(RegF3D, self)._run()

        cmd = REG_F3D_EXE + endl
        cmd += "-ref " + self._fixed_str + endl
        cmd += "-flo " + self._moving_str + endl

        if self._fixed_sitk_mask is not None:
            cmd += "-rmask " + self._fixed_mask_str + endl

        if self._moving_sitk_mask is not None:
            cmd += "-fmask " + self._moving_mask_str + endl

        cmd += "-res " + self._warped_moving_str + endl
        cmd += "-cpp " + self._registration_control_point_grid_str + endl

        cmd += self._options

        # Execute registration
        ph.execute_command(cmd, verbose=debug)

        # Read warped image
        self._warped_moving_sitk = sitk.ReadImage(self._warped_moving_str)

        # Has not been used. Thus, not tested!
        self._registration_transform_sitk = sitk.ReadImage(
            self._registration_control_point_grid_str)

    def _get_transformed_fixed_sitk(self):
        raise UnboundLocalError("Not implemented for RegF3D")
        # registration_transform_inv_sitk = self._get_inverted_transform(
        #     self._registration_transform_sitk, input_moving_sitk)
        # return self._get_transformed_image_sitk()

    def _get_transformed_fixed_sitk_mask(self):
        raise UnboundLocalError("Not implemented for RegF3D")

    def _get_warped_moving_sitk(self):
        return self._warped_moving_sitk

    def _get_warped_moving_sitk_mask(self, debug=0, endl=" \\\n"):

        warped_moving_sitk_mask = self.get_deformed_image_sitk(
            fixed_sitk=self._fixed_sitk,
            moving_sitk=self._moving_sitk_mask,
            interpolation_order=0)

        return warped_moving_sitk_mask

    ##
    # Gets the deformed image given the obtained deformable registration
    # transform.
    # \date       2017-08-09 16:57:39+0100
    #
    # \param      self                 The object
    # \param      fixed_sitk           Fixed image as sitk.Image
    # \param      moving_sitk          Moving image as sitk.Image
    # \param      interpolation_order  Interpolation order, integer
    # \param      debug                The debug

    # \param      endl                 The endl
    #
    # \return     The deformed image sitk.
    #
    def get_deformed_image_sitk(self, fixed_sitk, moving_sitk,
                                interpolation_order, debug=0, endl=" \\\n"):

        # Create and delete all possibly existing files in the directory
        ph.create_directory(self._dir_tmp, delete_files=True)

        sitk.WriteImage(fixed_sitk, self._fixed_str)
        sitk.WriteImage(moving_sitk, self._moving_str)
        sitk.WriteImage(self.get_registration_transform_sitk(),
                        self._registration_control_point_grid_str)

        cmd = REG_RESAMPLE_EXE + endl
        cmd += "-ref " + self._fixed_str + endl
        cmd += "-flo " + self._moving_str + endl
        cmd += "-trans " + self._registration_control_point_grid_str + endl
        cmd += "-res " + self._warped_moving_str + endl
        cmd += "-inter " + str(interpolation_order)

        # Execute registration
        ph.execute_command(cmd, verbose=debug)

        return sitk.ReadImage(self._warped_moving_str)

    # def _get_inverted_transform(self,
    #                             input_def_field_sitk,
    #                             input_moving_sitk,
    #                             debug=0,
    #                             endl=" \\\n"):

    #     filename_1 = os.path.join(self._dir_tmp, "filename_1.nii.gz")
    #     filename_2 = os.path.join(self._dir_tmp, "filename_2.nii.gz")
    #     filename_3 = os.path.join(self._dir_tmp, "filename_3.nii.gz")

    #     sitk.WriteImage(input_def_field_sitk, filename_1)
    #     sitk.WriteImage(input_moving_sitk, filename_2)

    #     cmd = REG_TRANSFORM_EXE + " -invNrr" + endl
    #     cmd += filename_1 + endl
    #     cmd += filename_2 + endl
    #     cmd += filename_3 + endl

    #     # Execute registration
    #     ph.execute_command(cmd, verbose=debug)

    #     return sitk.ReadImage(filename_3)

    # def _get_transformed_image_sitk(self,
    #                                moving_sitk,
    #                                fixed_sitk,
    #                                deformation_field,
    #                                debug=0,
    #                                endl=" \\\n"):
