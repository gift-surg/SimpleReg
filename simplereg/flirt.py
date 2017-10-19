# \file FLIRT.py
# \brief      This class makes FLIRT accessible via Python
#
# This class requires Convert3D Medical Image Processing Tool to be installed
# (https://sourceforge.net/projects/c3d/files/c3d/Nightly/)
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2017

# Import libraries
import os
import numpy as np
import SimpleITK as sitk
import nipype.interfaces.fsl
import nipype.interfaces.c3

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import DIR_TMP
from simplereg.wrapper_registration import WrapperRegistration


class FLIRT(WrapperRegistration):

    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask=None,
                 moving_sitk_mask=None,
                 options="",
                 subfolder="FLIRT"):

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

        self._registration_transform_str = os.path.join(
            self._dir_tmp, "registration_transform.txt")
        self._registration_transform_sitk_str = os.path.join(
            self._dir_tmp, "registration_transform_sitk.txt")

    def _run(self, debug=0, endl=" \\\n"):

        # Create and delete all possibly existing files in the directory
        ph.create_directory(self._dir_tmp, delete_files=True)

        sitk.WriteImage(self._fixed_sitk, self._fixed_str)
        sitk.WriteImage(self._moving_sitk, self._moving_str)

        flt = nipype.interfaces.fsl.FLIRT()
        flt.inputs.in_file = self._moving_str
        flt.inputs.reference = self._fixed_str
        flt.inputs.out_file = self._warped_moving_str
        flt.inputs.out_matrix_file = self._registration_transform_str

        if self._fixed_sitk_mask is not None:
            sitk.WriteImage(self._fixed_sitk_mask, self._fixed_mask_str)
            flt.inputs.ref_weight = self._fixed_mask_str

        if self._moving_sitk_mask is not None:
            sitk.WriteImage(self._moving_sitk_mask, self._moving_mask_str)
            flt.inputs.in_weight = self._moving_mask_str

        flt.inputs.args = self._options

        # Execute registration
        if debug:
            print(flt.cmdline)
        flt.run()

        # Read warped image
        self._warped_moving_sitk = sitk.ReadImage(self._warped_moving_str)

        # Convert to sitk affine transform
        self._registration_transform_sitk = self._convert_to_sitk_transform(
            verbose=debug, endl=endl)

    ##
    # Convert FSL to ITK transform and return it as sitk object
    #
    # \see        https://sourceforge.net/p/advants/discussion/840261/thread/5f5e054f/
    # \date       2017-08-08 18:35:40+0100
    #
    # \param      self     The object
    # \param      verbose  The verbose
    # \param      endl     The endl
    #
    # \return     Registration transform as sitk object
    #
    def _convert_to_sitk_transform(self, verbose, endl):

        c3d = nipype.interfaces.c3.C3dAffineTool()
        c3d.inputs.reference_file = self._fixed_str
        c3d.inputs.source_file = self._moving_str
        c3d.inputs.transform_file = self._registration_transform_str
        c3d.inputs.fsl2ras = True
        c3d.inputs.itk_transform = self._registration_transform_sitk_str

        # Execute conversion
        if verbose:
            print(c3d.cmdline)
        c3d.run()

        # Read transform and convert to affine registration
        trafo_sitk = sitk.ReadTransform(self._registration_transform_sitk_str)

        parameters = trafo_sitk.GetParameters()

        if self._fixed_sitk.GetDimension() == 2:
            parameters_ = np.zeros(6)
            parameters_[0:2] = parameters[0:2]
            parameters_[2:4] = parameters[3:5]
            parameters_[4:6] = parameters[10:12]
            parameters = parameters_

        registration_transform_sitk = sitk.AffineTransform(
            self._fixed_sitk.GetDimension())
        registration_transform_sitk.SetParameters(parameters)

        return registration_transform_sitk

    def _get_transformed_fixed_sitk(self):
        return sitkh.get_transformed_sitk_image(
            self._fixed_sitk, self.get_registration_transform_sitk())

    def _get_transformed_fixed_sitk_mask(self):
        return sitkh.get_transformed_sitk_image(
            self._fixed_sitk_mask, self.get_registration_transform_sitk())

    def _get_warped_moving_sitk(self):
        if self._warped_moving_sitk.GetDimension() == 2:
            raise Warning(
                "warped_moving_sitk seems to be flawed for 2D "
                "(see unit tests). "
                "Better resample moving image by using obtained registration "
                "transform registration_transform_sitk instead.")
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
