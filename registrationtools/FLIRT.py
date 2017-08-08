# \file FLIRT.py
# \brief      This class makes FLIRT accessible via Python
#
# This class requires Convert3D Medical Image Processing Tool to be installed
# (https://sourceforge.net/projects/c3d/files/c3d/Nightly/)
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2017

# Import libraries
import os
import SimpleITK as sitk

import pythonhelper.PythonHelper as ph

from registrationtools.definitions import DIR_TMP
from registrationtools.definitions import FLIRT_EXE
from registrationtools.definitions import C3D_AFFINE_TOOL_EXE

from registrationtools.WrapperRegistration import WrapperRegistration


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
        self._warped_str = os.path.join(self._dir_tmp, "warped.nii.gz")

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

        cmd = FLIRT_EXE + endl
        cmd += "-in " + self._moving_str + endl
        cmd += "-ref " + self._fixed_str + endl
        cmd += "-out " + self._warped_str + endl
        cmd += "-omat " + self._registration_transform_str + endl

        sitk.WriteImage(self._fixed_sitk, self._fixed_str)
        sitk.WriteImage(self._moving_sitk, self._moving_str)

        if self._fixed_sitk_mask is not None:
            sitk.WriteImage(self._fixed_sitk_mask, self._fixed_mask_str)
            cmd += "-refweight " + self._fixed_mask_str + endl

        if self._moving_sitk_mask is not None:
            sitk.WriteImage(self._moving_sitk_mask, self._moving_mask_str)
            cmd += "-inweight " + self._moving_mask_str + endl

        cmd += self._options

        # Execute registration
        ph.execute_command(cmd, verbose=debug)

        # Read warped image
        self._moving_warped_sitk = sitk.ReadImage(self._warped_str)

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

        cmd = C3D_AFFINE_TOOL_EXE + endl
        cmd += "-ref " + self._fixed_str + endl
        cmd += "-src " + self._moving_str + endl
        cmd += self._registration_transform_str + endl
        cmd += "-fsl2ras" + endl
        cmd += "-oitk " + self._registration_transform_sitk_str

        # Execute conversion
        ph.execute_command(cmd, verbose=verbose)

        # Read transform and convert to affine registration
        trafo_sitk = sitk.ReadTransform(self._registration_transform_sitk_str)
        registration_transform_sitk = sitk.AffineTransform(
            self._fixed_sitk.GetDimension())
        registration_transform_sitk.SetParameters(
            trafo_sitk.GetParameters())

        return registration_transform_sitk
