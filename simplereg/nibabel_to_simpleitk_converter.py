# \file nibabel_to_simpleitk_converter.py
#  \brief Class to convert between Nibabel and SimpleITK representations
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018


import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Class to convert between Nibabel and SimpleITK representations
# \date       2018-06-10 22:20:41-0600
#
class NibabelToSimpleItkConverter(object):

    @staticmethod
    def convert_sitk_to_nib_image(image_sitk):

        dim = image_sitk.GetDimension()
        D = np.array(image_sitk.GetDirection()).reshape(dim, dim)
        origin = np.array(image_sitk.GetOrigin())

        # Rotation matrix (x maps_to -x, and y maps_to -y)
        R = np.eye(dim)
        R[0, 0] = -1
        R[1, 1] = -1

        # Affine matrix for orientation
        A = np.eye(dim + 1)
        A[0:dim, 0:dim] = R.dot(D)
        A[0:dim, -1] = R.dot(origin)

        # Reshape [z,] y, x to x, y [,z]
        nda = sitk.GetArrayFromImage(image_sitk)
        nda = np.swapaxes(nda, axis1=0, axis2=dim - 1)

        n_components = image_sitk.GetNumberOfComponentsPerPixel()
        if n_components > 1:
            shape = nda.shape
            nda = nda.reshape(shape[0], shape[1], shape[2], 1, shape[3])

        spacing = np.ones(nda.ndim)
        spacing[0:dim] = image_sitk.GetSpacing()

        # Create nibabel image
        image_nib = nib.Nifti1Image(nda, A)
        image_nib.header.set_zooms(spacing)

        if n_components > 1:
            # Vector image
            image_nib.header["intent_code"] = 1007

        return image_nib
