# \file niftyreg_to_simpleitk_converter.py
#  \brief Class to convert between NiftyReg and SimpleITK representations
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

import simplereg.data_reader as dr
import simplereg.data_writer as dw


##
# Class to convert between NiftyReg and SimpleITK representations
# \date       2018-06-10 22:20:41-0600
#
class NiftyRegToSimpleItkConverter(object):

    @staticmethod
    def convert_nreg_to_sitk_transform(
            path_to_transform_nreg,
            path_to_output,
            verbose=0
    ):

        transform_nreg = dr.DataReader.read_transform_nreg(
            path_to_transform_nreg)

        if isinstance(transform_nreg, np.ndarray):
            transform_sitk = NiftyRegToSimpleItkConverter.\
                convert_regaladin_to_sitk_transform(transform_nreg)
            dw.DataWriter.write_transform(
                transform_sitk, path_to_output, verbose)

        else:
            transform_sitk = NiftyRegToSimpleItkConverter.\
                convert_regf3d_to_sitk_displacement(transform_nreg)
            dw.DataWriter.write_transform(
                transform_sitk, path_to_output, verbose)

        return transform_sitk

    @staticmethod
    def convert_sitk_to_nreg_transform(
            path_to_transform_sitk,
            path_to_output,
            verbose=0,
    ):

        transform_sitk = dr.DataReader.read_transform(
            path_to_transform_sitk, nii_as_nib=1)

        if isinstance(transform_sitk, sitk.Transform):
            matrix_nda = NiftyRegToSimpleItkConverter.\
                convert_sitk_to_regaladin_transform(
                    transform_sitk)
            dw.DataWriter.write_transform(matrix_nda, path_to_output, verbose)

        else:
            transform_nreg = NiftyRegToSimpleItkConverter.\
                convert_sitk_to_regf3d_displacement(transform_sitk)
            dw.DataWriter.write_transform(
                transform_nreg, path_to_output, verbose)

    @staticmethod
    def convert_sitk_to_regaladin_transform(transform_sitk):
        dim = transform_sitk.GetDimension()
        A_sitk = np.array(transform_sitk.GetMatrix()).reshape(dim, dim)
        t_sitk = np.array(transform_sitk.GetTranslation())

        # Convert to physical coordinate system
        matrix_nda = np.eye(4)
        R = np.eye(dim)

        R[0, 0] = -1
        R[1, 1] = -1
        matrix_nda[0:dim, 0:dim] = R.dot(A_sitk).dot(R)
        matrix_nda[0:dim, 3] = R.dot(t_sitk)

        return matrix_nda

    @staticmethod
    def convert_regaladin_to_sitk_transform(matrix_nda, dim=None):
        if matrix_nda.shape != (4, 4):
            raise IOError("matrix array must be of shape (4, 4)")

        if np.sum(np.abs(matrix_nda[-1, :] - np.array([0, 0, 0, 1]))):
            raise IOError("last row of matrix must be [0, 0, 0, 1]")

        # retrieve dimension if not given
        if dim is None:
            nda_2D = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
            if np.sum(np.abs(matrix_nda[2:, 0:] - nda_2D)) < 1e-12:
                dim = 2
            else:
                dim = 3

        # retrieve (dim x dim) matrix and translation
        A = matrix_nda[0:dim, 0:dim]
        t = matrix_nda[0:dim, -1]

        # Convert to SimpleITK physical coordinate system
        R = np.eye(dim)
        R[0, 0] = -1
        R[1, 1] = -1
        A = R.dot(A).dot(R)
        t = R.dot(t)

        # Convert to affine transform
        # (Note, it is not possible to extract a EulerxDTransform even for
        # rigid reg_aladin trafos: 'Error: Attempt to set a Non-Orthogonal matrix';
        # QR decomposition could work, but initial tests showed that np.linalg.qr
        # and scipy.linalg.qr (can) return a matrix R with negative diagonal
        # values)
        transform_sitk = sitk.AffineTransform(A.flatten(), t)

        return transform_sitk

    @staticmethod
    def convert_regf3d_to_sitk_displacement(displacement_nreg_nib):

        # Account for x maps_to -x and y maps_to -y in ITK
        nda = displacement_nreg_nib.get_data()
        nda[..., 0:2] *= -1

        displacement_sitk_nib = nib.Nifti1Image(
            nda, displacement_nreg_nib.affine, displacement_nreg_nib.header)

        return displacement_sitk_nib

    ##
    # Convert a (Simple)ITK displacement field to NiftyReg (RegF3D) one
    # \date       2018-06-18 09:12:04-0600
    #
    # RegF3D transformation types NIfTI header encoding:
    # *- Cubic B-Spline grid: intent_p1 = 5, intent_p2 = 6, intent_p3 = 0
    # *- Displacement field: intent_p1 = 1, intent_p2 = 0, intent_p3 = 0
    # *- Deformation field: intent_p1 = 0, intent_p2 = 0, intent_p3 = 0
    #
    # \param      displacement_sitk_nib  Displacement field image as
    #                                    nib.Nifti1Image object
    #
    # \return     Displacement field as nib.Nifti1Image object
    #
    @staticmethod
    def convert_sitk_to_regf3d_displacement(displacement_sitk_nib):

        # Account for x maps_to -x and y maps_to -y in ITK
        nda = displacement_sitk_nib.get_data()
        nda[..., 0:2] *= -1

        displacement_nreg_nib = nib.Nifti1Image(
            nda, displacement_sitk_nib.affine, displacement_sitk_nib.header)

        # Update NIfTI header to indicate a displacement field
        displacement_nreg_nib.header['intent_p1'] = 1
        displacement_nreg_nib.header['intent_p2'] = 0
        displacement_nreg_nib.header['intent_p3'] = 0

        return displacement_nreg_nib
