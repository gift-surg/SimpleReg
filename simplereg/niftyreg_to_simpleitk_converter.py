# \file niftyreg_to_simpleitk_converter.py
#  \brief Class to convert between NiftyReg and SimpleITK representations
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018


import os
import sys
import numpy as np
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

    ##
    # Convert a NiftyReg transformation into a SimpleITK transform
    # \date       2018-04-25 21:40:05-0600
    #
    # \param      matrix  affine matrix as np.ndarray
    #
    # \return     Affine transformation of type sitk.AffineTransform
    #
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
                convert_regaladin_to_sitk_transform(
                    transform_nreg)
            dw.DataWriter.write_transform(
                transform_sitk, path_to_output, verbose)

        elif isinstance(transform_nreg, sitk.Image):
            transform_sitk = NiftyRegToSimpleItkConverter.\
                convert_regf3d_to_sitk_displacement(
                    transform_nreg)
            dw.DataWriter.write_transform(
                transform_sitk, path_to_output, verbose)

        else:
            raise IOError(
                "Given transformation must be either "
                "a np.ndarray (reg_aladin) "
                "or a sitk.Image (reg_f3d displacement)")

        return transform_sitk

    ##
    # Convert SimpleITK Transform into a NiftyReg-RegAladin affine matrix
    # \date       2018-04-25 21:41:08-0600
    #
    # \param      transform_sitk  transformation as sitk.Transform
    #
    # \return     NiftyReg-RegAladin transformation as (4 x 4)-np.array
    #
    @staticmethod
    def convert_sitk_to_nreg_transform(
            path_to_transform_sitk,
            path_to_output,
            verbose=0,
    ):

        transform_sitk = dr.DataReader.read_transform(path_to_transform_sitk)

        if isinstance(transform_sitk, sitk.Transform):
            matrix_nda = NiftyRegToSimpleItkConverter.\
                convert_sitk_to_regaladin_transform(
                    transform_sitk)
            dw.DataWriter.write_transform_nreg(
                matrix_nda, path_to_output, verbose)

        elif isinstance(transform_sitk, sitk.Image):
            transform_nreg = NiftyRegToSimpleItkConverter.\
                convert_regf3d_to_sitk_displacement(transform_sitk)

        else:
            raise IOError("Input must be either a "
                          "sitk.Transform or sitk.Image (displacement)")

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
    def convert_regaladin_to_sitk_transform(matrix_nda):
        if matrix_nda.shape != (4, 4):
            raise IOError("matrix array must be of shape (4, 4)")

        if np.sum(np.abs(matrix_nda[-1, :] - np.array([0, 0, 0, 1]))):
            raise IOError("last row of matrix must be [0, 0, 0, 1]")

        # retrieve dimension
        nda_2D = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        if np.sum(np.abs(matrix_nda[2:, 0:] - nda_2D)) < 1e-6:
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
    def convert_regf3d_to_sitk_displacement(displacement_nreg_sitk):
        nda = sitk.GetArrayFromImage(displacement_nreg_sitk)
        nda[..., 0:2] *= -1

        displacement_sitk = sitk.GetImageFromArray(nda)
        displacement_sitk.CopyInformation(displacement_nreg_sitk)

        return displacement_sitk

    @staticmethod
    def get_niftyreg_jacobian_determinant_from_displacement_sitk(
            displacement_sitk):

        path_to_disp = os.path.join(DIR_TMP, "disp.nii.gz")
        path_to_jac = os.path.join(DIR_TMP, "disp_jac.nii.gz")

        displacement_nib = convert_sitk_to_nib_image(displacement_sitk)
        displacement_nib.header['intent_p1'] = 1
        nib.save(displacement_nib, path_to_disp)

        # sitkh.write_nifti_image_sitk(displacement_sitk, path_to_disp)
        # cmd_args = ["fslmodhd"]
        # cmd_args.append(path_to_disp)
        # cmd_args.append("intent_p1 1")
        # ph.execute_command(" ".join(cmd_args))

        cmd_args = ["reg_jacobian"]
        cmd_args.append("-trans %s" % path_to_disp)
        cmd_args.append("-jac %s" % path_to_jac)
        ph.execute_command(" ".join(cmd_args))

        det_jac_sitk = sitk.ReadImage(path_to_jac)
        return det_jac_sitk
