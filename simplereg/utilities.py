##
# \file utilities.py
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2018
#

import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph


##
# Compute fiducial registration error (FRE) between fixed and warped moving
# landmarks
# \date       2018-04-21 22:24:10-0600
#
# \param      reference_nda  Reference landmarks as (N x dim) numpy array where
#                            dim is either 2 or 3
# \param      estimate_nda   Estimate landmarks as (N x dim) numpy array where
#                            dim is either 2 or 3
#
# \return     FRE as scalar value
#
def fiducial_registration_error(reference_nda, estimate_nda):
    if not isinstance(reference_nda, np.ndarray):
        raise IOError("Fixed points must be of type np.array")

    if not isinstance(estimate_nda, np.ndarray):
        raise IOError("Moving points must be of type np.array")

    if reference_nda.shape[1] != 2 and reference_nda.shape[1] != 3:
        raise IOError("Fixed points must be of dimension N x 2 or N x 3")

    if estimate_nda.shape[1] != 2 and estimate_nda.shape[1] != 3:
        raise IOError(
            "Warped moving points must be of dimension N x 2 or N x 3")

    if reference_nda.shape != estimate_nda.shape:
        raise IOError(
            "Dimensions of fixed and warped moving points must be equal")

    N = float(reference_nda.shape[0])
    FRE = np.square(np.sum(np.square(reference_nda - estimate_nda)) / N)

    return FRE

##
# Convert a NiftyReg-RegAladin affine transformation matrix into a SimpleITK
# transform
# \date       2018-04-25 21:40:05-0600
#
# \param      matrix  affine matrix as np.ndarray
#
# \return     Affine transformation of type sitk.AffineTransform
#


def convert_regaladin_to_sitk_transform(matrix):

    if not isinstance(matrix, np.ndarray):
        raise IOError("matrix must be a np.ndarray")

    if matrix.shape != (4, 4):
        raise IOError("matrix array must be of shape (4, 4)")

    if np.sum(np.abs(matrix[-1, :] - np.array([0, 0, 0, 1]))):
        raise IOError("last row of matrix must be [0, 0, 0, 1]")

    # retrieve dimension
    nda_2D = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    if np.sum(np.abs(matrix[2:, 0:] - nda_2D)) < 1e-6:
        dim = 2
    else:
        dim = 3

    # retrieve (dim x dim) matrix and translation
    A = matrix[0:dim, 0:dim]
    t = matrix[0:dim, -1]

    # Convert to SimpleITK physical coordinate system
    R = np.eye(dim)
    R[0, 0] = -1
    R[1, 1] = -1
    A = R.dot(A).dot(R)
    t = R.dot(t)

    # Convert to affine transform
    # (Note, it is not possible to extract a EulerxDTransform even for
    # rigid reg_aladin trafos: 'Error: Attempt to set a Non-Orthogonal matrix')
    transform_sitk = sitk.AffineTransform(A.flatten(), t)

    return transform_sitk


##
# Convert SimpleITK Transform into a NiftyReg-RegAladin affine matrix
# \date       2018-04-25 21:41:08-0600
#
# \param      transform_sitk  transformation as sitk.Transform
#
# \return     NiftyReg-RegAladin transformation as (4 x 4)-np.array
#
def convert_sitk_to_regaladin_transform(transform_sitk):

    if not isinstance(transform_sitk, sitk.Transform):
        raise IOError("Input must be a sitk.Transform")

    dim = transform_sitk.GetDimension()
    A_sitk = np.array(transform_sitk.GetMatrix()).reshape(dim, dim)
    t_sitk = np.array(transform_sitk.GetTranslation())

    # Convert to physical coordinate system
    A = np.eye(4)
    R = np.eye(dim)

    R[0, 0] = -1
    R[1, 1] = -1
    A[0:dim, 0:dim] = R.dot(A_sitk).dot(R)
    A[0:dim, 3] = R.dot(t_sitk)

    return A
