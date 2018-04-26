##
# \file utilities.py
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2018
#

import numpy as np


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


def convert_regaladin_to_sitk_transform(path_to_regaladin):

    matrix = np.loadtxt(path_to_regaladin)

    nda_2D = np.array([0, 0, 1, 0], [0, 0, 0 1])
    if np.sum(np.abs(matrix[2:, 0:] - nda_2D)) < 1e-6:
        dim = 2
    else:
        dim = 3

    A = matrix[0:dim, 0:dim]
    t = matrix[0:dim, -1]

    # Convert to SimpleITK physical coordinate system
    R = np.eye(dim)
    R[0, 0] = -1
    R[1, 1] = -1
    A = R.dot(A).dot(R)
    t = R.dot(t)

    transform_sitk = sitk.AffineTransform(A.flatten(), t)

    return transform_sitk

