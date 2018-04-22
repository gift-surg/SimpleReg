##
# \file point_based_registration.py
# \brief      Class to perform rigid registration based on least-squares
#             fitting of two 3D point sets
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2017
#


from abc import ABCMeta, abstractmethod
import numpy as np

import pysitk.python_helper as ph


##
# Abstract class for point-based registration to find rotation matrix R and
# translation t that minimize || moving - (R.fixed - t) ||_2^2
# \date       2018-04-21 19:52:08-0600
#
class PointBasedRegistration(object):
    __metaclass__ = ABCMeta

    ##
    # Store information required for point-based registration.
    # \date       2018-04-21 19:58:19-0600
    #
    # \param      self               The object
    # \param      fixed_points_nda   Fixed points as (N x 3) numpy array
    # \param      moving_points_nda  Moving points as (N x 3) numpy array
    # \param      verbose            Verbose output, boolean
    #
    def __init__(self, fixed_points_nda, moving_points_nda, verbose):
        self._fixed_points_nda = fixed_points_nda
        self._moving_points_nda = moving_points_nda
        self._verbose = verbose

        self._computational_time = ph.get_zero_time()

        self._rotation_nda = None
        self._translation_nda = None

    ##
    # Sets the fixed points nda.
    # \date       2018-04-21 19:55:54-0600
    #
    # \param      self              The object
    # \param      fixed_points_nda  Fixed points as (N x 3) numpy array
    #
    def set_fixed_points_nda(self, fixed_points_nda):
        self._fixed_points_nda = fixed_points_nda

    ##
    # Return fixed points
    # \date       2018-04-21 19:57:12-0600
    #
    # \param      self  The object
    #
    # \return     The fixed points nda as (N x 3) numpy array
    #
    def get_fixed_points_nda(self):
        return np.array(self._fixed_points_nda)

    ##
    # Sets the moving points nda.
    # \date       2018-04-21 19:55:54-0600
    #
    # \param      self               The object
    # \param      moving_points_nda  moving points as (N x 3) numpy array
    #
    def set_moving_points_nda(self, moving_points_nda):
        self._moving_points_nda = moving_points_nda

    ##
    # Return moving points
    # \date       2018-04-21 19:57:12-0600
    #
    # \param      self  The object
    #
    # \return     The moving points nda as (N x 3) numpy array
    #
    def get_moving_points_nda(self):
        return np.array(self._moving_points_nda)

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
    # Gets the registration outcome, i.e. the rotation matrix R and translation
    # t that minimize  || moving - (R.fixed - t) ||_2^2
    # \date       2018-04-21 23:41:30-0600
    #
    # \param      self  The object
    #
    # \return     The registration outcome nda.
    #
    def get_registration_outcome_nda(self):
        return self._rotation_nda, self._translation_nda

    ##
    # Run the registration method
    # \date       2017-08-08 17:01:01+0100
    #
    # \param      self  The object
    #
    def run(self):
        if not isinstance(self._fixed_points_nda, np.ndarray):
            raise IOError("Fixed points must be of type np.array")

        if not isinstance(self._moving_points_nda, np.ndarray):
            raise IOError("Moving points must be of type np.array")

        if self._fixed_points_nda.shape[1] != 3:
            raise IOError("Fixed points must be of dimension N x 3")

        if self._moving_points_nda.shape[1] != 3:
            raise IOError("Moving points must be of dimension N x 3")

        if self._fixed_points_nda.shape != self._moving_points_nda.shape:
            raise IOError(
                "Dimensions of fixed and moving points must be equal")

        # Execute registration method
        time_start = ph.start_timing()
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
# Implementation of quaternion-based algorithm for point-based rigid
# registration as described in [Besl and McKay, 1992, Section III.C].
#
# Besl, P. J., & McKay, H. D. (1992). A method for registration of 3-D shapes.
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2),
# 239-256.
# \date       2018-04-21 20:16:41-0600
#
class BeslMcKayPointBasedRegistration(PointBasedRegistration):

    def __init__(self, fixed_points_nda, moving_points_nda, verbose=0):
        PointBasedRegistration.__init__(
            self,
            fixed_points_nda=fixed_points_nda,
            moving_points_nda=moving_points_nda,
            verbose=verbose,
        )

    def _run(self):

        # Compute centroids
        mu_fixed_nda = np.mean(self._fixed_points_nda, axis=0)
        mu_moving_nda = np.mean(self._moving_points_nda, axis=0)

        # Compute cross-variance matrix
        Sigma_fm = np.einsum(
            'ij,ik->jk', self._fixed_points_nda, self._moving_points_nda) \
            / float(self._fixed_points_nda.shape[0]) - \
            np.outer(mu_fixed_nda, mu_moving_nda)

        # Define anti-symmetric matrix and its cyclic components
        A = Sigma_fm - Sigma_fm.transpose()
        Delta = np.array([A[1, 2], A[2, 0], A[0, 1]])

        # Compute symmetric matrix
        trace = np.trace(Sigma_fm)
        Q = np.zeros((4, 4))
        Q[0, 0] = trace
        Q[0, 1:] = Delta
        Q[1:, 0] = Delta
        Q[1:, 1:] = Sigma_fm + Sigma_fm.transpose() - trace * np.eye(3)

        # Get eigenvector associated with maximum eigenvalue of Q
        eigval, eigvec = np.linalg.eig(Q)
        q = eigvec[:, np.argmax(eigval)]

        # Compute optimal rotation matrix
        R = np.zeros((3, 3))
        R[0, 0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        R[1, 1] = q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2
        R[2, 2] = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2
        R[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        R[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
        R[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
        R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        R[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])

        # Compute optimal translation vector
        t = mu_moving_nda - R.dot(mu_fixed_nda)

        self._rotation_nda = R
        self._translation_nda = t


##
# Implementation of SVD-based algorithm for point-based rigid registration as
# described in [Arun et al., 1987].
#
# Arun, K. S., Huang, T. S., & Blostein, S. D. (1987). Least-Squares Fitting of
# Two 3-D Point Sets. IEEE Transactions on Pattern Analysis and Machine
# Intelligence, PAMI-9(5), 698-700.
# \date       2018-04-21 23:52:49-0600
#
class ArunHuangBlosteinPointBasedRegistration(PointBasedRegistration):

    def __init__(self, fixed_points_nda, moving_points_nda, verbose=0):
        PointBasedRegistration.__init__(
            self,
            fixed_points_nda=fixed_points_nda,
            moving_points_nda=moving_points_nda,
            verbose=verbose,
        )

    def _run(self):

        # Compute centroids
        mu_fixed_nda = np.mean(self._fixed_points_nda, axis=0)
        mu_moving_nda = np.mean(self._moving_points_nda, axis=0)

        # Obtain centered point sets:
        fixed_nda = self._fixed_points_nda - mu_fixed_nda
        moving_nda = self._moving_points_nda - mu_moving_nda

        # Compute 3 x 3 matrix from sum of outer product of points
        H = np.einsum('ij,ik->jk', fixed_nda, moving_nda)

        # Compute SVD
        U, D, V_transpose = np.linalg.svd(H)

        # Compute orthogonal matrix X
        X = V_transpose.transpose().dot(U.transpose())

        # Compute rotation matrix based on determinant
        det = np.linalg.det(X)

        # det = 1 (rotation)
        if np.abs(det - 1.) < 1e-6:
            R = X

        # det = -1 (reflection)
        # additional case handling possible, see [Arun et. al, Sect. IV and V]
        else:
            raise RuntimeError("Algorithm has failed")

        # Compute translation
        t = mu_moving_nda - R.dot(mu_fixed_nda)

        self._rotation_nda = R
        self._translation_nda = t
