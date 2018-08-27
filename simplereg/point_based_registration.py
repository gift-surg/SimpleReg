##
# \file point_based_registration.py
# \brief      Class to perform rigid/affine registration of two 3D point sets
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2018
#


import itertools
import numpy as np
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph


##
# Abstract class for point-based registration to find rigid/affine matrix A and
# translation t that achieve moving ~ A.fixed + t
# \date       2018-04-21 19:52:08-0600
#
class PointBasedRegistration(object):
    __metaclass__ = ABCMeta

    ##
    # Store information required for point-based registration.
    # \date       2018-04-21 19:58:19-0600
    #
    # \param      self               The object
    # \param      fixed_points_nda   Fixed points as (N x dim) numpy array
    # \param      moving_points_nda  Moving points as (N x dim) numpy array
    # \param      verbose            Verbose output, boolean
    #
    def __init__(self, fixed_points_nda, moving_points_nda, verbose):
        self._fixed_points_nda = fixed_points_nda
        self._moving_points_nda = moving_points_nda
        self._verbose = verbose

        self._computational_time = ph.get_zero_time()

        self._matrix_nda = None
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
    # Gets the registration outcome, i.e. the matrix A and translation
    # t that  achieve moving ~ A.fixed + t
    # \date       2018-04-21 23:41:30-0600
    #
    # \param      self  The object
    #
    # \return     The registration outcome nda.
    #
    def get_registration_outcome_nda(self):
        return self._matrix_nda, self._translation_nda

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

        if self._fixed_points_nda.shape[1] != self._moving_points_nda.shape[1]:
            raise IOError(
                "Spatial dimensions of fixed and moving points must be equal")

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

    def _print_registration_estimate(self):
        ph.print_info("Transformation matrix:")
        print(self._matrix_nda)

        ph.print_info("Translation vector:")
        print(self._translation_nda)


##
# Implementation of quaternion-based algorithm for point-based rigid
# registration as described in Besl and McKay (1992), Section III.C.
#
# Algorithm computes the 3-D rigid body transformation that aligns two sets of
# points for which correspondence is known.
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

        if self._fixed_points_nda.shape[1] != 3:
            raise IOError("Fixed/Moving points must be of dimension N x 3")

        if self._fixed_points_nda.shape[0] != self._moving_points_nda.shape[0]:
            raise IOError(
                "Number of fixed and moving points must be equal")

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

        self._matrix_nda = R
        self._translation_nda = t

        if self._verbose:
            self._print_registration_estimate()


##
# Implementation of SVD-based algorithm for point-based rigid registration as
# described in Arun et al. (1987).
#
# Algorithm computes the 3-D rigid body transformation that aligns two sets of
# points for which correspondence is known.
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

        if self._fixed_points_nda.shape[1] != 3:
            raise IOError("Fixed/Moving points must be of dimension N x 3")

        if self._fixed_points_nda.shape[0] != self._moving_points_nda.shape[0]:
            raise IOError(
                "Number of fixed and moving points must be equal")

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
        V = V_transpose.transpose()

        # Compute orthogonal matrix X
        X = V.dot(U.transpose())

        # Compute rotation matrix based on determinant in {-1, 1}
        det = np.linalg.det(X)

        # det = 1 (rotation)
        if det > 0:
            R = X

        # det = -1 (reflection). Error handling [Arun et. al, Sect. IV and V]
        else:
            ph.print_info(
                "Algorithm has encountered reflection. Try to fix it ... ")
            eigval, eigvec = np.linalg.eig(H)

            i_zeros = 0
            for i in range(eigval.size):
                if eigval[i] < 0:
                    V[:, i] *= -1
                    i_zeros += 1
            if i_zeros == 0:
                raise RuntimeError("No negative eigenvalue. Algorithm fails.")
            else:
                R = V.dot(U.transpose())
                if np.linalg.det(R) < 0:
                    raise RuntimeError("Still a reflection. Algorithm fails.")

        # Compute translation
        t = mu_moving_nda - R.dot(mu_fixed_nda)

        self._matrix_nda = R
        self._translation_nda = t

        if self._verbose:
            self._print_registration_estimate()


##
# Implementation of Coherent Point Drift algorithm for point set registration
# as described in Myronenko et al. (2010).
#
# Algorithm computes the 3-D rigid body transformation that aligns two sets of
# points for which correspondence is not known.
#
# Myronenko, A., & Xubo Song. (2010). Point Set Registration: Coherent Point
# Drift. IEEE Transactions on Pattern Analysis and Machine Intelligence,
# 32(12), 2262-2275.
# \date       2018-04-28 16:11:13-0600
#
class CoherentPointDrift(PointBasedRegistration):
    __metaclass__ = ABCMeta

    ##
    # { constructor_description }
    # \date       2018-04-28 16:27:20-0600
    #
    # \param      self               The object
    # \param      fixed_points_nda   Set of fixed points as (M x dim) data
    #                                array
    # \param      moving_points_nda  Set of fixed points as (N x dim) data
    #                                array
    # \param      weight             Weight of uniform distribution, scalar in
    #                                [0, 1]
    # \param      iterations         The iterations
    # \param      verbose            The verbose
    #
    def __init__(self,
                 fixed_points_nda,
                 moving_points_nda,
                 weight,
                 iterations,
                 verbose,
                 tolerance,
                 ):
        PointBasedRegistration.__init__(
            self,
            fixed_points_nda=fixed_points_nda,
            moving_points_nda=moving_points_nda,
            verbose=verbose,
        )

        self._weight = float(weight)
        self._iterations = iterations
        self._tolerance = tolerance

    ##
    # Gets the initial isotropic covariance value sigma2.
    # \date       2018-04-28 20:31:24-0600
    #
    # \param      self  The object
    #
    # \return     Initial estimate for isotropic covariance value.
    #
    def _get_initial_sigma2(self):
        X = self._moving_points_nda
        Y = self._fixed_points_nda

        N = X.shape[0]
        M = Y.shape[0]
        dim = X.shape[1]

        sigma2 = 0.
        for m, n in itertools.product(range(M), range(N)):
            sigma2 += np.sum(np.square(X[n, :] - Y[m, :]))
        sigma2 /= float(dim * N * M)

        return sigma2

    ##
    # Gets the posterior probabilities.
    # \date       2018-04-28 20:33:45-0600
    #
    # \param      self         The object
    # \param      matrix       Transformation matrix
    # \param      translation  Translation
    # \param      sigma2       Isotropic covariance value
    #
    # \return     The posterior probabilities.
    #
    def _get_posterior_probabilities(self, matrix, translation, sigma2):
        X = self._moving_points_nda
        Y = self._fixed_points_nda
        w = self._weight

        dim = X.shape[1]
        N = X.shape[0]
        M = Y.shape[0]

        P = np.zeros((M, N))
        for m, n in itertools.product(range(M), range(N)):
            num = np.exp(- 0.5 * np.sum(np.square(
                X[n, :] - matrix.dot(Y[m, :]) - translation)) / sigma2)
            denom1 = np.sum([
                np.exp(- 0.5 * np.sum(np.square(
                    X[n, :] - matrix.dot(Y[k, :]) - translation)) / sigma2)
                for k in range(M)])
            denom2 = w / (1. - w) * M / float(N) * np.power(
                2. * np.pi * sigma2, dim / 2.)
            P[m, n] = num / float(denom1 + denom2)

        return P

    ##
    # Gets the mean vectors of the point sets
    # \date       2018-04-28 20:34:23-0600
    #
    # \param      self        The object
    # \param      posteriors  Posterior probabilities
    #
    # \return     The mean vectors mean_x, mean_y
    #
    def _get_mean_vectors(self, posteriors):

        N_p = np.sum(posteriors)

        X = self._moving_points_nda
        Y = self._fixed_points_nda

        mu_x = np.sum(X.transpose().dot(posteriors.transpose()), axis=1) / N_p
        mu_y = np.sum(Y.transpose().dot(posteriors), axis=1) / N_p

        return mu_x, mu_y, N_p

    ##
    # Gets the centered point set matrices.
    # \date       2018-04-28 20:35:51-0600
    #
    # \param      self    The object
    # \param      mean_x  Mean vector of moving point set
    # \param      mean_y  Mean vector of fixed point set
    #
    # \return     The centered point set matrices.
    #
    def _get_centered_point_set_matrices(self, mean_x, mean_y):
        X = self._moving_points_nda
        Y = self._fixed_points_nda

        return X - mean_x, Y - mean_y

    ##
    # Update isotropic covariance value
    # \date       2018-04-28 20:36:37-0600
    #
    # \param      N_pD    Product of N_p times spatial dimension D
    # \param      X_hat   Centered moving point set matrix
    # \param      P       Posterior probabilities
    # \param      matrix  Temp matrix
    #
    # \return     Updated isotropic covariance value
    #
    def _update_sigma2(self, N_pD, X_hat, P, matrix):
        term1 = np.trace(X_hat.transpose().dot(
            np.diag(np.sum(P, axis=0))).dot(X_hat))
        term2 = np.trace(matrix)
        sigma2 = (term1 - term2) / float(N_pD)

        # ensure positivity; not stated in CPD-paper. However,
        # frequently encountered negative sigma2 otherwise.
        sigma2 = np.max([2 * self._tolerance, np.abs(sigma2)])

        return sigma2

    ##
    # Determines if converged.
    # \date       2018-04-28 20:38:10-0600
    #
    # \param      self         The object
    # \param      matrix       Transformation matrix
    # \param      translation  Translation matrix
    # \param      sigma2       Isotropic covariance value
    # \param      iteration    The iteration
    #
    # \return     True if converged, False otherwise.
    #
    def _is_converged(self, matrix, translation, sigma2, iteration):
        criterias = [
            np.linalg.norm(self._translation_nda - translation) +
            np.linalg.norm(self._matrix_nda - matrix) < self._tolerance,
            iteration > self._iterations - 1,
            # sigma2 < self._tolerance,
        ]

        if True in criterias:
            if self._verbose:
                if criterias[0]:
                    ph.print_info(
                        "Tolerance (%.g) after %d iterations reached" % (
                            self._tolerance, iteration))
                if criterias[1]:
                    ph.print_info(
                        "Maximum number of iterations (%d) reached" %
                        self._iterations)
                if criterias[-1]:
                    ph.print_info(
                        "Zero isotropic covariance encountered "
                        "after %d iterations" % iteration)
            return True
        else:
            return False


##
# Implementation of rigid (+ scaling) point set registration algorithm, see
# Myronenko et al. (2010), Fig. 2
# \date       2018-04-28 19:49:46-0600
#
class RigidCoherentPointDrift(CoherentPointDrift):

    ##
    # Store information for rigid  (+scaling) Coherent Point Drift (CPD)
    # \date       2018-04-28 20:25:21-0600
    #
    # \param      self               The object
    # \param      fixed_points_nda   Fixed points as (N x dim) numpy array
    # \param      moving_points_nda  Moving points as (N x dim) numpy array
    # \param      iterations         Number of maximum iterations for algorithm
    # \param      weight             Weight of uniform distribution, in [0, 1]
    # \param      scaling            Scaling factor, scalar value > 0
    # \param      optimize_scaling   Turn on/off optimization for scaling
    #                                factor, bool
    # \param      tolerance          Tolerance for convergence
    # \param      verbose            Verbose output, bool
    #
    def __init__(self,
                 fixed_points_nda,
                 moving_points_nda,
                 iterations=100,
                 weight=0.5,
                 scaling=1,
                 optimize_scaling=False,
                 tolerance=1e-12,
                 verbose=1,
                 ):

        CoherentPointDrift.__init__(
            self,
            fixed_points_nda=fixed_points_nda,
            moving_points_nda=moving_points_nda,
            iterations=iterations,
            weight=weight,
            verbose=verbose,
            tolerance=tolerance,
        )
        self._scaling = float(scaling)
        self._optimize_scaling = bool(optimize_scaling)

        self._update_scaling = {
            True: self._update_scaling_true,
            False: self._update_scaling_false,
        }

    ##
    # Estimate scaling factor in case optimization desired
    # \date       2018-04-28 20:30:05-0600
    #
    @staticmethod
    def _update_scaling_true(A, R, Y_hat, P):
        num = np.trace(A.transpose().dot(R))
        denom = np.trace(Y_hat.transpose().dot(
            np.diag(np.sum(P, axis=1))).dot(Y_hat))
        return num / float(denom)

    ##
    # Return initial scaling value if no optimization desired
    # \date       2018-04-28 20:30:49-0600
    #
    def _update_scaling_false(self, A, R, Y_hat, P):
        return self._scaling

    def _run(self):

        # Get initial isotropic covariance value
        sigma2 = self._get_initial_sigma2()

        dim = self._fixed_points_nda.shape[1]
        R = np.eye(dim)

        t = np.zeros(dim)
        s = self._scaling

        self._matrix_nda = s * R
        self._translation_nda = t

        not_converged = True
        iteration = 0

        # EM-optimization
        while not_converged:

            # E-step
            P = self._get_posterior_probabilities(s * R, t, sigma2)

            # M-step
            mean_x, mean_y, N_p = self._get_mean_vectors(P)
            X_hat, Y_hat = self._get_centered_point_set_matrices(
                mean_x, mean_y)
            A = X_hat.transpose().dot(P.transpose()).dot(Y_hat)
            U, S2, V_transpose = np.linalg.svd(A)
            c = np.ones(dim)
            c[-1] = np.linalg.det(U.dot(V_transpose))
            C = np.diag(c)
            R = U.dot(C).dot(V_transpose)
            s = self._update_scaling[self._optimize_scaling](A, R, Y_hat, P)
            t = mean_x - s * R.dot(mean_y)
            sigma2 = self._update_sigma2(
                N_p * dim, X_hat, P, s * A.transpose().dot(R))

            # Check for convergence
            not_converged = not self._is_converged(
                matrix=s * R,
                translation=t,
                sigma2=sigma2,
                iteration=iteration)

            self._matrix_nda = s * R
            self._translation_nda = t
            iteration += 1

        if self._verbose:
            self._print_registration_estimate()


##
# Implementation of affine point set registration algorithm, see Myronenko et
# al. (2010), Fig. 3
# \date       2018-04-28 19:49:46-0600
#
class AffineCoherentPointDrift(CoherentPointDrift):

    ##
    # Store information for affine Coherent Point Drift (CPD)
    # \date       2018-04-28 20:25:21-0600
    #
    # \param      self               The object
    # \param      fixed_points_nda   Fixed points as (N x dim) numpy array
    # \param      moving_points_nda  Moving points as (N x dim) numpy array
    # \param      iterations         Number of maximum iterations for algorithm
    # \param      weight             Weight of uniform distribution, in [0, 1]
    # \param      tolerance          Tolerance for convergence
    # \param      verbose            Verbose output, bool
    #
    def __init__(self,
                 fixed_points_nda,
                 moving_points_nda,
                 iterations=100,
                 weight=0.5,
                 tolerance=1e-8,
                 verbose=1,
                 ):

        CoherentPointDrift.__init__(
            self,
            fixed_points_nda=fixed_points_nda,
            moving_points_nda=moving_points_nda,
            iterations=iterations,
            weight=weight,
            verbose=verbose,
            tolerance=tolerance,
        )

    def _run(self):

        # Get initial isotropic covariance value
        sigma2 = self._get_initial_sigma2()

        dim = self._fixed_points_nda.shape[1]
        B = np.eye(dim)
        t = np.zeros(dim)

        self._matrix_nda = B
        self._translation_nda = t

        not_converged = True
        iteration = 0

        # EM-optimization
        while not_converged:

            # E-step
            P = self._get_posterior_probabilities(B, t, sigma2)

            # M-step
            mean_x, mean_y, N_p = self._get_mean_vectors(P)
            X_hat, Y_hat = self._get_centered_point_set_matrices(
                mean_x, mean_y)
            B = X_hat.transpose().dot(P.transpose()).dot(Y_hat).dot(
                np.linalg.inv(Y_hat.transpose().dot(
                    np.diag(np.sum(P, axis=1))).dot(Y_hat)))
            t = mean_x - B.dot(mean_y)
            sigma2 = self._update_sigma2(
                N_p * dim, X_hat, P,
                X_hat.transpose().dot(P.transpose()).dot(
                    Y_hat).dot(B.transpose()))

            # Check for convergence
            not_converged = not self._is_converged(
                matrix=B,
                translation=t,
                sigma2=sigma2,
                iteration=iteration)

            self._matrix_nda = B
            self._translation_nda = t
            iteration += 1

        if self._verbose:
            self._print_registration_estimate()
