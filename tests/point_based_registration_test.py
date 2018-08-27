##
# \file point_based_registration_test.py
#  \brief  Class containing unit tests for module PointBasedRegistration
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2018

import os
import numpy as np
import SimpleITK as sitk
import unittest

import simplereg.point_based_registration as pbr
import simplereg.utilities as utils


class PointBasedRegistrationTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7

        # moving = R.dot(fixed) + t
        self.moving_points_nda = np.array([
            [1.19833243, 16.35080697, 376.43341308],
            [99.37186499, 104.33932213, 391.74496734],
            [177.26392303, 248.24850015, 231.95039992],
            [66.94425414, 193.58966455, 35.6035551],
            [-9.06603781, 82.50111604, 19.95151651]])
        self.fixed_points_nda = np.array([
            [39.3047, 71.7057, 372.7200],
            [171.8440, 65.8063, 376.3250],
            [312.0440, 77.5614, 196.0000],
            [176.5280, 78.7922, 8.0000],
            [43.4659, 53.2688, 10.5000],
        ])
        self.groundtruth_rotation_nda = np.array([
            [0.707454, -0.70305673, 0.07225005],
            [0.69995393, 0.68281896, -0.20933887],
            [0.0978434, 0.19866933, 0.97517033]])
        self.groundtruth_translation_nda = np.array(
            [-3.1238, 17.9021, -5.1235])

        # # Rotation only, i.e. moving = R.dot(fixed)
        # self.moving_points_nda = np.array([
        #     [4.32213243, -1.55129303, 381.55691308],
        #     [102.49566499, 86.43722213, 396.86846734],
        #     [180.38772303, 230.34640015, 237.07389992],
        #     [70.06805414, 175.68756455, 40.7270551],
        #     [-5.94223781, 64.59901604, 25.07501651]])
        # self.fixed_points_nda = np.array([
        #     [39.3047, 71.7057, 372.7200],
        #     [171.8440, 65.8063, 376.3250],
        #     [312.0440, 77.5614, 196.0000],
        #     [176.5280, 78.7922, 8.0000],
        #     [43.4659, 53.2688, 10.5000],
        # ])
        # self.groundtruth_rotation_nda = np.array([
        #     [0.707454, -0.70305673, 0.07225005],
        #     [0.69995393, 0.68281896, -0.20933887],
        #     [0.0978434, 0.19866933, 0.97517033]])
        # self.groundtruth_translation_nda = np.zeros(3)

    def test_BeslMcKay(self):

        point_based_registration = pbr.BeslMcKayPointBasedRegistration(
            fixed_points_nda=self.fixed_points_nda,
            moving_points_nda=self.moving_points_nda,
        )
        point_based_registration.run()
        R, t = point_based_registration.get_registration_outcome_nda()

        # Compute Fiducial Registration Error
        transformed_fixed = np.transpose(
            R.dot(self.fixed_points_nda.transpose())) + t

        FRE = utils.fiducial_registration_error(
            self.moving_points_nda, transformed_fixed)

        # Test FRE accuracy
        self.assertAlmostEqual(FRE, 0, places=self.precision)

        # Test rotation matrix accuracy
        self.assertAlmostEqual(
            np.sum(np.abs(R - self.groundtruth_rotation_nda)), 0,
            places=self.precision)

        # Test translation matrix accuracy
        self.assertAlmostEqual(
            np.sum(np.abs(t - self.groundtruth_translation_nda)), 0,
            places=self.precision)

        print("Computational time Besl and McKay: %s" %
              point_based_registration.get_computational_time())

    def test_ArunHuangBlostein(self):

        point_based_registration = pbr.ArunHuangBlosteinPointBasedRegistration(
            fixed_points_nda=self.fixed_points_nda,
            moving_points_nda=self.moving_points_nda,
        )
        point_based_registration.run()
        R, t = point_based_registration.get_registration_outcome_nda()

        # Compute Fiducial Registration Error
        transformed_fixed = np.transpose(
            R.dot(self.fixed_points_nda.transpose())) + t

        FRE = utils.fiducial_registration_error(
            self.moving_points_nda, transformed_fixed)

        # Test FRE accuracy
        self.assertAlmostEqual(FRE, 0, places=self.precision)

        # Test rotation matrix accuracy
        self.assertAlmostEqual(
            np.sum(np.abs(R - self.groundtruth_rotation_nda)), 0,
            places=self.precision)

        # Test translation matrix accuracy
        self.assertAlmostEqual(
            np.sum(np.abs(t - self.groundtruth_translation_nda)), 0,
            places=self.precision)

        print("Computational time Arun et al.: %s" %
              point_based_registration.get_computational_time())

    def test_RigidCoherentPointDrift(self):

        for optimize_scaling in [0, 1]:
            point_based_registration = pbr.RigidCoherentPointDrift(
                fixed_points_nda=self.fixed_points_nda,
                moving_points_nda=self.moving_points_nda[0:-1, :],
                optimize_scaling=optimize_scaling,
                verbose=0
            )
            point_based_registration.run()
            R, t = point_based_registration.get_registration_outcome_nda()

            # Compute Fiducial Registration Error
            transformed_fixed = np.transpose(
                R.dot(self.fixed_points_nda.transpose())) + t

            FRE = utils.fiducial_registration_error(
                self.moving_points_nda, transformed_fixed)

            # Test FRE accuracy
            self.assertAlmostEqual(FRE, 0, places=self.precision)

            # Test rotation matrix accuracy
            self.assertAlmostEqual(
                np.sum(np.abs(R - self.groundtruth_rotation_nda)), 0,
                places=self.precision)

            # Test translation matrix accuracy
            self.assertAlmostEqual(
                np.sum(np.abs(t - self.groundtruth_translation_nda)), 0,
                places=self.precision)

            print("Computational time Rigid CPD (optimize_scaling=%d): %s" % (
                  optimize_scaling,
                  point_based_registration.get_computational_time()))

    def test_AffineCoherentPointDrift(self):

        point_based_registration = pbr.AffineCoherentPointDrift(
            fixed_points_nda=self.fixed_points_nda,
            moving_points_nda=self.moving_points_nda,
            verbose=0,
        )
        point_based_registration.run()
        R, t = point_based_registration.get_registration_outcome_nda()

        # Compute Fiducial Registration Error
        transformed_fixed = np.transpose(
            R.dot(self.fixed_points_nda.transpose())) + t

        FRE = utils.fiducial_registration_error(
            self.moving_points_nda, transformed_fixed)

        # Test FRE accuracy
        self.assertAlmostEqual(FRE, 0, places=self.precision)

        # Test rotation matrix accuracy
        self.assertAlmostEqual(
            np.sum(np.abs(R - self.groundtruth_rotation_nda)), 0,
            places=self.precision)

        # Test translation matrix accuracy
        self.assertAlmostEqual(
            np.sum(np.abs(t - self.groundtruth_translation_nda)), 0,
            places=self.precision)

        print("Computational time Affine CPD: %s" %
              point_based_registration.get_computational_time())
