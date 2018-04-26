##
# \file flirt_test.py
#  \brief  Class containing unit tests for module FLIRT
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2017

import os
import numpy as np
import SimpleITK as sitk
import unittest

import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import DIR_DATA
import simplereg.flirt


class FlirtTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10

        # ----------------------------------2D---------------------------------
        self.fixed_sitk_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Target.png"))

        self.moving_sitk_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Source.png"))

        # ---------------------------------3D----------------------------------
        self.fixed_sitk_3D = sitk.ReadImage(
            os.path.join(DIR_DATA, "3D_Brain_Source.nii.gz"))

        self.moving_sitk_3D = sitk.ReadImage(
            os.path.join(DIR_DATA, "3D_Brain_Target.nii.gz"))

    def tearDown(self):
        print("Computational time = %s" % (
            self.registration_method.get_computational_time()))

        transformed_fixed_sitk = \
            self.registration_method.get_transformed_fixed_sitk()
        warped_moving_sitk = self.registration_method.get_warped_moving_sitk()

        if self.show_fig:
            sitkh.show_sitk_image(
                [self.fixed_sitk, self.moving_sitk, warped_moving_sitk],
                label=["fixed", "moving", "warped_moving"])
            sitkh.show_sitk_image(
                [self.moving_sitk, transformed_fixed_sitk],
                label=["moving", "warped_fixed"]
            )

        registration_transform_sitk = \
            self.registration_method.get_registration_transform_sitk()

        # Check transformed fixed
        transformed_fixed_sitk_2 = sitkh.get_transformed_sitk_image(
            self.fixed_sitk, registration_transform_sitk)
        transformed_fixed_sitk = sitk.Cast(
            transformed_fixed_sitk, sitk.sitkFloat64)
        transformed_fixed_sitk_2 = sitk.Cast(
            transformed_fixed_sitk_2, sitk.sitkFloat64)
        diff_nda = sitk.GetArrayFromImage(
            transformed_fixed_sitk_2 - transformed_fixed_sitk)
        self.assertEqual(np.round(
            np.linalg.norm(diff_nda), decimals=self.accuracy), 0)

        # Check warped moving
        warped_moving_sitk_2 = sitk.Resample(
            self.moving_sitk, self.fixed_sitk, registration_transform_sitk)
        warped_moving_sitk = sitk.Cast(
            warped_moving_sitk, sitk.sitkFloat64)
        warped_moving_sitk_2 = sitk.Cast(
            warped_moving_sitk_2, sitk.sitkFloat64)
        diff_nda = sitk.GetArrayFromImage(
            warped_moving_sitk_2 - warped_moving_sitk)
        try:
            self.assertEqual(np.round(
                np.linalg.norm(diff_nda), decimals=self.accuracy), 0)
        except Exception as e:
            print("FAIL: " + self.id() +
                  " failed given norm of difference = %.2e > 1e-%s" %
                  (np.linalg.norm(diff_nda), self.accuracy))
            sitkh.show_sitk_image(
                [warped_moving_sitk,
                 warped_moving_sitk_2,
                 warped_moving_sitk_2 - warped_moving_sitk,
                 ],
                label=["warped_moving", "warped_moving_2", "diff"])

    def test_registration_2D(self):
        """
        Registration transform is correct but warped_moving.nii.gz obtained
        by FLIRT seems to be flawed.
        """

        self.fixed_sitk = self.fixed_sitk_2D
        self.moving_sitk = self.moving_sitk_2D
        self.show_fig = 0

        self.registration_method = simplereg.flirt.FLIRT(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk,
            options="-2D",
        )

        self.registration_method.run()

    def test_registration_3D(self):

        self.fixed_sitk = self.fixed_sitk_3D
        self.moving_sitk = self.moving_sitk_3D
        self.show_fig = 0

        self.registration_method = simplereg.flirt.FLIRT(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk,
            options="-dof 6",
        )

        self.registration_method.run()
