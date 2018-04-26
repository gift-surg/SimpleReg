##
# \file niftyreg_test.py
#  \brief  Class containing unit tests for module NiftyReg
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2017

import os
import numpy as np
import SimpleITK as sitk
import unittest

import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import DIR_DATA
import simplereg.niftyreg


class NiftyRegTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10
        self.show_fig = 0

        # ----------------------------------2D---------------------------------
        self.fixed_sitk_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Target.png"))
        self.fixed_sitk_mask_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Target_mask.png"))

        self.moving_sitk_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Source.png"))
        self.moving_sitk_mask_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Source_mask.png"))

        # ---------------------------------3D----------------------------------
        self.fixed_sitk_3D = sitk.ReadImage(
            os.path.join(DIR_DATA, "3D_Brain_Source.nii.gz"))

        self.moving_sitk_3D = sitk.ReadImage(
            os.path.join(DIR_DATA, "3D_Brain_Target.nii.gz"))

    def tearDown_reg_aladin(self):
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

    def tearDown_reg_f3d(self):
        print("Computational time = %s" % (
            self.registration_method.get_computational_time()))

        warped_moving_sitk = self.registration_method.get_warped_moving_sitk()

        if self.show_fig:
            sitkh.show_sitk_image(
                [self.fixed_sitk, self.moving_sitk, warped_moving_sitk],
                label=["fixed", "moving", "warped_moving"])

    def test_registration_reg_aladin_2D(self):

        self.fixed_sitk = self.fixed_sitk_2D
        self.moving_sitk = self.moving_sitk_2D
        self.show_fig = 0

        self.registration_method = simplereg.niftyreg.RegAladin(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk,
            options="-voff",
        )

        self.registration_method.run()

        self.tearDown_reg_aladin()

    def test_registration_reg_aladin_3D(self):

        self.fixed_sitk = self.fixed_sitk_3D
        self.moving_sitk = self.moving_sitk_3D
        self.show_fig = 0

        self.registration_method = simplereg.niftyreg.RegAladin(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk,
            options="-voff",
        )

        self.registration_method.run()

        self.tearDown_reg_aladin()

    def test_registration_reg_f3d_2D(self):

        self.fixed_sitk = self.fixed_sitk_2D
        self.moving_sitk = self.moving_sitk_2D
        self.fixed_sitk_mask = self.fixed_sitk_mask_2D
        self.moving_sitk_mask = self.moving_sitk_mask_2D
        self.show_fig = 0

        # ------------------------------RegAladin---------------------------
        self.registration_method = simplereg.niftyreg.RegF3D(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk,
            fixed_sitk_mask=self.fixed_sitk_mask,
            moving_sitk_mask=self.moving_sitk_mask,
            options="-voff",
        )

        self.registration_method.run()
        self.registration_method.get_warped_moving_sitk_mask()
        self.tearDown_reg_f3d()

    def test_registration_reg_f3d_3D(self):

        self.fixed_sitk = self.fixed_sitk_3D
        self.moving_sitk = self.moving_sitk_3D
        self.show_fig = 1

        # ------------------------------RegAladin---------------------------
        self.registration_method = simplereg.niftyreg.RegF3D(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk,
            options="-voff",
        )

        self.registration_method.run()
        self.tearDown_reg_f3d()
