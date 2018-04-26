##
# \file wrap_itk_registration_test.py
#  \brief  Class containing unit tests for module SimpleItkRegistration
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2017

import os
import numpy as np
import itk
import SimpleITK as sitk
import unittest

import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import DIR_DATA
import simplereg.wrap_itk_registration as itkreg


class WrapItkRegistrationTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10

        # ----------------------------------2D---------------------------------
        self.fixed_sitk_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Target.png"), sitk.sitkFloat64)
        self.moving_sitk_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Source.png"), sitk.sitkFloat64)

        self.fixed_sitk_mask_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Target_mask.png"), sitk.sitkUInt8)
        self.moving_sitk_mask_2D = sitk.ReadImage(
            os.path.join(DIR_DATA, "2D_Brain_Source_mask.png"), sitk.sitkUInt8)

        self.fixed_itk_2D = sitkh.get_itk_from_sitk_image(self.fixed_sitk_2D)
        self.moving_itk_2D = sitkh.get_itk_from_sitk_image(self.moving_sitk_2D)
        self.fixed_itk_mask_2D = sitkh.get_itk_from_sitk_image(
            self.fixed_sitk_mask_2D)
        self.moving_itk_mask_2D = sitkh.get_itk_from_sitk_image(
            self.moving_sitk_mask_2D)

        # ---------------------------------3D----------------------------------
        self.fixed_sitk_3D = sitk.ReadImage(
            os.path.join(DIR_DATA, "3D_Brain_Source.nii.gz"))

        self.moving_sitk_3D = sitk.ReadImage(
            os.path.join(DIR_DATA, "3D_Brain_Target.nii.gz"))

        self.fixed_itk_3D = sitkh.get_itk_from_sitk_image(self.fixed_sitk_3D)
        self.moving_itk_3D = sitkh.get_itk_from_sitk_image(self.moving_sitk_3D)

    def tearDown(self):
        print("Computational time = %s" % (
            self.registration_method.get_computational_time()))

        registration_transform_sitk = \
            self.registration_method.get_registration_transform_sitk()

        transformed_fixed_sitk = sitkh.get_transformed_sitk_image(
            self.fixed_sitk, registration_transform_sitk)

        if self.show_fig:
            sitkh.show_sitk_image(
                [self.moving_sitk, self.fixed_sitk, transformed_fixed_sitk],
                label=["moving_itk", "fixed_itk", "warped_fixed_itk"]
            )

        # # Check transformed fixed
        # transformed_fixed_sitk_2 = sitkh.get_transformed_sitk_image(
        #     self.fixed_sitk, registration_transform_sitk)
        # transformed_fixed_sitk = sitk.Cast(
        #     transformed_fixed_sitk, sitk.sitkFloat64)
        # transformed_fixed_sitk_2 = sitk.Cast(
        #     transformed_fixed_sitk_2, sitk.sitkFloat64)
        # diff_nda = sitk.GetArrayFromImage(
        #     transformed_fixed_sitk_2 - transformed_fixed_sitk)
        # self.assertEqual(np.round(
        #     np.linalg.norm(diff_nda), decimals=self.accuracy), 0)

        # # Check warped moving
        # warped_moving_sitk_2 = sitk.Resample(
        #     self.moving_sitk, self.fixed_sitk, registration_transform_sitk)
        # warped_moving_sitk = sitk.Cast(
        #     warped_moving_sitk, sitk.sitkFloat64)
        # warped_moving_sitk_2 = sitk.Cast(
        #     warped_moving_sitk_2, sitk.sitkFloat64)
        # diff_nda = sitk.GetArrayFromImage(
        #     warped_moving_sitk_2 - warped_moving_sitk)
        # try:
        #     self.assertEqual(np.round(
        #         np.linalg.norm(diff_nda), decimals=self.accuracy), 0)
        # except Exception as e:
        #     print("FAIL: " + self.id() +
        #           " failed given norm of difference = %.2e > 1e-%s" %
        #           (np.linalg.norm(diff_nda), self.accuracy))
        #     sitkh.show_sitk_image(
        #         [warped_moving_sitk,
        #          warped_moving_sitk_2,
        #          warped_moving_sitk_2 - warped_moving_sitk,
        #          ],
        #         label=["warped_moving", "warped_moving_2", "diff"])

    def test_registration_2D(self):
        self.fixed_sitk = self.fixed_sitk_2D
        self.moving_sitk = self.moving_sitk_2D
        self.fixed_sitk_mask = self.fixed_sitk_mask_2D
        self.moving_sitk_mask = self.moving_sitk_mask_2D
        self.fixed_itk = self.fixed_itk_2D
        self.moving_itk = self.moving_itk_2D
        self.fixed_itk_mask = self.fixed_itk_mask_2D
        self.moving_itk_mask = self.moving_itk_mask_2D
        self.show_fig = 0

        self.registration_method = itkreg.WrapItkRegistration(
            fixed_itk=self.fixed_itk,
            moving_itk=self.moving_itk,
            # fixed_itk_mask=self.fixed_itk_mask,
            # moving_itk_mask=self.moving_itk_mask,
            # initializer_type="MOMENTS",
            initializer_type="GEOMETRY",
            dimension=2,
            verbose=1,
        )
        self.registration_method.run()

    # def test_registration_2D_itk_oriented_gaussian(self):
    #     self.fixed_sitk = self.fixed_sitk_2D
    #     self.moving_sitk = self.moving_sitk_2D
    #     self.fixed_sitk_mask = self.fixed_sitk_mask_2D
    #     self.moving_sitk_mask = self.moving_sitk_mask_2D
    #     self.fixed_itk = self.fixed_itk_2D
    #     self.moving_itk = self.moving_itk_2D
    #     self.fixed_itk_mask = self.fixed_itk_mask_2D
    #     self.moving_itk_mask = self.moving_itk_mask_2D
    #     self.show_fig = 0

    #     itk_oriented_gaussian_interpolator = \
    #         itk.OrientedGaussianInterpolateImageFunction[
    #             itk.Image.D2, itk.D].New()

    #     self.registration_method = itkreg.WrapItkRegistration(
    #         fixed_itk=self.fixed_itk,
    #         moving_itk=self.moving_itk,
    #         fixed_itk_mask=self.fixed_itk_mask,
    #         moving_itk_mask=self.moving_itk_mask,
    #         # metric="MeanSquares",
    #         # metric="MattesMutualInformation",
    #         # initializer_type="MOMENTS",
    #         initializer_type="GEOMETRY",
    #         # registration_type="Similarity",
    #         # optimizer="QuasiNewton",
    #         # use_multiresolution_framework=True,
    #         # optimizer_scales="PhysicalShift",
    #         itk_oriented_gaussian_interpolate_image_filter=itk_oriented_gaussian_interpolator,
    #         dimension=2,
    #         verbose=1,
    #     )
    #     self.registration_method.run()

    # def test_registration_3D(self):

    #     self.fixed_sitk = self.fixed_sitk_3D
    #     self.moving_sitk = self.moving_sitk_3D
    #     self.show_fig = 0

    #     self.registration_method = sitkreg.SimpleItkRegistration(
    #         fixed_sitk=self.fixed_sitk,
    #         moving_sitk=self.moving_sitk,
    #         options="-dof 6 -v",
    #     )

    #     self.registration_method.run()
