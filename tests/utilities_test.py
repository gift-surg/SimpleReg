##
# \file utilities_test.py
#  \brief  Class containing unit tests for utility functions
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2018

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import unittest

import pysitk.simple_itk_helper as sitkh

import simplereg.utilities as utils
from simplereg.definitions import DIR_TMP, DIR_DATA, DIR_TEST
from simplereg.niftyreg_to_simpleitk_converter import \
    NiftyRegToSimpleItkConverter as nreg2sitk
from simplereg.nibabel_to_simpleitk_converter import \
    NibabelToSimpleItkConverter as nib2sitk
from simplereg.flirt_to_simpleitk_converter import \
    FlirtToSimpleItkConverter as flirt2sitk


class UtilitiesTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7

    def test_convert_regaladin_to_sitk_transform(self):
        for dim in [2, 3]:
            path_to_regaladin_transform = os.path.join(
                DIR_TEST, "%dD_regaladin_Target_Source.txt" % dim)
            path_to_sitk_reference_transform = os.path.join(
                DIR_TEST, "%dD_sitk_Target_Source.txt" % dim)

            matrix_regaladin = np.loadtxt(path_to_regaladin_transform)
            transform_sitk = nreg2sitk.convert_regaladin_to_sitk_transform(
                matrix_regaladin)
            transform_reference_sitk = sitk.AffineTransform(
                sitk.ReadTransform(path_to_sitk_reference_transform))

            nda = np.array(transform_sitk.GetParameters())
            nda_reference = transform_reference_sitk.GetParameters()

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)

    def test_convert_regf3d_to_sitk_transform(self):
        for dim in [3]:
            path_to_regf3d_transform = os.path.join(
                DIR_TEST, "%dD_regf3d_Target_Source_cpp_disp.nii.gz" % dim)
            path_to_sitk_reference_transform = os.path.join(
                DIR_TEST, "%dD_regf3d_Target_Source_cpp_disp_sitk.nii.gz" % dim)

            transform_nreg_nib = nib.load(path_to_regf3d_transform)
            transform_sitk_nib = nreg2sitk.convert_regf3d_to_sitk_displacement(
                transform_nreg_nib)

            transform_reference_nib = nib.load(
                path_to_sitk_reference_transform)

            nda = transform_sitk_nib.get_data()
            nda_reference = transform_reference_nib.get_data()

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)

    def test_convert_sitk_to_regaladin_transform(self):
        for dim in [2, 3]:
            path_to_sitk_transform = os.path.join(
                DIR_TEST, "%dD_sitk_Target_Source.txt" % dim)
            path_to_reference_transform = os.path.join(
                DIR_TEST, "%dD_regaladin_Target_Source.txt" % dim)

            transform_sitk = sitk.AffineTransform(sitk.ReadTransform(
                path_to_sitk_transform))

            nda_reference = np.loadtxt(path_to_reference_transform)
            nda = nreg2sitk.convert_sitk_to_regaladin_transform(transform_sitk)

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)

    def test_convert_sitk_to_regf3d_transform(self):
        for dim in [3]:
            path_to_sitk_reference_transform = os.path.join(
                DIR_TEST, "%dD_regf3d_Target_Source_cpp_disp.nii.gz" % dim)
            path_to_sitk_transform = os.path.join(
                DIR_TEST, "%dD_regf3d_Target_Source_cpp_disp_sitk.nii.gz" % dim)

            transform_sitk_nib = nib.load(path_to_sitk_transform)
            transform_nreg_nib = nreg2sitk.convert_sitk_to_regf3d_displacement(
                transform_sitk_nib)

            transform_reference_nib = nib.load(
                path_to_sitk_reference_transform)

            nda = transform_nreg_nib.get_data()
            nda_reference = transform_reference_nib.get_data()

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)

    def test_convert_sitk_to_flirt_transform(self):
        for dim in [3]:
            path_to_sitk_transform = os.path.join(
                DIR_TEST, "%dD_sitk_Target_Source.txt" % dim)
            path_to_fixed = os.path.join(
                DIR_DATA, "%dD_Brain_Target.nii.gz" % dim)
            path_to_moving = os.path.join(
                DIR_DATA, "%dD_Brain_Source.nii.gz" % dim)
            path_to_res = os.path.join(
                DIR_TMP, "%dD_sitk2flirt_target_Source.txt" % dim)
            path_to_reference_transform = os.path.join(
                DIR_TEST, "%dD_flirt_Target_Source.txt" % dim)

            flirt2sitk.convert_sitk_to_flirt_transform(
                path_to_sitk_transform, path_to_fixed, path_to_moving, path_to_res)
            nda = np.loadtxt(path_to_res)

            nda_reference = np.loadtxt(
                path_to_reference_transform)

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)

    def test_convert_flirt_to_sitk_transform(self):
        for dim in [3]:
            path_to_flirt_transform = os.path.join(
                DIR_TEST, "%dD_flirt_Target_Source.txt" % dim)
            path_to_fixed = os.path.join(
                DIR_DATA, "%dD_Brain_Target.nii.gz" % dim)
            path_to_moving = os.path.join(
                DIR_DATA, "%dD_Brain_Source.nii.gz" % dim)
            path_to_res = os.path.join(
                DIR_TMP, "%dD_flirt2sitk_target_Source_.txt" % dim)
            path_to_reference_transform = os.path.join(
                DIR_TEST, "%dD_sitk_Target_Source.txt" % dim)

            flirt2sitk.convert_flirt_to_sitk_transform(
                path_to_flirt_transform, path_to_fixed, path_to_moving, path_to_res)
            transform_sitk = sitkh.read_transform_sitk(path_to_res)

            transform_ref_sitk = sitkh.read_transform_sitk(
                path_to_reference_transform)

            nda_reference = np.array(transform_ref_sitk.GetParameters())
            nda = np.array(transform_sitk.GetParameters())

            # Conversion to FLIRT only provides 4 decimal places
            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=2)

    def test_convert_sitk_to_nib_image_3D(self):
        path_to_image = os.path.join(
            DIR_DATA, "3D_SheppLoganPhantom_64.nii.gz")

        image_sitk = sitk.ReadImage(path_to_image)

        # Provide non-trivial orientation
        image_sitk.SetDirection((-0.16161674116998426,
                                 0.16786208904173827,
                                 -0.9724722886614946,
                                 0.9868296296827364,
                                 0.03435841234534616,
                                 -0.1580720658080342,
                                 -0.0068782958520129545,
                                 0.9852115603075567,
                                 0.17120417575706345))

        # Provide non-trivial spacing
        image_sitk.SetSpacing((1.3, 2.1, 3.9))

        image_nib = nib2sitk.convert_sitk_to_nib_image(image_sitk)

        path_to_image_sitk = os.path.join(DIR_TMP, "image_sitk.nii.gz")
        path_to_image_nib = os.path.join(DIR_TMP, "image_nib.nii.gz")

        sitk.WriteImage(image_sitk, path_to_image_sitk)
        nib.save(image_nib, path_to_image_nib)

        image1_sitk = sitk.ReadImage(path_to_image_sitk)
        image2_sitk = sitk.ReadImage(path_to_image_nib)

        diff_nda = sitk.GetArrayFromImage(image1_sitk - image2_sitk)
        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0,
            places=self.precision)

    def test_convert_sitk_to_nib_image_displacement(self):
        path_to_image = os.path.join(
            DIR_TEST, "3D_regf3d_Target_Source_cpp_disp.nii.gz")

        image_sitk = sitk.ReadImage(path_to_image)
        image_nib = nib2sitk.convert_sitk_to_nib_image(image_sitk)

        path_to_image_sitk = os.path.join(DIR_TMP, "image_disp_sitk.nii.gz")
        path_to_image_nib = os.path.join(DIR_TMP, "image_disp_nib.nii.gz")

        sitk.WriteImage(image_sitk, path_to_image_sitk)
        nib.save(image_nib, path_to_image_nib)

        image1_sitk = sitk.ReadImage(path_to_image_sitk)
        image2_sitk = sitk.ReadImage(path_to_image_nib)

        diff_nda = sitk.GetArrayFromImage(image1_sitk - image2_sitk)
        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0,
            places=self.precision)

    def test_compose_affine_transforms(self):

        # Composition of Rigid transforms is rigid transform
        self.assertIsInstance(
            utils.compose_affine_transforms(
                sitk.Euler3DTransform(), sitk.Euler3DTransform()),
            sitk.Euler3DTransform)

        # Composition with an affine transform returns affine transform
        self.assertIsInstance(
            utils.compose_affine_transforms(
                sitk.AffineTransform(3), sitk.Euler3DTransform()),
            sitk.AffineTransform)

    # def test_compose_displacement_field_transforms(self):
    #     transform_outer = sitk.Euler3DTransform()
    #     transform_outer.SetRotation(0.3, -0.1, 1.3)
    #     transform_outer.SetTranslation((-10.03, 13.12, -239.2))
    #     transform_outer.SetCenter((1.3, -23.3, 54.1))

    #     disp_outer = sitk.TransformToDisplacementField(transform_outer)
    #     transform_disp_outer = sitk.DisplacementFieldTransform(
    #         sitk.Image(disp_outer))

    #     transform_inner_ = sitk.Euler3DTransform()
    #     transform_inner_.SetRotation(-1.3, 0.31, 1.4)

    #     transform_inner = sitk.AffineTransform(3)
    #     transform_inner.SetMatrix(transform_inner_.GetMatrix())
    #     transform_inner.SetTranslation((-10.1, 30.1, 0.4))
    #     transform_inner.SetCenter((-13., 1.5, 0.7))
    #     disp_inner = sitk.TransformToDisplacementField(transform_inner)
    #     transform_disp_inner = sitk.DisplacementFieldTransform(
    #         sitk.Image(disp_inner))

    #     transform = utils.compose_affine_transforms(
    #         transform_outer, transform_inner)
    #     disp_ref = sitk.TransformToDisplacementField(transform)
    #     transform_disp_ref = sitk.DisplacementFieldTransform(
    #         sitk.Image(disp_ref))

    #     transform_disp = utils.compose_displacement_field_transforms(
    #         transform_disp_outer, transform_disp_inner)

    #     self.assertAlmostEqual(
    #         np.linalg.norm(
    #             np.array(transform_disp.GetParameters()
    #                      ) - transform_disp_ref.GetParameters()),
    #         0, precision=self.precision
    #     )
