##
# \file application_test.py
#  \brief  Unit tests based on fetal brain case study
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date November 2017


import os
import re
import unittest
import numpy as np
import nibabel as nib
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import simplereg.utilities as utils
import simplereg.data_reader as dr
from simplereg.definitions import DIR_TMP, DIR_TEST, DIR_DATA


class ApplicationTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_output = os.path.join(DIR_TMP, "simplereg-application")

        self.image_2D = os.path.join(DIR_DATA, "2D_Brain_Target.nii.gz")
        self.image_3D = os.path.join(DIR_DATA, "3D_Brain_Target.nii.gz")
        self.image_3D_moving = os.path.join(DIR_DATA, "3D_Brain_Source.nii.gz")

        self.transform_2D_sitk = os.path.join(
            DIR_TEST, "2D_sitk_Target_Source.txt")
        self.transform_3D_sitk = os.path.join(
            DIR_TEST, "3D_sitk_Target_Source.txt")

        self.transform_2D_nreg = os.path.join(
            DIR_TEST, "2D_regaladin_Target_Source.txt")
        self.transform_3D_nreg = os.path.join(
            DIR_TEST, "3D_regaladin_Target_Source.txt")
        self.transform_3D_nreg_disp = os.path.join(
            DIR_TEST, "3D_regf3d_Target_Source_cpp_disp.nii.gz")
        self.transform_3D_sitk_disp = os.path.join(
            DIR_TEST, "3D_regf3d_Target_Source_cpp_disp_sitk.nii.gz")

        self.transform_3D_flirt = os.path.join(
            DIR_TEST, "3D_flirt_Target_Source.txt")

        self.landmarks_3D = os.path.join(
            DIR_TEST, "3D_Brain_Template_landmarks.txt")

        self.output_transform = os.path.join(self.dir_output, "transform.txt")
        self.output_transform_disp = os.path.join(
            self.dir_output, "transform.nii.gz")
        self.output_landmarks = os.path.join(self.dir_output, "landmarks.txt")
        self.output_image = os.path.join(self.dir_output, "image.nii.gz")

    def test_transform_image(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-i %s %s %s" % (
            self.image_3D, self.transform_3D_sitk, self.output_image))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

    def test_transform_invert_transform(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-inv %s %s" % (
            self.transform_3D_sitk, self.output_transform))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        transform_sitk = dr.DataReader.read_transform(
            self.transform_3D_sitk)
        transform_inv_sitk = dr.DataReader.read_transform(
            self.output_transform)
        ref_nda = np.array(transform_sitk.GetInverse().GetParameters())
        res_nda = np.array(transform_inv_sitk.GetParameters())
        self.assertAlmostEqual(
            np.linalg.norm(ref_nda - res_nda), 0, places=self.precision)

    def test_transform_landmarks(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-l %s %s %s" % (
            self.landmarks_3D, self.transform_3D_sitk, self.output_landmarks))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

    def test_transform_sitk_to_regaladin(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-sitk2nreg %s %s" % (
            self.transform_3D_sitk, self.output_transform))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_nda = dr.DataReader.read_transform_nreg(self.output_transform)
        ref_nda = dr.DataReader.read_transform_nreg(self.transform_3D_nreg)
        self.assertAlmostEqual(
            np.linalg.norm(ref_nda - res_nda), 0, places=self.precision)

    def test_transform_sitk_to_regf3d(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-sitk2nreg %s %s" % (
            self.transform_3D_sitk_disp, self.output_transform_disp))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_nib = nib.load(self.output_transform_disp)
        ref_nib = nib.load(self.transform_3D_nreg_disp)

        HEADERS = ['intent_p1']
        for h in HEADERS:
            self.assertEqual(res_nib.header[h], res_nib.header[h])

        diff_nda = res_nib.get_data() - ref_nib.get_data()
        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0, places=self.precision)

    def test_transform_nreg_to_sitk_regaladin(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-nreg2sitk %s %s" % (
            self.transform_3D_nreg, self.output_transform))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_sitk = dr.DataReader.read_transform(self.output_transform)
        ref_sitk = dr.DataReader.read_transform(self.transform_3D_sitk)
        res_nda = np.array(res_sitk.GetParameters())
        ref_nda = np.array(ref_sitk.GetParameters())
        self.assertAlmostEqual(
            np.linalg.norm(ref_nda - res_nda), 0, places=self.precision)

    def test_transform_nreg_to_sitk_regf3d(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-nreg2sitk %s %s" % (
            self.transform_3D_nreg_disp, self.output_transform_disp))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_sitk = sitk.ReadImage(self.output_transform_disp)
        ref_sitk = sitk.ReadImage(self.transform_3D_sitk_disp)
        diff_nda = sitk.GetArrayFromImage(res_sitk - ref_sitk)
        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0, places=self.precision)

    def test_transform_flirt_to_sitk(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-flirt2sitk %s %s %s %s" % (
            self.transform_3D_flirt,
            self.image_3D,
            self.image_3D_moving,
            self.output_transform))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_sitk = dr.DataReader.read_transform(self.output_transform)
        ref_sitk = dr.DataReader.read_transform(self.transform_3D_sitk)
        res_nda = np.array(res_sitk.GetParameters())
        ref_nda = np.array(ref_sitk.GetParameters())
        self.assertAlmostEqual(
            np.linalg.norm(ref_nda - res_nda), 0, places=2)

    def test_transform_sitk_to_flirt(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-sitk2flirt %s %s %s %s" % (
            self.transform_3D_sitk,
            self.image_3D,
            self.image_3D_moving,
            self.output_transform))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_nda = dr.DataReader.read_transform_flirt(self.output_transform)
        ref_nda = dr.DataReader.read_transform_flirt(self.transform_3D_flirt)
        self.assertAlmostEqual(
            np.linalg.norm(ref_nda - res_nda), 0, places=self.precision)

    def test_transform_swap_sitk_nii(self):
        cmd_args = ["python simplereg_transform.py"]
        cmd_args.append("-sitk2nii %s %s" % (
            self.landmarks_3D, self.output_landmarks))
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_nda = dr.DataReader.read_landmarks(self.output_landmarks)
        ref_nda = dr.DataReader.read_landmarks(self.landmarks_3D)
        ref_nda[:, 0:2] *= -1
        self.assertAlmostEqual(
            np.linalg.norm(ref_nda - res_nda), 0, places=self.precision)

    # TODO
    def test_transform_split_labels(self):
        pass

    # TODO
    def test_transform_mask_to_landmark(self):
        pass

    def test_resample_bspline_spacing_atg(self):
        image = os.path.join(DIR_DATA, "3D_SheppLoganPhantom_64.nii.gz")
        reference = os.path.join(
            DIR_TEST, "3D_SheppLoganPhantom_64_BSpline_s113_atg-4.nii.gz")

        cmd_args = ["python simplereg_resample.py"]
        cmd_args.append("-m %s" % image)
        cmd_args.append("-f same")
        cmd_args.append("-i BSpline")
        cmd_args.append("-t %s" % self.transform_3D_sitk)
        cmd_args.append("-s 1 1 3")
        cmd_args.append("-atg -4")
        cmd_args.append("-o %s" % self.output_image)
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_sitk = sitk.ReadImage(self.output_image)
        ref_sitk = sitk.ReadImage(reference)
        diff_nda = sitk.GetArrayFromImage(res_sitk - ref_sitk)
        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0, places=self.precision)

    def test_resample_oriented_gaussian_spacing_atg(self):
        moving = os.path.join(DIR_DATA, "3D_SheppLoganPhantom_64.nii.gz")
        fixed = os.path.join(DIR_TMP, "3D_SheppLoganPhantom_64_rotated.nii.gz")
        rotation = sitk.Euler3DTransform()
        rotation.SetRotation(0.3, -0.2, -0.3)
        rotation.SetCenter((-40, -25, 17))
        image_rotated = utils.update_image_header(
            sitk.ReadImage(moving), rotation)
        sitk.WriteImage(image_rotated, fixed)

        reference = os.path.join(
            DIR_TEST, "3D_SheppLoganPhantom_64_OrientedGaussian_s113_atg4.nii.gz")

        cmd_args = ["python simplereg_resample.py"]
        cmd_args.append("-m %s" % moving)
        cmd_args.append("-f %s" % fixed)
        cmd_args.append("-i OrientedGaussian")
        cmd_args.append("-s 1 1 3")
        cmd_args.append("-atg 4")
        cmd_args.append("-p -1000")
        cmd_args.append("-o %s" % self.output_image)
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        res_sitk = sitk.ReadImage(self.output_image)
        ref_sitk = sitk.ReadImage(reference)
        diff_nda = sitk.GetArrayFromImage(res_sitk - ref_sitk)
        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0, places=self.precision)
