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
import itertools

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import simplereg.utilities as utils
from simplereg.definitions import DIR_TMP, DIR_DATA, DIR_TEST
from simplereg.niftyreg_to_simpleitk_converter import \
    NiftyRegToSimpleItkConverter as nreg2sitk
from simplereg.nibabel_to_simpleitk_converter import \
    NibabelToSimpleItkConverter as nib2sitk
from simplereg.flirt_to_simpleitk_converter import \
    FlirtToSimpleItkConverter as flirt2sitk
import simplereg.data_reader as dr


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
    #     norm_disp_ref = sitk.TransformToDisplacementField(transform)
    #     transform_disp_ref = sitk.DisplacementFieldTransform(
    #         sitk.Image(norm_disp_ref))

    #     transform_disp = utils.compose_displacement_field_transforms(
    #         transform_disp_outer, transform_disp_inner)

    #     self.assertAlmostEqual(
    #         np.linalg.norm(
    #             np.array(transform_disp.GetParameters()
    #                      ) - transform_disp_ref.GetParameters()),
    #         0, places=self.precision
    #     )

    def test_extract_rigid_from_affine(self):

        ##
        # Build the rotation matrix as NiftyReg does it and account for the
        # coordinate swap in x and y directions by ITK.
        # \date       2018-11-13 12:07:10+0000
        #
        def _get_rotation_matrix(rx, ry, rz):
            # Build combined rotation matrix as NiftyReg does
            Rx = np.eye(3)
            Rx[1, 1] = np.cos(rx)
            Rx[1, 2] = -np.sin(rx)
            Rx[2, 1] = np.sin(rx)
            Rx[2, 2] = np.cos(rx)

            Ry = np.eye(3)
            Ry[0, 0] = np.cos(ry)
            Ry[0, 2] = -np.sin(ry)
            Ry[2, 0] = np.sin(ry)
            Ry[2, 2] = np.cos(ry)

            Rz = np.eye(3)
            Rz[0, 0] = np.cos(rz)
            Rz[0, 1] = -np.sin(rz)
            Rz[1, 0] = np.sin(rz)
            Rz[1, 1] = np.cos(rz)

            R = Rx.dot(Ry).dot(Rz)

            # Basis transformation to account for x -> -x and y -> -y in ITK
            U = np.diag([-1, -1, 1])
            R = U.dot(R).dot(U)

            return R

        np.set_printoptions(precision=3)

        path_to_affine_nreg = os.path.join(DIR_TMP, "affine_nreg.txt")

        rot_max = np.pi

        scalings_min = 0.1
        scalings_max = 10

        shearings_min = -0.1
        shearings_max = 0.1

        translations_min = -10
        translations_max = 10

        n_steps = 10

        rotations = np.linspace(-rot_max, rot_max, n_steps)
        scalings = np.linspace(scalings_min, scalings_max, n_steps)
        shearings = np.linspace(shearings_min, shearings_max, n_steps)
        translations = np.linspace(translations_min, translations_max, n_steps)

        # Overwrite to simplify
        # scalings = [1]
        shearings = [0]
        translations = [0]

        print("Number of test runs: %d" % (len(rotations) *
                                           len(scalings) *
                                           len(shearings) *
                                           len(translations))**3
              )
        print("Might take a while ...")
        for rx, ry, rz in itertools.product(rotations, rotations, rotations):
            for tx, ty, tz in itertools.product(translations, translations, translations):
                for sx, sy, sz in itertools.product(scalings, scalings, scalings):
                    for shx, shy, shz in itertools.product(shearings, shearings, shearings):
                        # print([sx, sy, sz])

                        # ----------- Approximate affine transform -----------
                        # Build affine transform for NiftyReg
                        cmd_args = ["reg_transform"]
                        cmd_args.append("-makeAff")
                        cmd_args.append("%f %f %f" % (rx, ry, rz))
                        cmd_args.append("%f %f %f" % (tx, ty, tz))
                        cmd_args.append("%f %f %f" % (sx, sy, sz))
                        cmd_args.append("%f %f %f" % (shx, shy, shz))
                        cmd_args.append(path_to_affine_nreg)
                        self.assertEqual(
                            ph.execute_command(" ".join(cmd_args), verbose=0),
                            0)

                        # Convert affine to SimpleITK transform
                        affine_nreg = dr.DataReader.read_transform_nreg(
                            path_to_affine_nreg)
                        affine_sitk = \
                            nreg2sitk.convert_regaladin_to_sitk_transform(
                                affine_nreg, dim=3)

                        # Approximate affine by rigid transform
                        approx_euler_sitk = \
                            utils.extract_rigid_from_affine(affine_sitk)
                        m_approx_euler = np.array(
                            approx_euler_sitk.GetMatrix()).reshape(3, 3)

                        # ---- "exact" rigid transform based on rotations ----
                        m_euler = _get_rotation_matrix(rx, ry, rz)

                        # ---------- Compare obtained rigid matrices ----------
                        error = np.max(np.abs(m_approx_euler - m_euler))
                        tol = 1e-5
                        if error > tol:
                            print(error)
                            print("---")
                            print("m_euler = \n%s" % m_euler)
                            print("m_approx_euler = \n%s" % m_approx_euler)
                            print("---")
                            print("shearings: %s" % np.array([shx, shy, shz]))
                            print("scalings:  %s" % np.array([sx, sy, sz]))
                            print("rotations: %s" % np.array([rx, ry, rz]))

                        self.assertAlmostEqual(error, 0, places=5)

    def test_get_displacement_norm_between_images(self):

        shapes = {
            2: (100, 200),
            3: (100, 200, 50),
        }
        origin = {
            2: (-100, 10),
            3: (-100, 10, 33.3),
        }
        direction = {
            2: (0, -1, 1, 0),
            3: (0, 1, 0, 1, 0, 0, 0, 0, -1),
        }
        spacing = {
            2: (1.1, 2.5),
            3: (1.1, 2.5, 5),
        }

        parameters = {
            2: (-1.7, -10.23, 12),
            3: (0.3, -1.3, 2.1, -10.23, 12, 4),
        }
        center = {
            2: (-10, 15),
            3: (-10, 15, 17.3),
        }

        for dim in [2, 3]:
            print("Dimension: %d" % dim)
            transform_sitk = getattr(sitk, "Euler%dDTransform" % dim)()
            transform_sitk.SetParameters(parameters[dim])
            transform_sitk.SetCenter(center[dim])

            image_sitk = sitk.Image(shapes[dim], sitk.sitkFloat32)
            image_sitk.SetOrigin(origin[dim])
            image_sitk.SetDirection(direction[dim])
            image_sitk.SetSpacing(spacing[dim])

            # -------------------Compute mean displacements-------------------
            t0 = ph.start_timing()
            norm_disp = utils.get_voxel_displacements(
                image_sitk, transform_sitk)
            print("Time method: %s" % ph.stop_timing(t0))

            # -----------------------Compute ref result-----------------------
            disp_shape = sitk.GetArrayFromImage(image_sitk).shape
            t0 = ph.start_timing()
            disp = np.zeros(disp_shape + (dim,))
            for index in np.ndindex(shapes[dim]):
                point = image_sitk.TransformIndexToPhysicalPoint(index)
                point_ref = transform_sitk.TransformPoint(point)
                disp[index[::-1]] = point - np.array(point_ref)
            norm_disp_ref = np.sqrt(np.sum(np.square(disp), axis=-1))
            print("Time reference: %s" % ph.stop_timing(t0))

            self.assertAlmostEqual(
                np.linalg.norm(norm_disp - norm_disp_ref), 0,
                places=self.precision)
