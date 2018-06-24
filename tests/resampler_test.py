##
# \file resampler_test.py
#  \brief  Class containing unit tests for resampler class
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import unittest

import pysitk.simple_itk_helper as sitkh

import simplereg.utilities as utils
import simplereg.resampler as res
from simplereg.definitions import DIR_TMP, DIR_DATA, DIR_TEST


class ResamplerTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7

    def test_get_resampling_space_properties(self):
        for dim in [2, 3]:
            path_to_image = os.path.join(
                DIR_DATA, "%dD_Brain_Target.nii.gz" % dim)

            image_sitk = sitk.ReadImage(path_to_image)

            # image size is (181, 217). Division by 2 problematic otherwise
            if dim == 2:
                image_sitk = image_sitk[0:180, 0:200]

            spacing_in = np.array(image_sitk.GetSpacing())
            size_in = np.array(image_sitk.GetSize()).astype("int")
            origin = np.array(image_sitk.GetOrigin())
            direction = image_sitk.GetDirection()

            factor_spacing = 13.
            factor_add_to_grid = -3.5

            spacing = factor_spacing * spacing_in
            add_to_grid = factor_add_to_grid * spacing
            size = np.round(size_in / factor_spacing + 2 * factor_add_to_grid)

            size_out, origin_out, spacing_out, direction_out = \
                res.Resampler.get_space_resampling_properties(
                    image_sitk, spacing, add_to_grid)

            if dim == 3:
                a_x = image_sitk.TransformIndexToPhysicalPoint(
                    (1, 0, 0)) - origin
                a_y = image_sitk.TransformIndexToPhysicalPoint(
                    (0, 1, 0)) - origin
                a_z = image_sitk.TransformIndexToPhysicalPoint(
                    (0, 0, 1)) - origin
                e_x = a_x / np.linalg.norm(a_x)
                e_y = a_y / np.linalg.norm(a_y)
                e_z = a_z / np.linalg.norm(a_z)
                offset = (e_x + e_y + e_z) * add_to_grid
                origin -= offset
            else:
                a_x = image_sitk.TransformIndexToPhysicalPoint((1, 0)) - origin
                a_y = image_sitk.TransformIndexToPhysicalPoint((0, 1)) - origin
                e_x = a_x / np.linalg.norm(a_x)
                e_y = a_y / np.linalg.norm(a_y)
                offset = (e_x + e_y) * add_to_grid
                origin -= offset

            self.assertEqual(np.sum(np.abs(spacing_out - spacing)), 0)
            self.assertEqual(np.sum(np.abs(size_out - size)), 0)
            self.assertEqual(np.sum(np.abs(direction_out - direction)), 0)
            self.assertEqual(np.sum(np.abs(origin_out - origin)), 0)

            # check whether extension/cropping does not change 'pixel position'
            resampled_image_sitk = sitk.Resample(
                image_sitk,
                size_out,
                getattr(sitk, "Euler%dDTransform" % dim)(),
                sitk.sitkNearestNeighbor,
                origin_out,
                spacing_out,
                direction_out,
                0,
                image_sitk.GetPixelIDValue()
            )
            image_sitk = sitk.Resample(
                image_sitk,
                resampled_image_sitk,
                getattr(sitk, "Euler%dDTransform" % dim)(),
                sitk.sitkNearestNeighbor,
            )
            nda_diff = sitk.GetArrayFromImage(
                image_sitk - resampled_image_sitk)
            self.assertEqual(np.sum(np.abs(nda_diff)), 0)
