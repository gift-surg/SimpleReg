##
# \file utilities_test.py
#  \brief  Class containing unit tests for utility functions
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2018

import os
import numpy as np
import SimpleITK as sitk
import unittest

import pysitk.simple_itk_helper as sitkh

import simplereg.utilities as utils
from simplereg.definitions import DIR_TMP, DIR_DATA, DIR_TEST


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
            transform_sitk = utils.convert_regaladin_to_sitk_transform(
                matrix_regaladin)
            transform_reference_sitk = sitk.AffineTransform(
                sitk.ReadTransform(path_to_sitk_reference_transform))

            nda = np.array(transform_sitk.GetParameters())
            nda_reference = transform_reference_sitk.GetParameters()

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)

    def test_convert_sitk_to_regaladin_transofrm(self):
        for dim in [2, 3]:
            path_to_sitk_transform = os.path.join(
                DIR_TEST, "%dD_sitk_Target_Source.txt" % dim)
            path_to_reference_transform = os.path.join(
                DIR_TEST, "%dD_regaladin_Target_Source.txt" % dim)

            transform_sitk = sitk.AffineTransform(sitk.ReadTransform(
                path_to_sitk_transform))

            nda_reference = np.loadtxt(path_to_reference_transform)
            nda = utils.convert_sitk_to_regaladin_transform(transform_sitk)

            self.assertAlmostEqual(
                np.sum(np.abs(nda - nda_reference)), 0,
                places=self.precision)
