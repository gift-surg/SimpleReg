# \file landmark_registration_test.py
# \brief      Class containing unit tests for landmark registration
#             applications
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2018
#

import os
import re
import unittest
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph

import simplereg.application.estimate_landmarks as estimate_landmarks
import simplereg.application.register_landmarks as register_landmarks
from simplereg.definitions import DIR_TMP, DIR_DATA, DIR_TEST


class LandmarkRegistrationTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.verbose = 0

    def test_estimate_landmarks(self):
        for image in ["3D_Brain_Template", "3D_Brain_AD"]:
            path_to_image_fiducials = os.path.join(
                DIR_TEST, "%s_fiducials.nii.gz" % image)
            path_to_reference = os.path.join(
                DIR_TEST, "%s_landmarks.txt" % image)
            path_to_result = os.path.join(DIR_TMP, "%s_result.txt" % image)

            exe = re.sub(".pyc", ".py", os.path.abspath(
                estimate_landmarks.__file__))
            cmd_args = ["python %s" % exe]
            cmd_args.append("--output %s" % path_to_result)
            cmd_args.append("--filename %s" % path_to_image_fiducials)
            # cmd_args.append("--clusters 4")
            cmd_args.append("--verbose %s" % self.verbose)
            cmd = " ".join(cmd_args)
            flag = ph.execute_command(cmd, verbose=self.verbose)
            if flag != 0:
                raise RuntimeError("Cannot execute command '%s'" % cmd)

            result = np.loadtxt(path_to_result)
            reference = np.loadtxt(path_to_reference)

            # Find closest point
            indices = [
                np.argmin(np.linalg.norm(result[i, :] - reference, axis=1))
                for i in range(result.shape[0])]
            reference = [reference[i, :] for i in indices]

            self.assertAlmostEqual(
                np.sum(np.abs(result - reference)), 0, places=self.precision)

    def test_register_landmarks(self):
        path_to_fixed_landmarks = os.path.join(
            DIR_TEST, "3D_Brain_Template_landmarks.txt")
        path_to_moving_landmarks = os.path.join(
            DIR_TEST, "3D_Brain_AD_landmarks.txt")
        path_to_reference = os.path.join(
            DIR_TEST, "landmark_transform_3D_Brain_Source_to_Target_CPD.txt")
        path_to_result = os.path.join(DIR_TMP, "registration_transform.txt")

        exe = re.sub(".pyc", ".py", os.path.abspath(
            register_landmarks.__file__))
        cmd_args = ["python %s" % exe]
        cmd_args.append("--output %s" % path_to_result)
        cmd_args.append("--fixed %s" % path_to_fixed_landmarks)
        cmd_args.append("--moving %s" % path_to_moving_landmarks)
        cmd_args.append("--verbose %s" % self.verbose)
        cmd = " ".join(cmd_args)
        flag = ph.execute_command(cmd, verbose=self.verbose)
        if flag != 0:
            raise RuntimeError("Cannot execute command '%s'" % cmd)

        transform_result_sitk = sitk.Euler3DTransform(
            sitk.ReadTransform(path_to_result))
        transform_reference_sitk = sitk.Euler3DTransform(
            sitk.ReadTransform(path_to_reference))
        result = np.array(transform_result_sitk.GetParameters())
        reference = np.array(transform_reference_sitk.GetParameters())

        self.assertAlmostEqual(
            np.sum(np.abs(result - reference)), 0, places=self.precision)
