##
# \file installation_test.py
#  \brief  Class to test installation
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2017


# Import libraries
import SimpleITK as sitk
import unittest

from nipype.testing import example_data

import simplereg.flirt
import simplereg.niftyreg


class InstallationTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10

        self.path_to_fixed = example_data("segmentation0.nii.gz")
        self.path_to_moving = example_data("segmentation1.nii.gz")

        self.fixed_sitk = sitk.ReadImage(self.path_to_fixed)
        self.moving_sitk = sitk.ReadImage(self.path_to_moving)

    ##
    # Test whether FSL installation was successful
    # \date       2017-10-26 15:02:44+0100
    #
    def test_fsl(self):

        # Run flirt registration
        registration_method = simplereg.flirt.FLIRT(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk)
        registration_method.run()

    ##
    # Test whether NiftyReg installation was successful
    # \date       2017-10-26 15:08:59+0100
    #
    def test_niftyreg(self):

        # Run reg_aladin registration
        registration_method = simplereg.niftyreg.RegAladin(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk)
        registration_method.run()

        # Run reg_f3d registration
        registration_method = simplereg.niftyreg.RegF3D(
            fixed_sitk=self.fixed_sitk,
            moving_sitk=self.moving_sitk)
        registration_method.run()

    ##
    # Test whether ITK_NiftyMIC installation was successful
    # \date       2017-10-26 15:12:26+0100
    #
    def test_itk_niftymic(self):

        import itk
        image_itk = itk.Image.D3.New()
        filter_itk = itk.OrientedGaussianInterpolateImageFilter.ID3ID3.New()
