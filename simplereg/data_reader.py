# \file DataReader.py
#  \brief Class to read data
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018


import os
import sys
import itk
import numpy as np
import nibabel as nib
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import ALLOWED_IMAGES
from simplereg.definitions import ALLOWED_LANDMARKS
from simplereg.definitions import ALLOWED_TRANSFORMS
from simplereg.definitions import ALLOWED_TRANSFORMS_DISPLACEMENTS


class DataReader(object):

    ##
    # Reads an image and returns either an sitk.Image or itk.Image object.
    # \date       2018-06-23 16:25:53-0600
    #
    # \param      path_to_file  The path to file
    # \param      itk           Select between sitk.Image or itk.Image object; bool
    #
    # \return     Image as sitk.Image or itk.Image object
    #
    @staticmethod
    def read_image(path_to_file, as_itk=0):

        if not ph.file_exists(path_to_file):
            raise IOError("Image file '%s' not found" % path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_IMAGES:
            raise IOError("Image file extension must be of type %s " %
                          ", or ".join(ALLOWED_IMAGES))

        # Read as itk.Image object
        if as_itk:
            image = itk.imread(path_to_file)

        # Read as sitk.Image object
        else:
            image = sitk.ReadImage(path_to_file)

        return image

    @staticmethod
    def read_landmarks(path_to_file):

        if not ph.file_exists(path_to_file):
            raise IOError("Landmark file '%s' not found" % path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_LANDMARKS:
            raise IOError("Landmark file extension must be of type %s " %
                          ", or ".join(ALLOWED_LANDMARKS))

        return np.loadtxt(path_to_file)

    ##
    # Reads a transform.
    # \date       2018-06-12 23:59:38-0600
    #
    # \param      path_to_file  The path to file
    # \param      inverse       The inverse
    # \param      nii_as_nib    State whether NIfTI image should be read as
    #                           nibabel object (only used for sitk_to_nreg
    #                           displacement field conversion); bool
    #
    # \return     Transform as type np.array, sitk.Image or nib.Nifti
    #
    @staticmethod
    def read_transform(path_to_file, inverse=0, nii_as_nib=0, as_itk=0):

        if not ph.file_exists(path_to_file):
            raise IOError("Transform file '%s' not found" % path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_TRANSFORMS and \
                extension not in ALLOWED_TRANSFORMS_DISPLACEMENTS:
            raise IOError("Transform file extension must be of type "
                          "%s (transformation) or %s (displacements)" % (
                              ", ".join(ALLOWED_TRANSFORMS),
                              ", ".join(ALLOWED_TRANSFORMS_DISPLACEMENTS)))

        if extension in ALLOWED_TRANSFORMS:
            if as_itk:
                tranform_sitk = sitk.read_transform_itk(
                    path_to_file, inverse=inverse)
            else:
                transform_sitk = sitkh.read_transform_sitk(
                    path_to_file, inverse=inverse)
        else:
            # Used for sitk_to_nreg conversion only
            if nii_as_nib:
                displacement_sitk = nib.load(path_to_file)
                return displacement_sitk
            else:
                displacement_sitk = sitk.ReadImage(
                    path_to_file, sitk.sitkVectorFloat64)
                transform_sitk = sitk.DisplacementFieldTransform(
                    sitk.Image(displacement_sitk))
                if inverse:
                    # May throw RuntimeError
                    transform_sitk = transform_sitk.GetInverse()

        return transform_sitk

    @staticmethod
    def read_transform_nreg(path_to_file):

        if not ph.file_exists(path_to_file):
            raise IOError("NiftyReg transform file '%s' not found" %
                          path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_TRANSFORMS and \
                extension not in ALLOWED_TRANSFORMS_DISPLACEMENTS:
            raise IOError("NiftyReg transform file extension must be of type "
                          "%s (reg_aladin) or %s (reg_f3d displacement)" % (
                              ", ".join(ALLOWED_TRANSFORMS),
                              ", ".join(ALLOWED_TRANSFORMS_DISPLACEMENTS)))
        if extension in ALLOWED_TRANSFORMS:
            transform_nreg = np.loadtxt(path_to_file)
        else:
            transform_nreg = nib.load(path_to_file)

            # check that image is a NiftyReg displacement field
            header = transform_nreg.get_header()
            if int(header['intent_p1']) != 1 or \
                    int(header['intent_p2']) != 0 or \
                    int(header['intent_p3']) != 0 or \
                    int(header['intent_code']) != 1007:
                raise IOError(
                    "Provided image must represent a NiftyReg "
                    "displacement field")

        return transform_nreg

    @staticmethod
    def read_transform_flirt(path_to_file):

        if not ph.file_exists(path_to_file):
            raise IOError("FLIRT transform file '%s' not found" %
                          path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_TRANSFORMS:
            raise IOError(
                "FLIRT transform file extension must be of type %s" %
                ", or ".join(ALLOWED_TRANSFORMS))

        return np.loadtxt(path_to_file)
