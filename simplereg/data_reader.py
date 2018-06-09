# \file DataReader.py
#  \brief Class to read data
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018


import os
import sys
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import ALLOWED_IMAGES
from simplereg.definitions import ALLOWED_LANDMARKS
from simplereg.definitions import ALLOWED_TRANSFORMS
from simplereg.definitions import ALLOWED_TRANSFORMS_DISPLACEMENTS


class DataReader(object):

    @staticmethod
    def read_image(path_to_file):

        if not ph.file_exists(path_to_file):
            raise IOError("Image file '%s' not found" % path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_IMAGES:
            raise IOError("Image file extension must be of type %s " %
                          ", or ".join(ALLOWED_IMAGES))

        return sitk.ReadImage(path_to_file)

    @staticmethod
    def read_landmarks(path_to_file):

        if not ph.file_exists(path_to_file):
            raise IOError("Landmark file '%s' not found" % path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_LANDMARKS:
            raise IOError("Landmark file extension must be of type %s " %
                          ", or ".join(ALLOWED_LANDMARKS))

        return np.loadtxt(path_to_file)

    @staticmethod
    def read_transform(path_to_file, inverse=0):

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
            transform_sitk = sitkh.read_transform_sitk(
                path_to_file, inverse=inverse)
        else:
            displacement_sitk = sitk.ReadImage(path_to_file)
            transform_sitk = sitk.DisplacementFieldTransform(
                sitk.Image(displacement_sitk))
            if inverse:
                # May throw RuntimeError
                transform_sitk = transform_sitk.GetInverse()

        return transform_sitk

    @staticmethod
    def read_transform_nreg(path_to_file):

        if not ph.file_exists(path_to_file):
            raise IOError("RegAladin transform file '%s' not found" %
                          path_to_file)

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_TRANSFORMS:
            raise IOError(
                "RegAladin transform file extension must be of type %s" %
                ", or ".join(ALLOWED_TRANSFORMS))

        return np.loadtxt(path_to_file)
