# \file DataWriter.py
#  \brief Class to read data
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018


import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import ALLOWED_IMAGES
from simplereg.definitions import ALLOWED_LANDMARKS
from simplereg.definitions import ALLOWED_TRANSFORMS
from simplereg.definitions import ALLOWED_TRANSFORMS_DISPLACEMENTS


class DataWriter(object):

    @staticmethod
    def write_image(image_sitk, path_to_file, verbose=0):

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_IMAGES:
            raise IOError("Image file extension must be of type %s " %
                          ", or ".join(ALLOWED_IMAGES))
        if isinstance(image_sitk, sitk.Image):
            sitkh.write_nifti_image_sitk(
                image_sitk=image_sitk,
                path_to_file=path_to_file,
                verbose=verbose)
        else:
            sitkh.write_nifti_image_itk(
                image_itk=image_sitk,
                path_to_file=path_to_file,
                verbose=verbose)

    @staticmethod
    def write_landmarks(landmarks_nda, path_to_file, verbose=0):

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_LANDMARKS:
            raise IOError("Landmark file extension must be of type %s " %
                          ", or ".join(ALLOWED_LANDMARKS))

        ph.write_array_to_file(
            path_to_file, landmarks_nda, delimiter=" ", access_mode="w",
            verbose=verbose)

    @staticmethod
    def write_transform(transform_sitk, path_to_file, verbose=0):

        extension = ph.strip_filename_extension(path_to_file)[1]
        if extension not in ALLOWED_TRANSFORMS and \
                extension not in ALLOWED_TRANSFORMS_DISPLACEMENTS:
            raise IOError("Transform file extension must be of type "
                          "%s (transformation) or %s (displacements)" % (
                              ", ".join(ALLOWED_TRANSFORMS),
                              ", ".join(ALLOWED_TRANSFORMS_DISPLACEMENTS)))

        if extension in ALLOWED_TRANSFORMS:
            if isinstance(transform_sitk, sitk.Image):
                raise IOError("Cannot convert displacement field (%s) to "
                              "transform (%s)" % (
                                  ", ".join(ALLOWED_TRANSFORMS_DISPLACEMENTS),
                                  ", ".join(ALLOWED_TRANSFORMS),
                              ))

            if isinstance(transform_sitk, sitk.Transform):
                ph.create_directory(os.path.dirname(path_to_file))
                sitk.WriteTransform(transform_sitk, path_to_file)
                if verbose:
                    ph.print_info("Transform written to '%s'" % path_to_file)
            elif isinstance(transform_sitk, np.ndarray):
                ph.write_array_to_file(
                    path_to_file,
                    transform_sitk,
                    delimiter=" ",
                    access_mode="w",
                    verbose=verbose)
            else:
                raise IOError("Transform must be of type "
                              "sitk.Transform or np.ndarray")
        else:
            if isinstance(transform_sitk, sitk.Transform):
                raise IOError("Cannot convert transform (%s) to "
                              "displacement field (%s)" % (
                                  ", ".join(ALLOWED_TRANSFORMS),
                                  ", ".join(ALLOWED_TRANSFORMS_DISPLACEMENTS),
                              ))
            elif isinstance(transform_sitk, sitk.Image):
                sitkh.write_nifti_image_sitk(
                    image_sitk=transform_sitk,
                    path_to_file=path_to_file,
                    verbose=verbose)
            elif isinstance(transform_sitk, nib.nifti1.Nifti1Image):
                ph.create_directory(os.path.dirname(path_to_file))
                nib.save(transform_sitk, path_to_file)
            else:
                raise IOError("Transform must be of type "
                              "sitk.Image or nibabel.nifti1.Nifti1Image")
