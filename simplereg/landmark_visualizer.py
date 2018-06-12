##
# \file landmark_visualizer.py
# \brief      Class to create image mask from landmark coordinates. Landmarks
#             can also be embedded in image.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       June 2018
#

import os
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import skimage.measure

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Class to create image mask from landmark coordinates. Landmarks can also be
# embedded in image.
# \date       2018-06-08 16:54:56-0600
#
class LandmarkVisualizer(object):

    def __init__(self, landmarks_nda, direction, origin, spacing, size):
        self._landmarks_nda = landmarks_nda
        self._direction = direction
        self._origin = origin
        self._spacing = spacing
        self._size = size

        self._landmark_image_sitk = None

        self._get_pattern = {
            "hollow_sphere": self._get_pattern_hollow_sphere,
            "sphere": self._get_pattern_sphere,
            "cross": self._get_pattern_cross,
            "plus": self._get_pattern_plus,
        }

    def set_landmarks_nda(self, landmarks_nda):
        self._landmarks_nda = landmarks_nda

    def get_image_sitk(self):
        return sitk.Image(self._landmark_image_sitk)

    def get_image_nda(self):
        return np.array(self._landmark_image_nda)

    def build_landmark_image_sitk(self, pattern="plus"):
        nda = np.zeros(self._size[::-1], dtype=np.uint8)

        foo = sitk.GetImageFromArray(nda)
        foo.SetSpacing(self._spacing)
        foo.SetDirection(self._direction)
        foo.SetOrigin(self._origin)

        for i in range(self._landmarks_nda.shape[0]):
            landmark = self._landmarks_nda[i, :]
            index = foo.TransformPhysicalPointToIndex(landmark)[::-1]

            if pattern == "dot":
                nda[index] = 1
            else:
                pattern_ = self._get_pattern[pattern](2, 0.5)
                nda = self._apply_pattern(nda, index, pattern_, value=1)

        self._landmark_image_sitk = sitk.GetImageFromArray(nda)
        self._landmark_image_sitk.SetSpacing(self._spacing)
        self._landmark_image_sitk.SetDirection(self._direction)
        self._landmark_image_sitk.SetOrigin(self._origin)

        self._landmark_image_nda = nda

    def annotate_landmarks_on_image_sitk(self, image_sitk):
        val = 0

        image_nda = sitk.GetArrayFromImage(image_sitk)
        indices = np.where(self._landmark_image_nda == 1)
        image_nda[indices] = val

        image_landmarks_sitk = sitk.GetImageFromArray(image_nda)
        image_landmarks_sitk.CopyInformation(image_sitk)

        return sitk.Image(image_landmarks_sitk)

    @staticmethod
    def _get_pattern_hollow_sphere(radius, thickness=1):
        a = radius + np.ceil(thickness)
        x = np.linspace(-a, a, 2 * a + 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        pattern = np.zeros_like(xx)
        values = xx**2 + yy**2 + zz**2
        pattern[values <= (radius + thickness)**2] = 1
        pattern[values < (radius - thickness)**2] = 0
        return pattern

    @staticmethod
    def _get_pattern_sphere(radius, thickness=1):
        a = radius + np.ceil(thickness)
        x = np.linspace(-a, a, 2 * a + 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        pattern = np.zeros_like(xx)
        values = xx**2 + yy**2 + zz**2
        pattern[values <= (radius + thickness)**2] = 1
        return pattern

    @staticmethod
    def _get_pattern_cross(radius, thickness):
        x = np.arange(2 * radius + 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        pattern = np.zeros_like(xx)
        n = pattern.shape[0] - 1
        for i in range(n + 1):
            pattern[i, i, i] = 1
            pattern[i, i, n - i] = 1
            pattern[i, n - i, i] = 1
            pattern[i, n - i, n - i] = 1
        return pattern

    @staticmethod
    def _get_pattern_plus(radius, thickness):
        x = np.arange(2 * radius + 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        pattern = np.zeros_like(xx)
        n = pattern.shape[0] - 1
        for i in range(n + 1):
            pattern[i, radius, radius] = 1
            pattern[radius, i, radius] = 1
            pattern[radius, radius, i] = 1
        return pattern

    @staticmethod
    def _apply_pattern(nda, index, pattern, value):
        a = int((pattern.shape[0] - 1) / 2)
        index = np.array(index)

        indices = np.array(np.where(pattern == 1)) - a
        indices = tuple(indices + index[..., np.newaxis])
        nda[indices] = value

        return nda
